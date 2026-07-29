"""
generate.py — text -> motion through the standard stack, no research shims.

The whole chain on first-class parts:
  shelf codec facade (modalities.motion.tokenizers.<name>.load())
  -> shared vocab assembly (modalities.assembler.build_layout — text|control|motion)
  -> AR system from its self-describing checkpoint (core load_system)
  -> band-masked autoregressive sampling (local loop below)
  -> codec.decode -> rot139 features -> SMPL FK -> GIFs (render.py).

Band-masked sampling: over the SHARED vocab the model could emit any id, but
after `motion_start` only motion-band ids (+ the closing `motion_end`) are
legal — the mask enforces the grammar the recipe trained. This is a naive
full-recompute loop; the KV-cached banded path is a core inference-engine
milestone, and 6 prompts x ~250 tokens do not need it.

The seed is the prompt index, so a rerun reproduces the same samples — motion
quality varies enough between seeds that an un-seeded gallery says more about luck
than about the model.

Run (one GPU), after train_t2m.py has produced a checkpoint:

  python -m exemplars.nano_motion.generate --ckpt exemplars/nano_motion/models/<run>/step_XXXXX
  python -m exemplars.nano_motion.generate --ckpt ... --prompts "A person waves."
"""

import argparse
import importlib
import os

import numpy as np
import torch

from exemplars.nano_motion import spec

spec.pin_tokenizer()

import modalities.control                     # noqa: E402
import modalities.motion                      # noqa: E402
import modalities.text                        # noqa: E402
from modalities.assembler import build_layout  # noqa: E402
from modalities.control import make_control_resolver  # noqa: E402
from core.training.model_setup import load_system     # noqa: E402

from exemplars.nano_motion import render      # noqa: E402

MOTION_TYPE = modalities.motion.TYPE_ID  # = 1 (the fossil layout's motion band)


def assemble_vocab(codec):
    """The shared [text | control | motion] vocab the model was trained on.
    List order = band order; the layout derives every offset from the manifests."""
    tok = modalities.text.get_tokenizer()
    text = modalities.text.manifest(tok)
    control = modalities.control.manifest()
    motion = modalities.motion.manifest(codec)
    layout = build_layout([text, control, motion])
    resolver = make_control_resolver(control, layout)
    return layout, resolver, tok


@torch.no_grad()
def sample_motion(system, layout, prefix_ids, stop_id, *, seq_len, temperature,
                  top_k, seed, device="cuda"):
    """Autoregress motion-band ids after the prefix until stop_id -> LOCAL codes."""
    system.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    lo, hi = layout.ranges[MOTION_TYPE]
    seq = list(prefix_ids)
    out = []
    for _ in range(seq_len - len(seq)):
        toks = torch.tensor([seq], dtype=torch.long, device=device)
        types = layout.classify_token_types(toks)
        logits = system.head(system.trunk(toks, token_types=types))[0, -1]
        mask = torch.full_like(logits, float("-inf"))
        mask[lo:hi] = logits[lo:hi]
        if stop_id is not None:
            mask[stop_id] = logits[stop_id]
        logits = mask / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, int((logits > float("-inf")).sum())))
            logits[logits < v[-1]] = float("-inf")
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))
        if stop_id is not None and nxt == stop_id:
            break
        seq.append(nxt)
        out.append(nxt - lo)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="an AR checkpoint from train_t2m.py")
    ap.add_argument("--tokenizer", default=spec.TOKENIZER,
                    help="the codec the checkpoint was trained against")
    ap.add_argument("--codec", default=None, help="a .pt, instead of a shelf name")
    ap.add_argument("--prompts", nargs="*", default=None, help="default: spec.PROMPTS")
    ap.add_argument("--out", default=None, help="output dir (default results/<run>)")
    args = ap.parse_args()
    prompts = args.prompts or spec.PROMPTS

    if args.codec:
        from modalities.motion.tokenizers._convae import MotionCodec
        codec = MotionCodec(args.codec, device="cuda")
    else:
        codec = importlib.import_module(
            f"modalities.motion.tokenizers.{args.tokenizer}").load(device="cuda")
    layout, resolver, text_tok = assemble_vocab(codec)

    setup = load_system(args.ckpt, sequence_len=spec.SEQ_LEN)
    system = setup["system"]
    # A vocabulary mismatch here means the checkpoint was trained against a
    # different codec or a different text tokenizer. Every band offset would be
    # wrong, so the sampled ids would decode as unrelated motion rather than fail.
    assert setup["gpt_config"].vocab_size == layout.vocab_size, (
        f"checkpoint vocab {setup['gpt_config'].vocab_size} != assembled "
        f"{layout.vocab_size} — wrong codec or wrong text tokenizer")

    # the t2m recipe's prefix and closing tag:
    #   [bos, text_start, <caption>, text_end, motion_start] ... motion_end
    tags = {n: resolver.resolve(n)
            for n in ("bos", "text_start", "text_end", "motion_start", "motion_end")}

    out_dir = args.out or str(spec.RESULTS / os.path.basename(args.ckpt.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    samples = []
    for i, prompt in enumerate(prompts):
        prefix = [tags["bos"], tags["text_start"],
                  *text_tok.encode(prompt)[:spec.MAX_TEXT_TOKENS],
                  tags["text_end"], tags["motion_start"]]
        codes = sample_motion(system, layout, prefix, tags["motion_end"],
                              seq_len=spec.SEQ_LEN, temperature=spec.TEMPERATURE,
                              top_k=spec.TOP_K, seed=i)
        feats = codec.decode(np.asarray(codes, dtype=np.int64)) if len(codes) >= 2 else None
        samples.append((prompt, feats))
        n = 0 if feats is None else len(feats)
        print(f"  [{i}] '{prompt[:50]}' -> {len(codes)} codes / {n} frames")

    render.save_grid(samples, os.path.join(out_dir, "samples_grid.png"))
    for i, (prompt, feats) in enumerate(samples):
        if feats is None:
            continue
        slug = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
        np.savez(os.path.join(out_dir, f"{i:02d}_{slug}.npz"), features=feats)
        render.features_to_gif(feats, os.path.join(out_dir, f"{i:02d}_{slug}.gif"),
                               title=prompt[:60])
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
