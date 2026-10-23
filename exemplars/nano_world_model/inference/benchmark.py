"""
benchmark.py — does the sampler compute what training scored, at full size, and how fast does it generate?

Two questions, in that order. A sampler that is fast and subtly wrong is worse than a
slow one, so agreement is checked first and speed is only reported after.

    python -m exemplars.nano_world_model.inference.benchmark
    python -m exemplars.nano_world_model.inference.benchmark --ckpt <dir> --steps 16

Without --ckpt it runs on a randomly initialised model — every weight random, since
core's init zeroes the output projections and the head, under which every logit is equal
and agreement means nothing. That is not a shortcut: both questions are about which
positions each forward sees and what the kernels cost, not about what was learned.

WHAT IS CHECKED. tests/test_sampler.py checks, in float32 on the CPU, that the sampler's
block logits equal the training forward's to rounding. Here the same comparison runs at
the real geometry under bf16 autocast, the way generation actually runs, where the two
paths use different kernels (SDPA over a KV cache, flex over the two-stream row) and
agree only to bf16 noise: per block, the largest logit difference and the share of
masked cells whose most likely code is the same, with the top1-top2 margin wherever it is
not — near-zero margins are near-ties bf16 can flip either way, large ones would be a bug.
"""

import argparse
import time

import torch

from exemplars.nano_world_model import spec

spec.pin_tokenizer()

from core.data.layout import collate_samples, padding_layout      # noqa: E402
from core.model.gpt import GPT, GPTConfig                          # noqa: E402
from core.training.model_setup import build_system, load_system    # noqa: E402

from exemplars.nano_world_model import train_wm                    # noqa: E402
from exemplars.nano_world_model.block_diffusion import (           # noqa: E402
    BlockDiffusionSystem, two_stream_sample)
from exemplars.nano_world_model.inference.sampler import Sampler   # noqa: E402
from exemplars.nano_world_model.rope3d import install_rope3d       # noqa: E402


@torch.no_grad()
def training_logits(system, layout, rows, row, mask_id, pads, k):
    """Block k's video-band logits from the training forward, the row as the sampler holds it
    (tests/test_sampler.py)."""
    s, e = rows.spans[k - 1]
    masked = torch.zeros(rows.row_len, dtype=torch.bool)
    masked[s:e] = row[s:e] == mask_id
    clean = row.clone()
    clean[masked] = rows.v_off
    sample = two_stream_sample(clean, masked, masked.float(), rows, layout, mask_id, {})
    shape = (1, 2 * rows.row_len)
    batch = collate_samples([sample], padding_layout([shape[1]], shape), shape=shape, pad_values=pads)
    batch = {key: v.cuda() if torch.is_tensor(v) else v for key, v in batch.items()}
    idx, kwargs = system.prepare_trunk_inputs(batch)
    h = system.trunk(idx, **kwargs)[0, rows.row_len + s:rows.row_len + e]
    lo, hi = layout.ranges[spec.VIDEO_TYPE_ID]
    return system.head(h).float()[:, lo:hi]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", default=None, help="a trained checkpoint (else random init)")
    ap.add_argument("--frames", type=int, default=spec.FRAMES)
    ap.add_argument("--steps", type=int, default=8, help="denoising rounds per latent frame")
    ap.add_argument("--clips", type=int, default=5, help="clips generated for the timing")
    ap.add_argument("--depth", type=int, default=spec.DEPTH)
    ap.add_argument("--dim", type=int, default=spec.DIM)
    ap.add_argument("--n-head", type=int, default=spec.N_HEAD)
    args = ap.parse_args()

    layout, resolver = train_wm.assemble_vocab()
    rows = train_wm.build_row_layout(layout, resolver, spec.shape_contract(args.frames))
    mask_id, pads = resolver.resolve(spec.MASK_SLOT), train_wm.batch_pad_values(layout, resolver)
    if args.ckpt:
        base = load_system(args.ckpt)["system"]
    else:
        cfg = GPTConfig(sequence_len=2 * rows.row_len, vocab_size=layout.vocab_size, n_layer=args.depth,
                        n_head=args.n_head, n_kv_head=args.n_head, n_embd=args.dim, n_token_types=layout.n_token_types)
        # bf16 storage like the --ckpt path (load_system loads in bf16), so both paths benchmark the same arithmetic
        base = build_system(GPT, cfg, seed=0, param_dtype=torch.bfloat16)["system"]
        g_init = torch.Generator(next(base.parameters()).device.type).manual_seed(0)
        with torch.no_grad():
            for p in base.parameters():
                p.normal_(0, 0.02, generator=g_init)
    system = BlockDiffusionSystem(base.trunk, base.head, rows, spec.T_MIN, spec.T_MAX).eval()
    install_rope3d(system.trunk, rows)
    sampler = Sampler(system, layout, rows, mask_id, steps=args.steps)
    g = torch.Generator().manual_seed(0)
    print(f"model: d{system.trunk.config.n_layer} dim{system.trunk.config.n_embd}, {rows}, "
          f"{'checkpoint' if args.ckpt else 'random init'}, {args.steps} rounds per latent frame\n")

    # --- 1. agreement with the training forward ---------------------------------
    with torch.autocast("cuda", torch.bfloat16):
        sampler.start(torch.randint(0, spec.CODEC_VOCAB, (rows.cpf,), generator=g), seed=0)
        for k in range(1, rows.n_blocks + 1):
            sampler.begin(torch.randint(0, spec.N_ACTIONS, (rows.td,), generator=g))
            got = sampler.block_logits()
            want = training_logits(system, layout, rows, sampler.row.cpu(), mask_id, pads, k)
            top = want.topk(2, dim=-1)
            differ = (got.argmax(-1) != top.indices[:, 0]).nonzero().flatten()
            margins = (top.values[differ, 0] - top.values[differ, 1]).tolist()
            print(f"block {k}: max |logit diff| {(got - want).abs().max().item():.4f}, argmax agrees on "
                  f"{rows.cpf - len(differ)}/{rows.cpf}"
                  + (f", top1-top2 margins where not: {[round(m, 4) for m in margins[:8]]}" if len(differ) else ""))
            sampler.denoise()
            sampler.commit()

    # --- 2. speed ------------------------------------------------------------------
    def clip():
        sampler.start(torch.randint(0, spec.CODEC_VOCAB, (rows.cpf,), generator=g), seed=1)
        for _ in range(rows.n_blocks):
            sampler.step(torch.randint(0, spec.N_ACTIONS, (rows.td,), generator=g))

    with torch.autocast("cuda", torch.bfloat16):
        clip()                                                         # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.clips):
            clip()
        torch.cuda.synchronize()
    per_frame = (time.perf_counter() - t0) / (args.clips * rows.n_blocks)
    print(f"\n{per_frame * 1e3:.1f} ms per latent frame ({rows.td} game frames): "
          f"{rows.td / per_frame:.0f} game frames/s, eager trunk, dynamic KV cache")


if __name__ == "__main__":
    main()
