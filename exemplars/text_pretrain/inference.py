"""
inference.py — stage 3: sample text from the trained champion.

Shows the trained model actually runs as a language model — continuations through
the CORE inference engine (core.model.inference.autoregressive_generate + KV
cache). The checkpoint is SELF-DESCRIBING: core's load_system reads the trained
architecture from the checkpoint's own meta.json, assembles the System, and
fills + validates the weights — no geometry is re-derived here. The
[text, control] vocab is assembled exactly as in training.

Run (repo root):
  CUDA_VISIBLE_DEVICES=0 NANOINFRA_BASE_DIR=$PWD/outputs \
    .venv/bin/python -m exemplars.text_pretrain.inference                  # after a single-GPU pretrain
  ... -m exemplars.text_pretrain.inference --parallel ddp                   # after pretrain --nproc N --parallel ddp
"""
import argparse
import glob
import os

import torch

from core.model.inference import autoregressive_generate
from core.training.model_setup import load_system

import modalities.text
import modalities.control
from modalities.assembler import build_layout
from modalities.control import make_control_resolver
from modalities.text import get_tokenizer

from exemplars.text_pretrain import spec


def latest_ckpt(parallel=None):
    """The most-recent checkpoint of THIS project's champion (spec.ckpt_dir(); a run launched with
    `pretrain.py --parallel X` saves under spec.ckpt_dir(parallel=X), so pass the same --parallel here)."""
    root = spec.ckpt_dir(parallel=parallel)
    steps = sorted(glob.glob(os.path.join(root, "step_*")))
    if not steps:
        raise SystemExit(f"no checkpoint under {root} — run pretrain.py first "
                         f"(a run launched with --parallel saves under that name: pass the same --parallel here)")
    return steps[-1]


PROMPTS = [
    "The history of the Roman Empire",
    "In a surprising turn of events, scientists have discovered",
    "The best way to learn a new language is",
    "Once upon a time, in a small village",
]
# outputs/, not example_results/: the committed samples.md is what this exemplar
# claims it produces, and a run must not overwrite the thing it is checked against.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "samples.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", default=None, choices=["ddp", "fsdp"],
                    help="read the champion a `pretrain.py --parallel` run saved (default: the path a run without "
                         "--parallel saves to)")
    args = ap.parse_args()
    device = "cuda"
    tok = get_tokenizer()
    text = modalities.text.manifest(tok)
    control = modalities.control.manifest()
    layout = build_layout([text, control])
    resolver = make_control_resolver(control, layout)
    bos, text_start = resolver.resolve("bos"), resolver.resolve("text_start")

    ckpt = latest_ckpt(args.parallel)
    print(f"loading champion: {ckpt}")
    # blueprint from the artifact; a short context is an instantiation choice
    setup = load_system(ckpt, sequence_len=256)
    system = setup["system"]
    assert setup["gpt_config"].vocab_size == layout.vocab_size, \
        "checkpoint vocab != assembled layout — wrong tokenizer world?"

    lines = [f"# text_pretrain — inference samples (champion d{spec.DEPTH}, lr {spec.LR_MAX})\n",
             f"d{spec.DEPTH} champion, sampling (temp 0.8, top-k 40) through the core "
             "KV-cache engine. Prompt in **bold**, continuation plain.\n"]
    # One prompt at a time, framed the way training opens every row — bos, text_start, then text — and stopped at
    # the next bos, which is how the training stream marks where a document ends (text_end only closes a row, at a
    # fixed position, whatever the text). Prompts of different lengths cannot share a batched prefill without left
    # padding, and this vocabulary has no padding token: padding with bos puts several bos in a row, which no training
    # row contains, and the short prompts' continuations derail.
    for p in PROMPTS:
        prompt = torch.tensor([[bos, text_start] + tok.encode(p)], device=device)
        ptypes = layout.classify_token_types(prompt.cpu()).to(device)
        g = torch.Generator(device=device).manual_seed(0)
        gen = autoregressive_generate(
            system, prompt, ptypes, max_new_tokens=80, gen_token_type=0,
            stop_token=bos, temperature=0.8, top_k=40, generator=g)
        toks = [t for t in gen[0].tolist() if t != bos]
        cont = tok.decode(toks)
        lines.append(f"> **{p}**{cont}\n")
        print(f"[{p}] -> {cont[:120]}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
