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
    .venv/bin/python exemplars/text_pretrain/inference.py
"""
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

import spec


def latest_ckpt():
    """The most-recent checkpoint of THIS project's champion (spec.ckpt_dir())."""
    steps = sorted(glob.glob(os.path.join(spec.ckpt_dir(), "step_*")))
    if not steps:
        raise SystemExit(f"no checkpoint under {spec.ckpt_dir()} — run pretrain.py first")
    return steps[-1]


PROMPTS = [
    "The history of the Roman Empire",
    "In a surprising turn of events, scientists have discovered",
    "The best way to learn a new language is",
    "Once upon a time, in a small village",
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "samples.md")


def main():
    device = "cuda"
    tok = get_tokenizer()
    text = modalities.text.manifest(tok)
    control = modalities.control.manifest()
    layout = build_layout([text, control])
    resolver = make_control_resolver(control, layout)
    bos, eos = resolver.resolve("bos"), resolver.resolve("eos")

    ckpt = latest_ckpt()
    print(f"loading champion: {ckpt}")
    # blueprint from the artifact; a short context is an instantiation choice
    setup = load_system(ckpt, sequence_len=256)
    system = setup["system"]
    assert setup["gpt_config"].vocab_size == layout.vocab_size, \
        "checkpoint vocab != assembled layout — wrong tokenizer world?"

    # batched prefill needs same-length prompts -> left-pad with bos to the max
    enc = [[bos] + tok.encode(p) for p in PROMPTS]
    T = max(len(e) for e in enc)
    prompt = torch.tensor([[bos] * (T - len(e)) + e for e in enc], device=device)
    ptypes = layout.classify_token_types(prompt.cpu()).to(device)

    g = torch.Generator(device=device).manual_seed(0)
    gen = autoregressive_generate(
        system, prompt, ptypes, max_new_tokens=80, gen_token_type=0,
        stop_token=eos, temperature=0.8, top_k=40, generator=g)

    lines = [f"# text_pretrain — inference samples (champion d{spec.DEPTH}, lr {spec.LR_MAX})\n",
             f"d{spec.DEPTH} champion, sampling (temp 0.8, top-k 40) through the core "
             "KV-cache engine. Prompt in **bold**, continuation plain.\n"]
    for p, row in zip(PROMPTS, gen.tolist()):
        toks = [t for t in row if t != eos]
        cont = tok.decode(toks)
        lines.append(f"> **{p}**{cont}\n")
        print(f"[{p}] -> {cont[:120]}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
