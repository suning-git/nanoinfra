"""
pretrain.py — stage 1: train the champion.

A thin driver, on purpose: it does NOT reimplement training. It invokes the
blessed text Orchestrator (modalities.text.train_text) with THIS project's
recipe (spec.py). Training runs as a subprocess — the normal way to launch it:
GPU-pinned, detachable, torchrun-able for multi-GPU. To read the full
assemble -> train flow, open modalities/text/train_text.py (a maintained
building block); this file only chooses knobs and launches.

Run (single GPU):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python exemplars/text_pretrain/pretrain.py

The checkpoint lands in spec.ckpt_dir(); inference.py and scaling.py read from it.
"""
import subprocess
import sys

import spec


def main():
    overrides = spec.train_overrides(
        # stage-1 concerns (the champion persists; scaling cells will not):
        **{
            "checkpoint.enabled": "true",
            "checkpoint.save_dir": spec.ckpt_dir(),
            "checkpoint.save_every": 2500,
            "checkpoint.keep_last_n": 2,
            "evaluation.text.interval_steps": 500,
            "evaluation.text.eval_tokens": 2097152,
            "logging.log_every": 10,
        }
    )
    cmd = [sys.executable, "-u", "-m", spec.ORCHESTRATOR, *overrides]
    print("launching:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
