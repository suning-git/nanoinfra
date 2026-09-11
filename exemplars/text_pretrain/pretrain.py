"""
pretrain.py — stage 1: train the champion, on one GPU or several.

A thin driver, on purpose: it does NOT reimplement training. It invokes the
blessed text Orchestrator (modalities.text.train_text) with THIS project's
recipe (spec.py). Training runs as a subprocess — the normal way to launch it:
GPU-pinned, detachable, torchrun-able for multi-GPU. To read the full
assemble -> train flow, open modalities/text/train_text.py (a maintained
building block); this file only chooses knobs and launches.

Whether a run is single-GPU or multi-GPU is decided by the LAUNCHER, not by a
config key: the orchestrator reads the RANK environment variable, and only
torchrun sets it. `--nproc 2` launches torch.distributed.run; `--parallel` then
chooses the placement (ddp replicates the model, and core/parallel/nano_ddp.py
all_reduces gradients during backward; fsdp shards parameters per block — the
orchestrator's default). See the README's multi-GPU section for the measured
numbers on which to pick.

Run (repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m exemplars.text_pretrain.pretrain
  .venv/bin/python -m exemplars.text_pretrain.pretrain --nproc 2 --parallel ddp
  .venv/bin/python -m exemplars.text_pretrain.pretrain --nproc 2 --parallel ddp --smoke 30

Trailing `key=value` arguments pass through to the orchestrator as Hydra
overrides — `total_batch_size` is the interesting one for multi-GPU, because it
sets the gradient-accumulation count and therefore how far the once-per-step
all_reduce is amortized.

The checkpoint lands in spec.ckpt_dir(); inference.py and scaling.py read from it.
"""
import argparse
import os
import subprocess
import sys

from exemplars.text_pretrain import spec


def launch(overrides, nproc):
    """Run the orchestrator, on one process or on `nproc` of them.

    torchrun is invoked as `python -m torch.distributed.run` rather than by
    finding the `torchrun` script, so the launcher inherits THIS interpreter — a
    venv that was not on PATH still gets its own torch.
    """
    if nproc > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc_per_node={nproc}", "--standalone",
               "-m", spec.ORCHESTRATOR, *overrides]
    else:
        cmd = [sys.executable, "-u", "-m", spec.ORCHESTRATOR, *overrides]
    print("launching:", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}   # torchrun has no -u to pass on
    return subprocess.run(cmd, env=env).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nproc", type=int, default=1,
                    help="processes = GPUs (default 1)")
    ap.add_argument("--parallel", default=None, choices=["ddp", "fsdp"],
                    help="multi-GPU placement (default: orchestrator's fsdp); "
                         "ignored at --nproc 1 (one device compiles the whole trunk "
                         "whatever this says)")
    ap.add_argument("--smoke", type=int, default=0, metavar="STEPS",
                    help="short run: STEPS steps, no checkpointing, log every step")
    ap.add_argument("overrides", nargs="*", metavar="key=value",
                    help="extra Hydra overrides passed to the orchestrator")
    args = ap.parse_args()

    if args.smoke:
        # A smoke run proves the loop, not the model: nothing persists, and every
        # step is logged so the throughput line is readable rather than averaged.
        stage = {
            "max_steps": args.smoke,
            "checkpoint.enabled": "false",
            "logging.log_every": 1,
        }
    else:
        # stage-1 concerns (the champion persists; scaling cells will not):
        stage = {
            "checkpoint.enabled": "true",
            "checkpoint.save_dir": spec.ckpt_dir(parallel=args.parallel),
            "checkpoint.save_every": 2500,
            "checkpoint.keep_last_n": 2,
            "evaluation.interval_steps": 500,
            "evaluation.eval_tokens": 2097152,
            "logging.log_every": 10,
        }
    overrides = spec.train_overrides(parallel=args.parallel, **stage) + args.overrides
    raise SystemExit(launch(overrides, args.nproc))


if __name__ == "__main__":
    main()
