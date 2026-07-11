"""run.py — train both arms (baseline GPT vs residual-free) and collect their
val-loss trajectories. Same model / data / budget; the ONLY difference is the
trunk. Mirrors the exemplar's scaling.py driving pattern: subprocess the blessed
orchestrator, parse the scheduled val evals straight from its log.

Needs a FineWeb shard on disk (outputs/base_data/) — fetch a couple first:
    python exemplars/text_pretrain/data/download_shards.py

    python projects/example_residual_ablation/run.py      # trains both arms
    python projects/example_residual_ablation/plot.py     # -> residual_ablation.png
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import spec

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent           # repo root — the orchestrator subprocess runs here
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    """~n log-spaced integer steps in [first, max_steps] (deduped, sorted)."""
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def run_arm(label, trunk_class, max_steps, steps):
    ov = spec.train_overrides(trunk_class, max_steps, steps)
    print(f"[run ] {label}: d{spec.DEPTH} -> {max_steps} steps ...", flush=True)
    out = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                         cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                         capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "val": float(v)} for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        raise SystemExit(f"arm {label} FAILED (rc={out.returncode}, {len(traj)} evals):\n{text[-3000:]}")
    print(f"[done] {label}: {len(traj)} evals, val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}",
          flush=True)
    return {"arm": label, "trajectory": traj}


def main():
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    arms = [run_arm(label, tc, max_steps, steps) for label, tc in spec.ARMS]
    out = {"depth": spec.DEPTH, "max_steps": max_steps, "arms": arms}
    (RESULTS / "curves.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE {RESULTS / 'curves.json'}")


if __name__ == "__main__":
    main()
