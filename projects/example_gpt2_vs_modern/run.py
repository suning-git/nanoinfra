"""run.py — train each arm through nanoinfra's orchestrator and collect the val
curves. Self-contained + portable: works both inside a nanoinfra checkout and as a
standalone folder (nanoinfra installed as a library, FineWeb data via
NANOINFRA_BASE_DIR).

    python run.py            # trains every arm in spec.ARMS
    python plot.py           # -> the figure

The experiment's OWN directory is put on PYTHONPATH, so `model.trunk_class` names a
LOCAL module (e.g. `gpt2.GPT2Trunk`) — the folder is a drop-in unit you can copy
anywhere.
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
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def _nanoinfra_checkout(start):
    """The nanoinfra source tree (a dir holding core/ + modalities/) if we're running
    inside one; None when nanoinfra is only pip-installed."""
    p = start
    while p != p.parent:
        if (p / "core").is_dir() and (p / "modalities").is_dir():
            return p
        p = p.parent
    return None


def _subprocess_env():
    """PYTHONPATH = this experiment's dir (for its local trunk module) + the nanoinfra
    checkout if present; NANOINFRA_BASE_DIR points the orchestrator at the FineWeb data."""
    nano = _nanoinfra_checkout(HERE)
    parts = [str(HERE)] + ([str(nano)] if nano else [])
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(parts)}
    env.setdefault("NANOINFRA_BASE_DIR", str(nano / "outputs") if nano else "./outputs")
    cwd = str(nano) if nano else str(HERE)
    return env, cwd


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def run_arm(label, trunk_class, max_steps, steps, env, cwd):
    ov = spec.train_overrides(trunk_class, max_steps, steps)
    print(f"[run ] {label}: d{spec.DEPTH} -> {max_steps} steps ...", flush=True)
    out = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                         cwd=cwd, env=env, capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "val": float(v)} for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        raise SystemExit(f"arm {label} FAILED (rc={out.returncode}, {len(traj)} evals):\n{text[-3000:]}")
    print(f"[done] {label}: {len(traj)} evals, val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}", flush=True)
    return {"arm": label, "trajectory": traj}


def main():
    env, cwd = _subprocess_env()
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    arms = [run_arm(label, tc, max_steps, steps, env, cwd) for label, tc in spec.ARMS]
    (RESULTS / "curves.json").write_text(
        json.dumps({"depth": spec.DEPTH, "max_steps": max_steps, "arms": arms}, indent=2))
    print(f"WROTE {RESULTS / 'curves.json'}")


if __name__ == "__main__":
    main()
