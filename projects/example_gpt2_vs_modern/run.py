"""run.py — train each arm through nanoinfra's orchestrator and collect the val
curves. Self-contained + portable: works both inside a nanoinfra checkout and as a
standalone folder (nanoinfra installed as a library, FineWeb data via
NANOINFRA_BASE_DIR).

    python run.py            # trains every arm in spec.ARMS
    python run.py param_dtype=bf16    # extra key=value overrides go to every arm (bf16: as example_results/)
    python plot.py           # -> the figure

The experiment's OWN directory is put on PYTHONPATH, so `model.trunk_class` names a
LOCAL module (e.g. `gpt2.GPT2Trunk`) — the folder is a drop-in unit you can copy
anywhere.
"""
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import spec

HERE = Path(__file__).resolve().parent
# outputs/, not example_results/: the committed curves are the reference this
# example is compared against; a run must not overwrite them.
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# Anchor on "Step N |" and read fields BY NAME. The orchestrator builds that line
# by joining whatever metrics exist, so neither field order nor the presence of
# any given field is guaranteed — a positional regex silently matches nothing when
# the line changes, and the arm is then reported as FAILED while it trained fine.
STEP_RE = re.compile(r"^Step\s+(\d+)\s+\|\s+(.+)$")


def parse_eval_line(line):
    """'Step 00029 | val/text_ce: 6.46 | val/bpb: 2.11' -> (29, {name: value}) or None."""
    m = STEP_RE.match(line.strip())
    if not m:
        return None
    fields = {}
    for part in m.group(2).split(" | "):
        k, sep, v = part.partition(": ")
        if sep:
            try:
                x = float(v)
            except ValueError:
                continue                  # non-numeric field (a label, a time)
            if math.isfinite(x):          # a diverged run logs nan/inf; float()
                fields[k.strip()] = x     # accepts both, and one would poison the fit

    return (int(m.group(1)), fields) if fields else None


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


def run_arm(label, trunk_class, max_steps, steps, env, cwd, extra=()):
    ov = [*spec.train_overrides(trunk_class, max_steps, steps), *extra]
    log = OUT / f"arm_{label}.log"
    print(f"[run ] {label}: d{spec.DEPTH} -> {max_steps} steps "
          f"-> {log.parent.name}/{log.name} (tail -f to watch)", flush=True)
    # The child's output goes straight to a file; this process never reads the
    # stream. Reading it would mean deciding, live, which lines to echo and which
    # to parse — two jobs braided into one loop. Parsing a log is fine; parsing it
    # while relaying it is what was not. Run, then parse the file, once.
    # `-u` keeps the child unbuffered so `tail -f` is live; subprocess.run also
    # cleans the child up if this process dies.
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                            cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    traj = []
    with open(log, errors="replace") as f:
        for line in f:
            parsed = parse_eval_line(line)
            if parsed and "val/text_ce" in parsed[1]:
                traj.append({"step": parsed[0], "val": parsed[1]["val/text_ce"]})
    if rc != 0 or len(traj) < 3:
        raise SystemExit(f"arm {label} FAILED (rc={rc}, {len(traj)} evals) — see {log}")
    print(f"[done] {label}: {len(traj)} evals, val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}", flush=True)
    return {"arm": label, "trajectory": traj}


def main():
    env, cwd = _subprocess_env()
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    extra = sys.argv[1:]                 # key=value overrides for every arm, e.g. param_dtype=bf16
    arms = [run_arm(label, tc, max_steps, steps, env, cwd, extra) for label, tc in spec.ARMS]
    (OUT / "curves.json").write_text(
        json.dumps({"depth": spec.DEPTH, "max_steps": max_steps, "overrides": extra, "arms": arms}, indent=2))
    print(f"WROTE {OUT / 'curves.json'}")


if __name__ == "__main__":
    main()
