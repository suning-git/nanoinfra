"""
scaling.py — the compute-optimal scaling law, on the text line.

Trains one curve per model size at a FIXED token budget, takes the lower envelope
of the per-model curves, and reads the exponent a in N_opt ∝ C^a from it
(Chinchilla ≈ 0.5).

    python -m projects.nano_multimodal.scaling run --depths 8 6     # one GPU
    python -m projects.nano_multimodal.scaling run --depths 4 3 2   # the other
    python -m projects.nano_multimodal.scaling fit

The ladder is embarrassingly parallel — one depth per GPU, no communication — so a
pair of cards finishes it in about an hour. The fitting math is reused from
exemplars/text_pretrain/scaling_fit.py; the study it belongs to is done and
reproduced a ≈ 0.52 on its July curves (0.50 on the curves it ships now), and none of it is reimplemented here.

WHY 500M PER SIZE, NOT 2B. The exemplar uses 2B so that every curve's bend AND its
flattening are visible. Its own comment records that at 500M this pipeline
reproduces the exponent to 0.004 — so the fourth decimal is what the extra 1.5B
buys, while a student is looking at "the curves cross, the envelope has a slope,
the slope is about a half". 500M turns roughly eight GPU-hours into two, which
fits a class. The cut is printed by `fit` with the 2B number beside it: a budget
quietly reduced is a result quietly weakened, and one reduced on the record with
its reason is a lesson about measurement.

WHY CONSTANT LR. Every point along one run has to be a legitimate endpoint, and an
end-of-run warmdown moves the last point by a lot (~0.37 nat on the video line).
Warmup is ABSOLUTE steps for the same reason: a ratio-based warmup re-tunes itself
whenever the budget changes, so the recipe would depend on how far right the line
runs.

WHAT THIS READS. metrics.jsonl, written by metrics.py — not the training log. The
exemplar has to parse its own stdout with a regex anchored on "Step N |", and it
carries a paragraph explaining why that regex is field-name-based rather than
positional. Having the numbers on disk in a machine format removes that whole
class of problem; it is the first payoff of the local metrics sink.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from projects.nano_multimodal import spec

DEPTHS = [2, 3, 4, 6, 8]              # model sizes N (non-embedding params)
SEQ_LEN, DBS, ROWS = 1024, 32, 32     # ROWS * SEQ_LEN = 32768 tokens per step
MAX_TOKENS = 500_000_000              # per size; see the module docstring
EXEMPLAR_TOKENS = 2_000_000_000       # what the full study uses, printed by `fit`
WARMUP_STEPS = 200                    # ABSOLUTE (core's native unit)
N_EVALS = 40                          # log-spaced eval points per curve
EVAL_BATCHES, EVAL_BATCH = 64, 8      # ~512K supervised val tokens per eval point

OUT = spec.SCALING_DIR
TBS = ROWS * SEQ_LEN


def n_nonembed(depth):
    """Non-embedding parameter count for this GPT geometry, mirroring the family
    rule in configs/text.yaml (dim = 64*depth)."""
    dim = depth * 64
    return 12 * depth * dim * dim + 3 * dim


def eval_schedule(max_steps, n=N_EVALS, first=20):
    """~n log-spaced integer steps in [first, max_steps] (deduped, sorted)."""
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def read_curve(metrics_path, N):
    """metrics.jsonl -> the (compute, val) trajectory. One pass, no regex."""
    traj = []
    with open(metrics_path, errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val/text_ce" not in rec:
                continue
            v = rec["val/text_ce"]
            if not np.isfinite(v):       # a diverged run logs nan; one would poison the fit
                continue
            step = int(rec["step"])
            traj.append({"step": step, "tokens": step * TBS,
                         "compute": 6.0 * N * step * TBS, "val": float(v)})
    return traj


def run_curve(depth, max_tokens=MAX_TOKENS, warmup=WARMUP_STEPS):
    """One constant-LR run to max_tokens through THIS project's orchestrator."""
    N = n_nonembed(depth)
    max_steps = int(max_tokens // TBS)
    steps = eval_schedule(max_steps)
    # THE LADDER GETS ITS OWN DIRECTORY, and that is not tidiness. The default
    # save_dir is nmm_<line>_d<depth> — the same name the EXHIBIT run of that depth
    # uses. A ladder rung at depth 6 would therefore unlink the metrics.jsonl of the
    # trained d6 text model, and with it every val reading the inference panel ranks
    # its checkpoints by: nine saved steps would survive as an unlabelled list with
    # no floor and no best. The runs are different experiments (constant LR, a
    # different batch, no warmdown) and must not share a namespace.
    ckpt = OUT / f"d{depth}"
    ckpt.mkdir(parents=True, exist_ok=True)
    metrics = ckpt / "metrics.jsonl"
    if metrics.exists():
        metrics.unlink()               # a fresh curve, not an append to the last one

    ov = [
        "--config-name", "text",
        f"model.depth={depth}",
        f"sequence_len={SEQ_LEN}", f"device_batch_size={DBS}", f"total_batch_rows={ROWS}",
        f"max_steps={max_steps}",
        f"optimizer.scheduler.warmup_steps={warmup}",
        "optimizer.scheduler.warmdown_ratio=0.0",   # constant LR after warmup —
        "optimizer.scheduler.final_lr_frac=1.0",    #   no end-of-run dip
        "checkpoint.enabled=false",
        f"checkpoint.save_dir={ckpt}",   # see above: never the exhibit run's directory
        "evaluation.eval_at=[" + ",".join(map(str, steps)) + "]",
        f"evaluation.n_batches={EVAL_BATCHES}", f"evaluation.batch={EVAL_BATCH}",
        "logging.log_every=200",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"curve_d{depth}.log"
    print(f"[run ] d{depth} N={N/1e6:.1f}M -> {max_tokens/1e6:.0f}M tokens "
          f"({max_steps} steps, {len(steps)} evals) -> {log.name} (tail -f to watch)",
          flush=True)
    # A subprocess rather than an in-process call, deliberately: train.py is a
    # @hydra.main CLI entry, and reaching it any other way would exercise a
    # different code path than the one students run.
    with open(log, "w") as f:
        rc = subprocess.run(
            [sys.executable, "-u", "-m", "projects.nano_multimodal.train", *ov],
            cwd=spec.REPO, env={**os.environ, "PYTHONPATH": str(spec.REPO)},
            stdout=f, stderr=subprocess.STDOUT).returncode
    traj = read_curve(metrics, N) if metrics.exists() else []
    if rc != 0 or len(traj) < 3:
        raise SystemExit(f"curve d{depth} FAILED (rc={rc}, {len(traj)} evals) — see {log}")
    print(f"[done] d{depth}: {len(traj)} points, val {traj[0]['val']:.3f} -> "
          f"{traj[-1]['val']:.3f}", flush=True)
    return {"depth": depth, "N": N, "trajectory": traj}


def cmd_run(args):
    """One curve per depth on this GPU. Per-shard JSON, resumable."""
    OUT.mkdir(parents=True, exist_ok=True)
    shard = OUT / f"curves_{'-'.join(map(str, args.depths))}.json"
    curves = json.loads(shard.read_text())["curves"] if shard.exists() else []
    done = {c["depth"] for c in curves}
    for d in args.depths:
        if d in done:
            print(f"[skip] d{d}", flush=True)
            continue
        curves.append(run_curve(d, args.tokens, args.warmup))
        shard.write_text(json.dumps({"curves": curves}, indent=2))
    print(f"WROTE {shard} ({len(curves)} curves)")


def cmd_fit(args):
    """Merge shards, fit the frontier exponent, write scaling.json + the figure."""
    from exemplars.text_pretrain import scaling_fit

    by_depth = {}
    for f in sorted(glob.glob(str(OUT / "curves_*.json"))):
        for c in json.loads(Path(f).read_text())["curves"]:
            by_depth[c["depth"]] = c
    curves = sorted(by_depth.values(), key=lambda c: c["N"])
    if len(curves) < 3:
        raise SystemExit(f"only {len(curves)} curves found — run the study first")

    a = scaling_fit.frontier_exponent(curves)
    print(f"compute-optimal frontier exponent a = {a:.3f}   (Chinchilla ~0.5)"
          if a else "frontier exponent: undetermined (need >=2 sizes competing)")
    budget = curves[0]["trajectory"][-1]["tokens"]
    print(f"budget: {budget/1e6:.0f}M tokens per size "
          f"(the full study in exemplars/text_pretrain uses "
          f"{EXEMPLAR_TOKENS/1e9:.0f}B and reproduces a to 0.004 — the cut is "
          f"deliberate, see this module's docstring)")

    # A frontier EXISTS only where the curves cross: a small model has to reach its
    # floor while a bigger one is still descending. Under-budget, every curve is
    # still falling, the envelope is just the biggest model's curve, and the fit
    # returns a number that looks like a result and is not one. Say so, loudly,
    # rather than let a smoke run be quoted.
    if budget < MAX_TOKENS * 0.5:
        print(f"\n  ⚠ {budget/1e6:.0f}M per size is far below this study's design "
              f"point ({MAX_TOKENS/1e6:.0f}M). At this budget no curve has "
              f"saturated, so they never cross, the 'frontier' is just the largest "
              f"model's own curve, and a is NOT a measurement of anything. Fine as a "
              f"wiring check; do not quote it.\n")
    _crossings(curves)

    out = {"study": {"depths": [c["depth"] for c in curves], "seq_len": SEQ_LEN,
                     "total_batch_size": TBS, "max_tokens": budget,
                     "lr_schedule": "constant", "eval_points": N_EVALS},
           "a_frontier": a, "curves": curves}
    (OUT / "scaling.json").write_text(json.dumps(out, indent=2))
    _plot(curves, a)
    print(f"wrote {OUT/'scaling.json'} + {OUT/'scaling_law.png'}")


def _crossings(curves):
    """How many times the per-size curves actually cross. The frontier is made of
    crossings; zero of them means there is no frontier to fit, whatever the fit
    returned."""
    n = 0
    for i in range(len(curves) - 1):
        a_t, b_t = curves[i]["trajectory"], curves[i + 1]["trajectory"]
        lo = max(a_t[0]["compute"], b_t[0]["compute"])
        hi = min(a_t[-1]["compute"], b_t[-1]["compute"])
        if hi <= lo:
            continue
        grid = np.logspace(np.log10(lo), np.log10(hi), 64)
        da = np.interp(np.log(grid), np.log([p["compute"] for p in a_t]),
                       [p["val"] for p in a_t])
        db = np.interp(np.log(grid), np.log([p["compute"] for p in b_t]),
                       [p["val"] for p in b_t])
        n += int(np.sum(np.diff(np.sign(da - db)) != 0))
    print(f"  curve crossings between adjacent sizes: {n}"
          + ("  <- a frontier needs at least a few" if n < 2 else ""))


def _plot(curves, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in curves:
        t = c["trajectory"]
        ax.plot([p["compute"] for p in t], [p["val"] for p in t],
                lw=1.2, alpha=0.85, label=f"d{c['depth']} ({c['N']/1e6:.1f}M)")
    from exemplars.text_pretrain import scaling_fit
    fc, fl = scaling_fit.envelope(curves)
    ax.plot(fc, fl, "k--", lw=2, label="compute-optimal frontier")
    ax.set_xscale("log")
    ax.set_xlabel("compute C = 6ND (FLOPs)")
    ax.set_ylabel("val CE (nats/token)")
    ax.set_title(f"nano_multimodal / text — frontier exponent a = {a:.3f}"
                 if a else "nano_multimodal / text")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "scaling_law.png", dpi=140)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train one curve per depth on this GPU")
    r.add_argument("--depths", type=int, nargs="+", required=True)
    r.add_argument("--tokens", type=int, default=MAX_TOKENS)
    # Warmup is ABSOLUTE steps, so a tiny --tokens smoke has to lower it too or the
    # schedule asserts. That is the knob working as intended, not a rough edge: a
    # ratio-based warmup would silently re-tune itself with every budget change.
    r.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("fit", help="merge shards, fit the frontier, plot")
    f.set_defaults(fn=cmd_fit)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
