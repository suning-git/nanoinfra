"""
scaling.py — stage 2: the compute-optimal scaling law (per-model training curves).

Trains ONE run per model size (d2..d8) through the SAME blessed Orchestrator as
pretrain (spec.ORCHESTRATOR) to a fixed token budget at CONSTANT LR, evaluating
at ~40 LOG-SPACED steps along the way. The schedule is computed HERE and injected
via `evaluation.text.eval_at` — core evaluators accept an explicit step set; core
never computes schedules. Plotted as loss vs compute (C = 6·N·D), each size's
curve drops -> bends -> flattens toward its capacity floor; the LOWER ENVELOPE of
all curves is the compute-optimal frontier, from which

    a = slope(log N_opt vs log C) ~ 0.5      (N_opt proportional to C^a; Chinchilla ~0.5)

Deliberate choices:
  * Constant LR (scheduler warmdown off) so each curve reflects genuine
    loss-vs-compute, not an end-of-run warmdown dip.
  * LOG-spaced eval starting at step 20 (~0.66M tokens): the frontier exponent
    is acutely sensitive to the EARLY low-compute segment — sample it too late
    and the smallest and largest sizes never share a compute budget (RESULTS §2).
  * The SAME Orchestrator as pretrain (not a hand-rolled loop) — the scaling law
    characterises the real training loop, and rides the same snapshot guard.
  * ONE shared LR (the champion's 3e-4) across sizes — fine for the frontier's
    shape at this scale; a precision study would tune LR per size.

WHY curves + frontier (not a 2D grid + parametric fit): the frontier IS the
compute-optimal object — at each compute budget several sizes compete and the
envelope picks the single best.

Run — split the depths across both GPUs, then collect + fit + plot:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python exemplars/text_pretrain/scaling.py run --depths 8 6
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python exemplars/text_pretrain/scaling.py run --depths 4 3 2
  .venv/bin/python exemplars/text_pretrain/scaling.py fit   # -> scaling.json + scaling_law.png
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import spec
import scaling_fit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent           # repo root — the orchestrator subprocess runs here
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

DEPTHS = [2, 3, 4, 6, 8]            # model sizes N (non-embedding params)
SEQ_LEN, DBS, TBS = 1024, 32, 32768
MAX_TOKENS = 2_000_000_000          # per size. EVERY size gets the same fixed budget ON
                                    #   PURPOSE: small models saturate to their floor while big
                                    #   ones still descend — that is what makes the curves cross
                                    #   and the frontier exist. 2B puts d8 at ~79 tok/param
                                    #   (4x Chinchilla) and d6 at ~188: every curve's bend AND
                                    #   flattening are VISIBLE. Still single-epoch (<3.63B train
                                    #   tokens). (The pre-refactor study used 500M; at that
                                    #   budget this pipeline reproduces its exponent to 0.004 —
                                    #   RESULTS §2.)
WARMUP_STEPS = 200                  # ABSOLUTE warmup (core's native unit). The recipe must not
                                    #   depend on how far right the line runs — a ratio-based
                                    #   warmup silently re-tunes itself whenever the budget
                                    #   changes. 200 = the pre-refactor study's value.
N_EVALS = 40                        # log-spaced eval points per curve, first at step 20
EVAL_TOKENS = 524288                # 512K val tokens per eval. MEASURED (same trajectory through
                                    #   128K vs 512K windows): per-point jitter ~0.009 either
                                    #   way — trajectory-dominated, so a bigger window buys no
                                    #   smoothness; and a scaling study needs RELATIVE CE only
                                    #   (deterministic window -> bias consistent across curves).

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def n_nonembed(depth):
    """Non-embedding parameter count for this GPT geometry.

    Mirrors the family rule in modalities/text/configs/train_text.yaml
    (dim = 64*depth): 12 blocks-worth of dim^2 per layer + final norm.
    """
    dim = depth * 64
    return 12 * depth * dim * dim + 3 * dim


def eval_schedule(max_steps, n=N_EVALS, first=20):
    """~n log-spaced integer steps in [first, max_steps] (deduped, sorted)."""
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def run_curve(depth):
    """One constant-LR run to MAX_TOKENS through the Orchestrator; return its
    (compute, val) trajectory parsed from the log-scheduled evaluations."""
    N = n_nonembed(depth)
    max_steps = int(MAX_TOKENS // TBS)
    steps = eval_schedule(max_steps)
    ov = spec.train_overrides(depth=depth, **{
        "sequence_len": SEQ_LEN, "device_batch_size": DBS, "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,   # budget-invariant by construction
        "optimizer.scheduler.warmdown_ratio": 0.0,      # constant LR after warmup —
        "optimizer.scheduler.final_lr_frac": 1.0,       #   no end-of-run warmdown dip
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, steps)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 200,
    })
    print(f"[run ] d{depth} N={N/1e6:.1f}M -> {MAX_TOKENS/1e6:.0f}M tokens "
          f"({max_steps} steps, {len(steps)} log-spaced evals) ...", flush=True)
    out = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                         cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                         capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "tokens": int(s) * TBS,
             "compute": 6.0 * N * int(s) * TBS, "val": float(v)}
            for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        raise SystemExit(f"curve d{depth} FAILED (rc={out.returncode}, {len(traj)} evals):\n{text[-2000:]}")
    print(f"[done] d{depth}: {len(traj)} eval points, val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}",
          flush=True)
    return {"depth": depth, "N": N, "trajectory": traj}


def cmd_run(args):
    """Train one curve per depth (one GPU). Per-shard JSON, resumable."""
    shard = RESULTS / f"curves_{'-'.join(map(str, args.depths))}.json"
    curves = json.loads(shard.read_text())["curves"] if shard.exists() else []
    done = {c["depth"] for c in curves}
    for d in args.depths:
        if d in done:
            print(f"[skip] d{d}", flush=True)
            continue
        curves.append(run_curve(d))
        shard.write_text(json.dumps({"curves": curves}, indent=2))
    print(f"WROTE {shard} ({len(curves)} curves)")


def cmd_fit(args):
    """Merge shards, fit the frontier exponent, write scaling.json + the figure."""
    by_depth = {}
    for f in sorted(glob.glob(str(RESULTS / "curves_*.json"))):
        for c in json.loads(Path(f).read_text())["curves"]:
            by_depth[c["depth"]] = c
    if not by_depth and (RESULTS / "scaling.json").exists():   # re-fit/re-plot from committed output
        for c in json.loads((RESULTS / "scaling.json").read_text()).get("curves", []):
            by_depth[c["depth"]] = c
    curves = sorted(by_depth.values(), key=lambda c: c["N"])
    if len(curves) < 3:
        raise SystemExit(f"only {len(curves)} curves found — run the study first")
    a = scaling_fit.frontier_exponent(curves)
    print(f"compute-optimal frontier exponent a = {a:.3f}   (Chinchilla ~0.5)"
          if a else "frontier exponent: undetermined (need >=2 sizes competing)")
    out = {"study": {"depths": DEPTHS, "seq_len": SEQ_LEN, "total_batch_size": TBS,
                     "max_tokens": MAX_TOKENS, "lr_max": spec.LR_MAX, "lr_schedule": "constant",
                     "eval_schedule": f"{N_EVALS} log-spaced from step 20",
                     "eval_tokens": EVAL_TOKENS},
           "a_frontier": a, "curves": curves}
    (RESULTS / "scaling.json").write_text(json.dumps(out, indent=2))
    _plot(curves, a)
    print(f"wrote {RESULTS / 'scaling.json'} + {HERE / 'scaling_law.png'}")


def _plot(curves, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    cmap = plt.cm.viridis
    for i, c in enumerate(curves):
        tr = [p for p in c["trajectory"] if p["compute"] > 0]
        ax.plot([p["compute"] for p in tr], [p["val"] for p in tr], "-o",
                color=cmap(i / max(1, len(curves) - 1)), lw=1.7, ms=4,
                label=f"N = {c['N']/1e6:.2f}M")

    fc, fl = scaling_fit.envelope(curves)
    ax.plot(fc, fl, "k--", lw=2.6, label="compute-optimal frontier", zorder=6)

    # make "at each compute several sizes compete, ONE is optimal" explicit:
    # at a few sample budgets, drop a guide line and star the frontier (optimal) size.
    if fc:
        for C in np.logspace(np.log10(min(fc)) + 0.5, np.log10(max(fc)) - 0.3, 3):
            best = scaling_fit.optimal_at(curves, C)
            if not best:
                continue
            loss, N = best
            ax.axvline(C, color="0.75", ls=":", lw=1, zorder=1)
            ax.plot([C], [loss], "*", color="crimson", ms=17, zorder=7)
            ax.annotate(f"optimal\nN={N/1e6:.1f}M", (C, loss), textcoords="offset points",
                        xytext=(7, -1), fontsize=8, color="crimson", va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute  C = 6·N·D  (FLOPs)")
    ax.set_ylabel("validation cross-entropy")
    ax.set_title("text_pretrain — per-model training curves, loss vs compute\n"
                 "at each compute several sizes compete; the lower envelope (★) is the "
                 "compute-optimal choice")
    if a:
        ax.text(0.97, 0.95, f"compute-optimal exponent\n$N_{{opt}}\\propto C^{{a}}$,  a ≈ {a:.2f}"
                "\n(Chinchilla ~0.5)", transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    # log-y over the FULL drop (init ~10 -> each floor): the whole story —
    # plunge, bend, saturation — stays visible, like the reference figure.
    yt = [4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(HERE / "scaling_law.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train one curve per given depth (one GPU)")
    r.add_argument("--depths", type=int, nargs="+", required=True)
    r.set_defaults(func=cmd_run)
    f = sub.add_parser("fit", help="merge curves, fit the frontier, write scaling.json + figure")
    f.set_defaults(func=cmd_fit)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
