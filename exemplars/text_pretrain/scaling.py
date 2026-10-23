"""
scaling.py — stage 2: the compute-optimal scaling law (per-model training curves).

Trains ONE run per model size (d2..d8) through the SAME blessed Orchestrator as
pretrain (spec.ORCHESTRATOR) to a fixed token budget at CONSTANT LR, evaluating
at ~40 LOG-SPACED steps along the way. The schedule is computed HERE and injected
via `evaluation.eval_at` — core evaluators accept an explicit step set; core
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
  * ONE shared LR across sizes, 3e-4, with the compiled head (LR_MAX, HEAD_CE
    below), pinned HERE rather than read from spec.py, whose champion recipe has
    since moved on. The shipped curves were measured with the liger head; the
    study moved to compiled on 2026-10-22, the fastest arm. Fine for the
    frontier's shape at this scale; a precision study would tune LR per size.

WHY curves + frontier (not a 2D grid + parametric fit): the frontier IS the
compute-optimal object — at each compute budget several sizes compete and the
envelope picks the single best.

Run — split the depths across both GPUs, then collect + fit + plot:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m exemplars.text_pretrain.scaling run --depths 8 6
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m exemplars.text_pretrain.scaling run --depths 4 3 2
  .venv/bin/python -m exemplars.text_pretrain.scaling fit   # -> outputs/scaling.json + .png
`run --param-dtype` / `fit --param-dtype` pick the storage of parameters, gradients and AdamW moments; both default to
fp32, the storage the shipped curves were measured in (and core's default), so that a reproduction is compared like
with like (bf16 storage gives different curves). Each storage has its own shards, and a fit takes one storage's curves.
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from exemplars.text_pretrain import scaling_fit, spec

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent           # repo root — the orchestrator subprocess runs here

# Two directories, and NOTHING here writes to the first one. `example_results/`
# holds the run this exemplar ships as its claim (committed, linked from
# RESULTS.md); OUT holds yours. They used to be one directory, so running the
# exemplar overwrote the numbers it was published to demonstrate — and the
# comparison you actually want ("mine vs theirs") became impossible at the exact
# moment you produced something to compare. `outputs/` is the name on purpose:
# .gitignore and make_release.py already exclude an `outputs/` at any depth.
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
EXAMPLE = HERE / "example_results"   # read-only here: `fit` compares against it, never writes it

DEPTHS = [2, 3, 4, 6, 8]            # model sizes N (non-embedding params)
SEQ_LEN, DBS, TBS = 1024, 32, 32768
LR_MAX, HEAD_CE = "3e-4", "compiled"   # NOT the champion's recipe (spec.py). compiled since
                                       #   2026-10-22; the shipped curves were measured with liger
                                       #   (RESULTS §2), and `fit` prints which arm each side used.
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

# The orchestrator's eval line is assembled by joining whatever metrics exist
# (core/training/trainer.py), so neither the field ORDER nor the PRESENCE of any
# given field is guaranteed: `val/bpb` only appears when a token_bytes.pt is on
# disk. Anchor on "Step N |" and then look fields up BY NAME — a positional
# regex would simply stop matching on a setup without bpb, and the run would be
# reported as "0 eval points, curve FAILED" while the training was in fact fine.
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


def parse_curve_log(path, N):
    """A finished run's log -> its (compute, val) trajectory. One pass, no side effects.

    Reading the numbers back out of a human-readable log is a normal thing for a
    small local study to do, and it keeps the training log free to be whatever it
    is. What this deliberately is NOT is a filter on a live stream: the parse runs
    over a file that has stopped changing, so it has one job.

    bpb rides along when the log has it. CE is per-token and comparable only
    within one tokenizer artifact; bpb is per-UTF-8-byte and survives a retrained
    BPE, which is what a reproduction on someone else's tokenizer needs. Nothing
    fits on bpb — it is recorded so that comparison becomes possible at all.
    """
    traj = []
    with open(path, errors="replace") as f:
        for line in f:
            parsed = parse_eval_line(line)
            if parsed is None:
                continue
            step, fields = parsed
            if "val/text_ce" not in fields:
                continue
            point = {"step": step, "tokens": step * TBS,
                     "compute": 6.0 * N * step * TBS, "val": fields["val/text_ce"]}
            if "val/bpb" in fields:
                point["bpb"] = fields["val/bpb"]
            traj.append(point)
    return traj


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


def run_curve(depth, param_dtype="fp32"):
    """One constant-LR run to MAX_TOKENS through the Orchestrator; return its
    (compute, val) trajectory parsed from the log-scheduled evaluations."""
    N = n_nonembed(depth)
    max_steps = int(MAX_TOKENS // TBS)
    steps = eval_schedule(max_steps)
    ov = spec.train_overrides(depth=depth, lr=LR_MAX, **{
        "head_ce": HEAD_CE,
        "sequence_len": SEQ_LEN, "device_batch_size": DBS, "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,   # budget-invariant by construction
        "optimizer.scheduler.warmdown_ratio": 0.0,      # constant LR after warmup —
        "optimizer.scheduler.final_lr_frac": 1.0,       #   no end-of-run warmdown dip
        "checkpoint.enabled": "false",
        "evaluation.eval_at": "[" + ",".join(map(str, steps)) + "]",
        "evaluation.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 200,
        "param_dtype": param_dtype,
    })
    log = OUT / f"curve_d{depth}.log"
    print(f"[run ] d{depth} N={N/1e6:.1f}M -> {MAX_TOKENS/1e6:.0f}M tokens "
          f"({max_steps} steps, {len(steps)} log-spaced evals) -> {log.name} "
          f"(tail -f to watch)", flush=True)
    # The child's output goes STRAIGHT to a file; this process never reads the
    # stream. Reading it would mean deciding, live, which lines to echo and which
    # to parse — two jobs braided into one loop, over a stream that can contain
    # anything. Parsing a log afterwards is fine; parsing it while relaying it is
    # what was not. `-u` above keeps the child unbuffered, so the file is live
    # under `tail -f`, and subprocess.run cleans the child up if this process dies.
    #
    # A subprocess rather than an in-process call, deliberately: the orchestrator
    # is a @hydra.main CLI entry, and reaching it any other way (the compose API)
    # would characterise a different code path than the one users run.
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                            cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                            stdout=f, stderr=subprocess.STDOUT).returncode
    traj = parse_curve_log(log, N)
    if rc != 0 or len(traj) < 3:
        raise SystemExit(f"curve d{depth} FAILED (rc={rc}, {len(traj)} evals) — see {log}")
    print(f"[done] d{depth}: {len(traj)} eval points, val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}",
          flush=True)
    return {"depth": depth, "N": N, "param_dtype": param_dtype, "head_ce": HEAD_CE, "trajectory": traj}


def _ours(curve, param_dtype):
    """A curve this study may resume from or fit: the same storage and the same head CE. A curve that records no
    head_ce (written before 2026-10-22, when the study pinned liger) is not taken for any arm."""
    return curve.get("param_dtype", "bf16") == param_dtype and curve.get("head_ce") == HEAD_CE


def cmd_run(args):
    """Train one curve per depth (one GPU). Per-shard JSON, resumable."""
    shard = OUT / f"curves_{'-'.join(map(str, args.depths))}_{args.param_dtype}_{HEAD_CE}.json"
    curves = json.loads(shard.read_text())["curves"] if shard.exists() else []
    curves = [c for c in curves if _ours(c, args.param_dtype)]
    done = {c["depth"] for c in curves}
    for d in args.depths:
        if d in done:
            print(f"[skip] d{d}", flush=True)
            continue
        curves.append(run_curve(d, args.param_dtype))
        shard.write_text(json.dumps({"curves": curves}, indent=2))
    print(f"WROTE {shard} ({len(curves)} curves)")


def cmd_fit(args):
    """Merge shards, fit the frontier exponent, write scaling.json + the figure."""
    by_depth, others = {}, 0
    for f in sorted(glob.glob(str(OUT / "curves_*.json"))):
        for c in json.loads(Path(f).read_text())["curves"]:
            if _ours(c, args.param_dtype):   # storage from before its key: bf16, all there was
                by_depth[c["depth"]] = c
            else:
                others += 1
    if not by_depth and (OUT / "scaling.json").exists():   # re-plot from YOUR last fit
        for c in json.loads((OUT / "scaling.json").read_text()).get("curves", []):
            if _ours(c, args.param_dtype):
                by_depth[c["depth"]] = c
    if others:
        print(f"(left out {others} curves of another storage or head_ce, or recording no head_ce)")
    curves = sorted(by_depth.values(), key=lambda c: c["N"])
    if len(curves) < 3:
        raise SystemExit(f"only {len(curves)} {args.param_dtype} / {HEAD_CE} curves found — run the study first")
    a = scaling_fit.frontier_exponent(curves)
    print(f"compute-optimal frontier exponent a = {a:.3f}   (Chinchilla ~0.5)"
          if a else "frontier exponent: undetermined (need >=2 sizes competing)")
    out = {"study": {"depths": DEPTHS, "seq_len": SEQ_LEN, "total_batch_size": TBS,
                     "max_tokens": MAX_TOKENS, "lr_max": LR_MAX, "head_ce": HEAD_CE, "lr_schedule": "constant",
                     "eval_schedule": f"{N_EVALS} log-spaced from step 20",
                     "eval_tokens": EVAL_TOKENS, "param_dtype": args.param_dtype},
           "a_frontier": a, "curves": curves}
    (OUT / "scaling.json").write_text(json.dumps(out, indent=2))
    _plot(curves, a)
    print(f"wrote {OUT / 'scaling.json'} + {OUT / 'scaling_law.png'}")
    _compare_to_example(curves, a)


def _compare_to_example(curves, a):
    """Print your fit beside the one this exemplar ships, if it is still there.

    The point of a reproduction is the comparison, and it is the first thing you
    want and the last thing the old layout let you have: `fit` used to overwrite
    the shipped numbers, so producing your own destroyed what you meant to check
    against. Both live now, so print the difference instead of making everyone
    diff two JSON files by hand.

    Absolute CE is NOT comparable across tokenizer artifacts (a different BPE
    means a different number of bytes behind each token). The exponent is: it is
    a log-log slope, and it is what should reproduce.
    """
    ref = EXAMPLE / "scaling.json"
    if not ref.exists():
        return
    try:
        r = json.loads(ref.read_text())
    except (json.JSONDecodeError, OSError):
        return
    ref_curves = r.get("curves", [])
    ra, rc = r.get("a_frontier"), {c["depth"]: c for c in ref_curves}
    print(f"\n  vs {ref.relative_to(HERE)} (the run this exemplar ships):")
    theirs_dtype = r.get("study", {}).get("param_dtype", "bf16")   # shipped before the key: bf16, all there was
    mine_dtype = curves[0].get("param_dtype", "bf16") if curves else theirs_dtype
    print(f"    storage         example {theirs_dtype}   yours {mine_dtype}"
          + ("" if mine_dtype == theirs_dtype else "   (different: the curves are not comparable)"))
    theirs_ce = r.get("study", {}).get("head_ce", "unrecorded")    # shipped before the key: not recorded
    print(f"    head_ce         example {theirs_ce}   yours {HEAD_CE}"
          + ("" if theirs_ce == HEAD_CE else "   (different arms: the numbers differ a little)"))
    if a and ra:
        print(f"    a_frontier      example {ra:.4f}   yours {a:.4f}   ({a - ra:+.4f})")
    for c in curves:
        o = rc.get(c["depth"])
        if not o or not c.get("trajectory") or not o.get("trajectory"):
            continue
        mine, theirs = c["trajectory"][-1], o["trajectory"][-1]
        print(f"    val CE @ d{c['depth']:<3}    example {theirs['val']:.4f}   "
              f"yours {mine['val']:.4f}   ({mine['val'] - theirs['val']:+.4f})")
        if "bpb" in mine and "bpb" in theirs:   # bits per byte survives a tokenizer change
            print(f"    val bpb @ d{c['depth']:<3}   example {theirs['bpb']:.4f}   "
                  f"yours {mine['bpb']:.4f}   ({mine['bpb'] - theirs['bpb']:+.4f})")
    print("    (CE is per-token: comparable only within ONE tokenizer artifact. "
          "The exponent is the part that should reproduce.)")
    # A reference from before bpb collection has none: say so, rather than letting
    # a reader lean on the CE column just called untrustworthy.
    if not any("bpb" in p for c in ref_curves for p in c.get("trajectory", [])):
        print("    (The shipped reference predates val/bpb, so no byte-normalised "
              "comparison is possible against it — yours records bpb for the next one.)")


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
    fig.savefig(OUT / "scaling_law.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train one curve per given depth (one GPU)")
    r.add_argument("--depths", type=int, nargs="+", required=True)
    r.add_argument("--param-dtype", choices=["bf16", "fp32"], default="fp32",
                   help="storage of parameters, gradients and AdamW moments (default fp32: what the shipped curves used)")
    r.set_defaults(func=cmd_run)
    f = sub.add_parser("fit", help="merge curves, fit the frontier, write scaling.json + figure")
    f.add_argument("--param-dtype", choices=["bf16", "fp32"], default="fp32",
                   help="which storage's curves to fit (default fp32: what the shipped curves used)")
    f.set_defaults(func=cmd_fit)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
