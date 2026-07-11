"""
scaling_fit.py — the compute-optimal FRONTIER from per-model training curves.

Each model size N is trained once; along the run we log (compute = 6·N·tokens,
val). Bigger models reach lower loss but need more compute, so the curves cross,
and their LOWER ENVELOPE (running-min over compute) is the compute-optimal
frontier. The exponent a (N_opt ∝ C^a; Chinchilla ≈ 0.5) is read from that
frontier: at each compute C the optimal size N_opt(C) is the one with the lowest
interpolated loss; fit log N_opt vs log C.
"""
import numpy as np


def envelope(curves):
    """Compute-optimal frontier = running-min val over all (compute, val) points."""
    pts = sorted((p["compute"], p["val"])
                 for c in curves for p in c["trajectory"] if p["compute"] > 0)
    fc, fl, run = [], [], float("inf")
    for c, l in pts:
        if l < run - 1e-9:
            run = l
            fc.append(c)
            fl.append(l)
    return fc, fl


def _series(curves):
    """Each curve as (N, log10 compute array, loss array), needing >=2 points."""
    out = []
    for c in curves:
        tr = [(p["compute"], p["val"]) for p in c["trajectory"] if p["compute"] > 0]
        if len(tr) >= 2:
            out.append((c["N"], np.log10([t[0] for t in tr]), np.array([t[1] for t in tr])))
    return out


def optimal_at(curves, C):
    """(loss, N) of the best model size at compute C (interpolated); None if uncovered."""
    lc = np.log10(C)
    best = None
    for N, cc, ll in _series(curves):
        if cc.min() <= lc <= cc.max():
            loss = float(np.interp(lc, cc, ll))
            if best is None or loss < best[0]:
                best = (loss, N)
    return best


def frontier_exponent(curves):
    """a = slope of log N_opt vs log C, over the compute range where >=2 sizes
    COMPETE. Sampled across the full envelope range (not just where ALL sizes
    overlap) — with per-size curves offset by N, consecutive sizes overlap
    pairwise even when the smallest and largest never share a compute budget."""
    series = _series(curves)
    if len(series) < 2:
        return None
    lo = min(s[1].min() for s in series)
    hi = max(s[1].max() for s in series)
    Cs, Nopt = [], []
    for lc in np.linspace(lo, hi, 80):
        covering = [(float(np.interp(lc, cc, ll)), N)
                    for N, cc, ll in series if cc.min() <= lc <= cc.max()]
        if len(covering) >= 2:                     # >=2 sizes compete at this budget
            Cs.append(10 ** lc)
            Nopt.append(min(covering)[1])
    if len(set(Nopt)) < 2:
        return None
    return float(np.polyfit(np.log(Cs), np.log(Nopt), 1)[0])
