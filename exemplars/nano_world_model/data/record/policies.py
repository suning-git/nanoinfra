"""
data/policies.py — recording policies: v2's blind libraries (ported verbatim
where possible) + the v3 gaze-aware event engine.

Two policy shapes coexist:
  - GENERATORS (coverage/jitter, from nano_t2v record_v2/record_vizdoom):
    blind endless action-id streams; still the FILLER between events.
  - EventPolicy: a stateful wrapper that watches the live labels buffer and
    interrupts the filler with entity-anchored revisit maneuvers (approach ->
    gaze -> pan/leg away -> silent dwell -> return -> re-gaze). This is the
    upgrade the corpus exists for: v2's cons5 panned blindly, so "entity seen ->
    look away -> look back -> entity still there" was rare; here it is scripted
    AGAINST A TRACKED ENTITY and bookkept per event (docs/DESIGN_DATA_v3.md §4).

Turn calibration is inherited from the frozen line: a held turn key sweeps
1.76 deg/tic for 5 tics then 3.52 deg/tic (SLOWTURNTICS, measured 2026-07-21).

v4 (action table v2): stillness is a REAL action now — NOOP id 18. The v3
wiggle (single-tic TL/TR alternation) is retired: it put ±1.76°/tic flicker
into every dwell and owned a third of the corpus's turning mass (corpus_stats,
2026-08-02). Dwells are 80% noop / 20% strafe-hold (3-4 tic held side-steps,
alternating), and a dwell tic that would stare into a nearby wall rotates away
instead (ep104 lesson: face-planted noop dwell is a full-screen texture stall).
"""

import numpy as np

# pure-button ids (spec.ACTION_COMBOS order)
TL, TR, MR, ML, FWD, ATK, NOOP = 0, 1, 2, 5, 8, 17, 18
FWD_TL, FWD_TR = 9, 10
MAIN = [0, 1, 8, 8, 8, 9, 10, 11, 14, 17]     # record_vizdoom's jitter pool

DWELL = -1        # queue sentinel: resolved per-tic (noop / strafe-hold / wall-avoid)
FWD_FAMILY = {8, 9, 10, 11, 12, 13, 14, 15, 16}   # any combo holding MOVE_FORWARD

SEGMENT_ACTIONS = {          # v2 coverage segment -> held action id (verbatim)
    "pure_TL": 0, "pure_TR": 1, "pure_ML": 5, "pure_MR": 2,
    "FWD": 8, "FWD_TL": 9, "FWD_TR": 10, "FWD_ML": 14, "FWD_MR": 11,
    "circle_L": 7, "circle_R": 3,
    "ATK": 17, "sticky_random": None,
    "noop": 18,              # v4: true stillness as a coverage segment
}


def turn_tics(deg):
    """Tics to sweep ~deg with a held turn key (piecewise TRUE rate)."""
    if deg <= 5 * 1.76:
        return max(1, round(deg / 1.76))
    return 5 + round((deg - 5 * 1.76) / 3.52)


class BalancedChooser:
    """v2 verbatim: sample categories toward target shares."""

    def __init__(self, targets, rng):
        self.names = list(targets)
        self.t = np.array([targets[k] for k in self.names], float)
        self.t /= self.t.sum()
        self.done = np.zeros(len(self.names), float)
        self.rng = rng

    def pick(self):
        share = self.done / max(self.done.sum(), 1e-9)
        deficit = self.t - share
        if self.done.sum() > 0 and deficit.max() > 0.02:
            return self.names[int(np.argmax(deficit))]
        return self.names[self.rng.choice(len(self.names), p=self.t)]

    def credit(self, name, amount):
        self.done[self.names.index(name)] += amount


def coverage_policy(rng, recipe, chooser, no_atk=False):
    """v2's endless L/R-symmetric segment stream. no_atk swaps the ATK segment
    for a silent wiggle (sleeping-monster worlds: gunfire wakes)."""
    lo, hi = recipe["coverage"]["seg_tics"]
    while True:
        seg = chooser.pick()
        T = int(rng.integers(lo, hi + 1))
        chooser.credit(seg, T)
        a = SEGMENT_ACTIONS[seg]
        if seg == "ATK" and no_atk:
            for i in range(T):                 # v4: true stillness (was wiggle)
                yield NOOP
        elif a is None:                        # sticky_random
            cur = int(rng.choice(MAIN))
            for _ in range(T):
                if rng.random() > 0.7:
                    cur = int(rng.choice(MAIN))
                if no_atk and cur == ATK:
                    cur = FWD
                yield cur
        else:
            for _ in range(T):
                yield a


class PansPolicy:
    """The pans layer's actor: the v2 coverage segment library uninterrupted —
    committed turns and L/R-symmetric quotas ARE the layer (v3 demoted coverage
    to inter-event filler and the strafe/pure-turn shares collapsed). Interface-
    compatible with the other actors so the runner stays uniform."""

    def __init__(self, filler, rng=None):
        self.filler, self.events, self.mode = filler, [], "filler"
        self.rng = rng or np.random.default_rng(0)
        self.wall_q, self.wall_cool = [], 0

    def act(self, labels, t, hunt_err=None, wall_close=False, wall_near=False, kills=0, ammo=None, mover_ids=None):
        if self.wall_cool > 0:
            self.wall_cool -= 1
        if self.wall_q:
            return self.wall_q.pop(0)
        a = next(self.filler)
        if self.wall_cool == 0 and ((a in (NOOP, ATK) and wall_close)
                                    or (a in FWD_FAMILY and wall_near)):
            side = TR if self.rng.random() < 0.5 else TL
            self.wall_q = [side] * 8                     # committed, then cool
            self.wall_cool = 25
            return self.wall_q.pop(0)
        return a

    def abort(self):
        pass
