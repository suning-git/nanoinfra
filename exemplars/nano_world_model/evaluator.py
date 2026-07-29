"""
evaluator.py — the rulers: one per objective, both on core's Evaluator contract.

The val set is FROZEN in three ways at once, which is what makes the number
comparable across runs, checkpoints, and hardware: fixed rows (the first N of the
cache's val split), a fixed grid of noise levels, and fixed mask RNG per level. Two
checkpoints evaluated here differ only by their weights.

The metric reads in nats per predicted token and is one-directionally comparable to
an autoregressive nll — see block_diffusion.py for why.

Checkpoint policy note: core saves periodically (save_every / keep_last_n), not on
"best". This evaluator therefore reports `val/nelbo_best` as a LOG metric so the run
still shows whether it is improving; it does not reach for the checkpoint manager.
"""

import torch

from core.evaluation.evaluator import Evaluator


class NELBOEvaluator(Evaluator):
    metric = "val/nelbo"

    """Args:
        objective: BlockDiffusion — supplies val_elbo
        dataset:   VideoRowDataset over the val split
        rows:      RowLayout — assembles the frozen rows once, up front
        n_rows:    how many val rows to use
        t_grid:    the fixed noise levels to average over
        batch:     rows per forward during evaluation
        interval_steps / eval_at: cadence (core's Evaluator scheduling)
    """

    def __init__(self, objective, dataset, rows, n_rows, t_grid, batch=4,
                 interval_steps=1000, eval_at=None, device="cuda"):
        self.objective = objective
        self.t_grid = tuple(t_grid)
        self.batch = batch
        self.interval_steps = interval_steps
        self.eval_at = {int(s) for s in eval_at} if eval_at else None
        self.best = float("inf")

        codes, actions = dataset.take(n_rows)
        self.idx = rows.assemble(codes, actions).to(device)
        self.n_rows = len(self.idx)

    def describe(self):
        return (f"NELBO on {self.n_rows} frozen val rows, t grid {self.t_grid}, "
                f"every {self.interval_steps} steps")

    def evaluate(self, system, autocast_ctx):
        with autocast_ctx:
            mean, per_t = self.objective.val_elbo(
                system, self.idx, self.t_grid, batch=self.batch)
        self.best = min(self.best, mean)
        out = {"val/nelbo": mean, "val/nelbo_best": self.best}
        out.update({f"val/nelbo_t{t}": v for t, v in per_t.items()})
        return out


class NLLEvaluator(Evaluator):
    """Next-token nll per PREDICTED token, on the same frozen rows the NELBO uses.

    Deliberately the same rows and the same units as NELBOEvaluator, so the two
    objectives can be put side by side. The comparison is one-directional — NELBO is
    an upper bound on nll for the same model class, so a diffusion number BELOW an AR
    number is a real win and one above it is inconclusive.

    Args:
        objective: Autoregressive — supplies val_nll
        dataset:   VideoRowDataset over the val split
        rows:      RowLayout — assembles the frozen rows once, up front
    """

    metric = "val/nll"

    def __init__(self, objective, dataset, rows, n_rows, batch=8,
                 interval_steps=1000, eval_at=None, device="cuda"):
        self.objective = objective
        self.batch = batch
        self.interval_steps = interval_steps
        self.eval_at = {int(s) for s in eval_at} if eval_at else None
        self.best = float("inf")

        codes, actions = dataset.take(n_rows)
        self.idx = rows.assemble(codes, actions).to(device)
        self.n_rows = len(self.idx)

    def describe(self):
        return (f"next-token nll on {self.n_rows} frozen val rows "
                f"({self.objective.n_supervised} predicted tokens each), "
                f"every {self.interval_steps} steps")

    def evaluate(self, system, autocast_ctx):
        with autocast_ctx:
            mean = self.objective.val_nll(system, self.idx, batch=self.batch)
        self.best = min(self.best, mean)
        return {"val/nll": mean, "val/nll_best": self.best}
