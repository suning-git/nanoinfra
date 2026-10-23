"""
evaluator.py — the ruler: the val NELBO, on core's loss evaluation.

The val set is FROZEN in three ways at once, which is what makes the number
comparable across runs, checkpoints and numbers of GPUs: fixed clips (n of the cache's
val split, spread evenly over it), a fixed grid of noise levels, and a fixed mask per
clip and level (drawn on the CPU from a generator seeded by the clip's row and the
level). Two checkpoints evaluated here differ only by their weights.

At level t every cell of the predicted blocks is masked with probability t, and the
reading is the CE summed over the masked cells divided by their number; val/nelbo is
the mean of the levels' readings. Nats per predicted token, the training loss's units
(block_diffusion.py).

ONE PATH WITH TRAINING. A val clip is prepared the way a training clip is
(block_diffusion.two_stream_sample: the same two copies, targets and fields; only its
noise is fixed rather than drawn), batched the same way, and evaluated by core's
LossEvaluator, which runs the System's own prepare_trunk_inputs, trunk and
head.loss_per_token (core/evaluation/eval_loss.py). Nothing here knows how the trunk is
called.

One thing is added on the way: the head runs at the masked cells only (_AtTargets), as
in training. core's evaluate_loss runs it at every position of the row, and here only
the masked cells of the noisy copy have targets — about a fifth of the row at 17 frames,
while the head costs about as much per position as the whole trunk. At 17 frames an
evaluation of 420 clips x 4 levels takes 8.3 s and 1.8 GiB at its peak this way, against
11.1 s and 3.2 GiB with the head at every position (the reading is the same to 1e-8); at
129 frames a batch of 4 clips peaks at 6.9 GiB instead of 14.

With N GPUs every rank holds the same plan of batches and evaluates batches r, r + N,
...; core adds each level's CE sum and masked count over the ranks before dividing, so
the reading is the one-GPU reading (up to the order of the float64 sums) and a GPU runs
about 1/N of the forwards. A rank left one batch short runs a copy of the first batch
with every target IGNORE, which adds nothing: every rank must run the same number of
forwards (under FSDP each is a collective).

The batches are built again from the clips at every evaluation, on the CPU, the same
every time — the masks are a function of (row, level) — rather than held: at 129
frames, 500 clips x 4 levels of prepared two-stream samples would take GBs.

Checkpoint policy note: core saves periodically (save_every / keep_last_n), not on
"best". This evaluator therefore reports `val/nelbo_best` as a LOG metric so the run
still shows whether it is improving; it does not reach for the checkpoint manager.
"""

import hashlib

import torch

from core.data.layout import collate_samples, padding_layout
from core.data.supervision import IGNORE
from core.evaluation.evaluator import Evaluator, LossEvaluator
from core.utils import process_identity

from exemplars.nano_world_model.block_diffusion import two_stream_sample


def level_sample(clean, t, row, rows, vocab, mask_id):
    """A val clip prepared at noise level t: each cell of the predicted blocks masked with probability t, from a
    generator seeded by (row, t) — the same mask at every evaluation, on every rank."""
    key = f"val/{row}/{t}".encode()
    g = torch.Generator().manual_seed(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little"))
    masked = (torch.rand(rows.row_len, generator=g) < t) & (rows.blk > 0)
    return two_stream_sample(clean, masked, masked.float(), rows, vocab, mask_id, {"row": int(row), "t": t})


class _LevelBatches:
    """This rank's val batches, built on iteration: one batch per (level, run of `batch` clips) of its share of the
    plan, collated as core's padding batcher collates (one clip per row), with the batch's level as batch["t"], then
    moved to `device`; then `fillers` copies of the batch `first` ((level, first clip) of the whole plan's first) with
    every target IGNORE."""

    def __init__(self, clips, sel, plan, fillers, first, batch, rows, vocab, mask_id, pads, device):
        self.clips, self.sel, self.plan, self.fillers, self.first, self.batch = clips, sel, plan, fillers, first, batch
        self.rows, self.vocab, self.mask_id, self.pads, self.device = rows, vocab, mask_id, pads, device

    def _build(self, t, i):
        samples = [level_sample(self.clips[j], t, self.sel[j], self.rows, self.vocab, self.mask_id)
                   for j in range(i, min(i + self.batch, len(self.clips)))]
        shape = (len(samples), 2 * self.rows.row_len)
        batch = collate_samples(samples, padding_layout([2 * self.rows.row_len] * len(samples), shape),
                                shape=shape, pad_values=self.pads)
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["t"] = t
        return batch

    def __iter__(self):
        for t, i in self.plan:
            yield self._build(t, i)
        for _ in range(self.fillers):
            first = self._build(*self.first)
            yield {**first, "targets": torch.full_like(first["targets"], IGNORE)}


class _HeadAtTargets:
    """head.loss_per_token at the non-IGNORE targets only, put back into a zero CE shaped like the targets (what
    evaluate_loss expects from the head). Each CE value is the one the head computes for that position; the work
    scales with the masked cells, not the row."""

    def __init__(self, head):
        self.head = head

    def loss_per_token(self, hidden, targets):
        sel = targets != IGNORE
        ce = self.head.loss_per_token(hidden[sel], targets[sel])
        out = ce.new_zeros(targets.shape)
        out[sel] = ce
        return out


class _AtTargets:
    """The System as evaluate_loss uses it: its own prepare_trunk_inputs and trunk, its head at the targets only
    (_HeadAtTargets). parameters() gives evaluate_loss the device."""

    def __init__(self, system):
        self.system, self.trunk, self.head = system, system.trunk, _HeadAtTargets(system.head)

    def parameters(self):
        return self.system.parameters()

    def prepare_trunk_inputs(self, batch):
        return self.system.prepare_trunk_inputs(batch)


def _masked_ce(t):
    """val/nelbo_t{t}: the CE summed over the masked cells of t's batches, and their number."""
    def metric(batch, ce):
        if batch["t"] != t:
            return 0, 0
        sel = batch["targets"] != IGNORE
        return ce[sel].sum(), sel.sum()
    return metric


class NELBOEvaluator(Evaluator):
    """The val NELBO on frozen clips (module docstring).

    Args:
        dataset:  VideoRowDataset over the val split
        rows:     RowLayout; vocab: VocabLayout; mask_id: global id of [MASK]
        pads:     block_diffusion.pad_values
        n_rows:   val clips, spread over the split (capped at what it holds)
        t_grid:   the fixed noise levels to average over
        batch:    clips per forward
        interval_steps / eval_at: cadence (core's Evaluator scheduling)
    """

    metric = "val/nelbo"

    def __init__(self, dataset, rows, vocab, mask_id, pads, n_rows, t_grid, batch=4,
                 interval_steps=1000, eval_at=None, device="cuda"):
        self.t_grid = tuple(t_grid)
        self.interval_steps = interval_steps
        self.eval_at = {int(s) for s in eval_at} if eval_at else None
        self.best = float("inf")

        sel = dataset.spread(n_rows)
        clips = rows.assemble(dataset.codes[sel], dataset.actions[sel])
        self.n_rows = len(clips)
        plan = [(t, i) for t in self.t_grid for i in range(0, self.n_rows, batch)]
        who = process_identity()
        self.world_size, self.n_batches = who.world_size, len(plan)
        mine = plan[who.rank::who.world_size]
        steps = -(-len(plan) // who.world_size)
        batches = _LevelBatches(clips, sel, mine, steps - len(mine), plan[0], batch, rows, vocab, mask_id, pads,
                                device)
        self._eval = LossEvaluator(batches, steps, {f"val/nelbo_t{t}": _masked_ce(t) for t in self.t_grid})

    def describe(self):
        shared = (f", {self.n_batches} batches shared by {self.world_size} processes ({self._eval.eval_steps} each)"
                  if self.world_size > 1 else "")
        return (f"NELBO on {self.n_rows} frozen val clips (spread over the split), t grid {self.t_grid}{shared}, "
                f"every {self.interval_steps} steps")

    def evaluate(self, system, autocast_ctx):
        res = self._eval.evaluate(_AtTargets(system), autocast_ctx)     # LossEvaluator enters the context
        per_t = {t: res[f"val/nelbo_t{t}"] for t in self.t_grid}
        mean = sum(per_t.values()) / len(per_t)
        self.best = min(self.best, mean)
        out = {"val/nelbo": mean, "val/nelbo_best": self.best}
        out.update({f"val/nelbo_t{t}": v for t, v in per_t.items()})
        return out
