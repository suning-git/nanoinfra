"""
evaluator.py — one ruler, three lines.

Mean next-token cross-entropy over the SUPERVISED tokens of a frozen val set. The
same class serves all three lines, because once the data source has written its
supervision_mask there is nothing modality-specific left to measure: the loader's
next-token preparation has already turned unsupervised tokens into IGNORE_INDEX
targets, and cross-entropy already skips them.

FROZEN AT CONSTRUCTION, on purpose. A val set that is re-sampled every time it is
used is a ruler that moves, and two runs measured with it are not comparable. The
cost is holding a few hundred rows of integers; the alternative has silently
invalidated comparisons in this repo before.

THE AGGREGATION IS TOKEN-WEIGHTED, not mean-of-batch-means: total nats over all
supervised positions, divided by the count of them. Two functionals with the same
name are not the same metric — a batch-mean and a pooled mean differ whenever rows
carry different numbers of supervised tokens, which is exactly the t2m case
(motions vary in length, and a long caption leaves less room for its motion).
Written down because the difference is invisible in the number and only shows up
when someone compares against a run that aggregated the other way.

ONE GPU OR MANY, THE SAME BATCHES. Every rank freezes the same batches (train.py
builds the val loader as rank 0 of 1, which every source honours) and evaluates its
shard of them, batches rank, rank + N, ...; core's LossEvaluator adds the nats and
the token counts over the ranks before dividing. So whatever N is, every frozen
batch is evaluated once, on one GPU, and a GPU runs ceil(K / N) of the K forwards
(two GPUs under NanoDDP took about half the time, 2026-10-21). The READING is the
same only on the same compute path (up to the order of the fp64 sums): train.py
compiles the whole trunk on one GPU and block by block under NanoDDP, and FSDP
casts the floating inputs of its sharded modules to bf16, so one GPU, NanoDDP and
FSDP read 2e-5 to 3.5e-4 apart (measured 2026-10-21; one GPU compiled block by
block read exactly what two-GPU NanoDDP read). Before 2026-10-21
every rank evaluated all of its frozen batches and rank 0's reading was reported;
on the text and video lines the sources ignored "rank 0 of 1" and each rank froze
other rows, so a multi-GPU reading covered rank 0's share of them only.

The evaluation runs core's path (core/evaluation/eval_loss.py): the System's
prepare_trunk_inputs, the trunk, then head.loss_per_token, the CE implementation
training uses (head_ce).

Scheduling is core's Evaluator contract: the Trainer asks should_eval(step) every
step. The default is periodic; scaling.py passes an explicit log-spaced eval_at.
"""

import torch

from core.evaluation.evaluator import Evaluator, LossEvaluator
from core.evaluation.metrics import mean_ce
from core.tokenization.vocab_layout import VocabLayout
from core.utils import process_identity


class SupervisedCE(Evaluator):
    """Args:
        loader:     a val-split MixedDataLoader (assembly.build_loader(..., "val")), built as rank 0 of 1 so that
                    every rank reads the same batches
        n_batches:  how many batches to freeze
        metric:     metric name, e.g. "val/video_ce"
    """

    def __init__(self, loader, n_batches=20, interval_steps=500, eval_at=None,
                 metric="val/ce", label=""):
        self.interval_steps = interval_steps
        self.eval_at = {int(s) for s in eval_at} if eval_at else None
        self.metric = metric
        self.label = label
        self.best = float("inf")

        it = iter(loader)
        frozen = []
        n_sup = 0
        for _ in range(n_batches):
            b = next(it)
            # Keep only what the evaluation reads. `state_dict` holds live source
            # objects; carrying it would pin the whole loader alive for the run.
            kept = {k: b[k] for k in ("idx", "token_types", "targets", "layout_facts") if k in b}
            frozen.append(kept)
            n_sup += int((kept["targets"] != VocabLayout.IGNORE_INDEX).sum())
        self.n_batches, self.n_supervised = len(frozen), n_sup

        # This rank's shard: batches rank, rank + N, ... Every rank runs ceil(K / N) of them, because under FSDP
        # every forward is a collective; a rank left one short also runs a copy of the first batch with every target
        # IGNORE, which mean_ce counts as nothing (a metric whose denominator does not come from the targets would
        # count it; this evaluator has none).
        identity = process_identity()
        self.world_size = identity.world_size
        shard = frozen[identity.rank::identity.world_size]
        steps = -(-len(frozen) // identity.world_size)
        if len(shard) < steps:
            shard.append({**frozen[0], "targets": torch.full_like(frozen[0]["targets"], VocabLayout.IGNORE_INDEX)})
        self._eval = LossEvaluator(shard, steps, {metric: mean_ce()})

    def describe(self) -> str:
        """One line for the startup log: WHICH rows, HOW many, HOW often. Both
        rulers must speak in the log — the train/val drift bug survived for months
        because only the training side did."""
        when = (f"at steps {sorted(self.eval_at)[:4]}..." if self.eval_at
                else f"every {self.interval_steps} steps")
        shared = (f", {self._eval.eval_steps} forwards per rank on {self.world_size}"
                  if self.world_size > 1 else "")
        return (f"{self.metric}: {self.n_batches} frozen val batches "
                f"({self.n_supervised:,} supervised tokens){self.label}{shared}, {when}")

    def evaluate(self, system, autocast_ctx) -> dict:
        mean = self._eval.evaluate(system, autocast_ctx)[self.metric]   # LossEvaluator enters the context
        self.best = min(self.best, mean)
        return {self.metric: mean, f"{self.metric}_best": self.best}
