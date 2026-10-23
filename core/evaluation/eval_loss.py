"""
Evaluation service layer: the loss evaluation of a System over a stream of batches.

One path. Each batch goes through the System's own input preparation and trunk (the module training runs,
compiled in place when the orchestrator compiled it), then `head.loss_per_token`, which follows the head's CE
implementation (head_ce). What is reported is a set of metrics over that per-position CE, each a ratio of two
sums: a function `metric(batch, ce) -> (numerator, denominator)` called once per batch, read as the sum of its
numerators over every batch and every rank divided by the sum of its denominators. core's metrics (total CE,
per-type CE, bits per byte) are in core/evaluation/metrics.py; a project writes its own the same way, e.g. a
weighted objective or a per-group breakdown.
"""

import torch
import torch.distributed as dist


@torch.no_grad()
def evaluate_loss(model, loader, steps, metrics):
    """
    Evaluate per-position CE metrics of a System over `steps` batches.

    Args:
        model: System with prepare_trunk_inputs(batch) -> (idx, kwargs); model.trunk(idx, **kwargs) returns hidden;
            model.head.loss_per_token(hidden, targets) returns the unreduced CE, 0 at IGNORE targets
        loader: iterable yielding batch dicts. Each needs `targets` aligned position by position with the trunk's
            output (whether they are shifted is the data's business), plus whatever the metrics read.
        steps: number of batches, > 0. The SAME on every rank, and the loader must yield at least that many: every
            rank runs the trunk and the head on every batch (under FSDP each forward is a collective), whether or
            not its batch holds anything for a given metric.
        metrics: {name: fn}, fn(batch, ce) -> (numerator, denominator), with ce the per-position CE shaped like
            batch["targets"]. Each of the two is a Python number or a 0-dim tensor on ce's device; a batch with
            nothing for the metric returns (0, 0). fn reads batch and ce only: no reduction across ranks, no state
            across batches. Every rank builds the evaluator from the same code and config, so every rank holds the
            same names with the same meaning; rows are laid out by sorted name.

    Returns:
        {name: sum of numerators / sum of denominators} over every batch and rank, in the order of `metrics`;
        inf where the global denominator is 0 (no data for that metric anywhere).

    What this measures is the CE the batch's `targets` define, not `model.loss`: an auxiliary term, targets the
    System builds itself, or another objective are not in it. Any average that splits into two sums fits (a mean of
    batch means is (batch mean, 1) per batch); a median, or a non-linear function of a global value (perplexity is
    exp of the global CE), does not, and the latter the caller computes from the result.

    Precision: the per-position CE and each metric's sum over a batch are whatever the head and the metric compute;
    only the running totals across batches are float64, one number per numerator and denominator.
    """
    if steps <= 0:
        raise ValueError(f"evaluate_loss: steps must be positive, got {steps}")
    names = sorted(metrics)                   # one row per metric, by name: the same layout on every rank
    device = next(model.parameters()).device
    sums = torch.zeros(len(names), 2, dtype=torch.float64, device=device)

    batch_iter = iter(loader)
    for i in range(steps):
        try:
            batch = next(batch_iter)
        except StopIteration:
            raise RuntimeError(f"evaluate_loss: the loader ran out after {i} of {steps} batches") from None
        y = batch["targets"]
        # The System's prepare_trunk_inputs turns the batch into trunk arguments, the same translation training
        # uses, and system.trunk is the same module training runs.
        idx, trunk_kwargs = model.prepare_trunk_inputs(batch)
        hidden = model.trunk(idx, **trunk_kwargs)
        ce = model.head.loss_per_token(hidden, y).reshape(y.shape)
        for row, name in enumerate(names):
            num, den = metrics[name](batch, ce)
            sums[row, 0].add_(num)            # a Python number goes in as a kernel scalar: no host-to-device copy
            sums[row, 1].add_(den)

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    totals = dict(zip(names, sums.tolist()))
    return {name: totals[name][0] / totals[name][1] if totals[name][1] != 0 else float("inf")
            for name in metrics}
