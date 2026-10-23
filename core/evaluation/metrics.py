"""
core's evaluation metrics, in the form evaluate_loss takes (core/evaluation/eval_loss.py): a function of
(batch, per-position CE) returning (numerator, denominator) for one batch. Each factory makes one metric, and the
caller names it:

    LossEvaluator(loader, steps, metrics={"val/ce": mean_ce(), "val/bpb": bits_per_byte(token_bytes)})
    metrics = {f"val/ce_type{t}": type_ce(t) for t in (0, 1, 2)}

Subsets are taken by boolean selection, never as `ce * mask`: a NaN at an unselected position times 0 is still NaN.
"""

import math

import torch

from core.tokenization.vocab_layout import VocabLayout

IGNORE = VocabLayout.IGNORE_INDEX


def mean_ce():
    """Mean CE over the non-IGNORE targets, weighted by target count: across batches and ranks too, so not a mean
    of batch means."""
    def metric(batch, ce):
        valid = batch["targets"] != IGNORE
        return ce[valid].sum(), valid.sum()
    return metric


def type_ce(type_id):
    """Mean CE over the non-IGNORE targets whose TARGET type is type_id: batch["target_types"], not the input
    token_types. A batch with no such target contributes nothing to it."""
    def metric(batch, ce):
        if "target_types" not in batch:
            raise ValueError("type_ce: the batch has no 'target_types'")
        target_types, targets = batch["target_types"], batch["targets"]
        if target_types.shape != targets.shape:
            # A mismatched shape would broadcast (a one-element target_types puts every target in one type) and
            # read wrong without an error.
            raise ValueError(f"type_ce: target_types has shape {tuple(target_types.shape)}, "
                             f"targets {tuple(targets.shape)}")
        sel = (targets != IGNORE) & (target_types == type_id)
        return ce[sel].sum(), sel.sum()
    return metric


def bits_per_byte(token_bytes):
    """Bits per byte over the byte-bearing targets: their CE in bits over their UTF-8 bytes. Zero-byte targets
    (control tokens) and IGNORE count in neither sum, so this is not the total CE / log 2, which would charge the
    control targets' CE to the text's bytes (public issue #6).

    token_bytes: [vocab_size] byte counts indexed by the batch's global token IDs (the modality keeps it aligned
    with its tokenizer). It stays where it is given, the CPU say, and is moved to the CE's device on first use.
    """
    table = token_bytes

    def metric(batch, ce):
        nonlocal table
        if table.device != ce.device:
            table = table.to(ce.device)
        y = batch["targets"]
        valid = y != IGNORE
        num_bytes = torch.where(valid, table[torch.where(valid, y, 0)], 0)
        sel = num_bytes > 0
        return ce[sel].sum().double() / math.log(2), num_bytes[sel].sum()
    return metric
