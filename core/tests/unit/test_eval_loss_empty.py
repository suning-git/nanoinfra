"""evaluate_loss with a batch that has no supervised target, or none of one type.

Every metric is a ratio of sums, so such a batch adds (0, 0) and the result equals the evaluation without it; no
0/0 is ever taken inside a batch. (The removed fused path accumulated "this batch's mean loss x its target count",
and with no target the mean was 0/0 = NaN, NaN x 0 still NaN: one such batch turned the whole evaluation, or that
type's, into NaN.) A real NaN on a non-empty set still comes through.
"""
import math

import torch

from core.evaluation.eval_loss import evaluate_loss
from core.evaluation.metrics import mean_ce, type_ce
from core.model.heads import LMHead
from core.model.system import prepare_unpacked_AR_trunk_inputs
from core.tokenization.vocab_layout import VocabLayout

IGNORE = VocabLayout.IGNORE_INDEX
V, H = 16, 8
METRICS = {"total": mean_ce(), "type0": type_ce(0), "type1": type_ce(1)}


class _TinySystem(torch.nn.Module):
    prepare_trunk_inputs = staticmethod(prepare_unpacked_AR_trunk_inputs)

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.head = LMHead(H, V)
        with torch.no_grad():
            self.head.lm_head.weight.normal_(0, 1.0)     # nonuniform per-token losses

    def trunk(self, x, token_types=None):
        return self.emb(x)


def _batch(seed, types=(0, 1)):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, V, (2, 6), generator=g)
    targets = torch.randint(0, V, (2, 6), generator=g)
    target_types = torch.tensor(types * 6)[:12].view(2, 6)
    targets[1, 5] = IGNORE
    return {"idx": idx, "targets": targets, "target_types": target_types}


def _run(batches, system=None):
    return evaluate_loss(system or _TinySystem(), batches, len(batches), METRICS)


def test_a_batch_without_targets_adds_nothing():
    full = _batch(1)
    empty = {**_batch(2), "targets": torch.full((2, 6), IGNORE)}
    alone, with_empty = _run([full]), _run([full, empty])
    assert all(math.isfinite(with_empty[k]) and with_empty[k] == alone[k] for k in METRICS)


def test_a_batch_without_one_type_adds_nothing_to_that_type():
    both, only0 = _batch(1, types=(0, 1)), _batch(2, types=(0, 0))
    alone, mixed = _run([both]), _run([both, only0])
    assert math.isfinite(mixed["type1"]) and mixed["type1"] == alone["type1"]


def test_a_real_nan_on_a_batch_with_targets_still_comes_through():
    class _NanHead(LMHead):
        def loss_per_token(self, hidden, targets):
            return super().loss_per_token(hidden, targets) * float("nan")
    system = _TinySystem()
    system.head.__class__ = _NanHead
    result = _run([_batch(1)], system)
    assert all(math.isnan(v) for v in result.values())


def test_a_nan_at_one_type_stays_in_that_type():
    """Subsets are selected, not multiplied by a mask: a NaN at a type-1 position must not reach type 0 (NaN x 0 is
    still NaN), while the total, which holds that position, is NaN."""
    class _NanAtHighIds(LMHead):
        def loss_per_token(self, hidden, targets):
            ce = super().loss_per_token(hidden, targets)
            return torch.where(targets.reshape(-1) >= V // 2, float("nan"), ce)
    system = _TinySystem()
    system.head.__class__ = _NanAtHighIds
    batch = _batch(1)
    batch["target_types"] = (batch["targets"] >= V // 2).long()          # type 1 exactly where the NaNs are
    assert (batch["target_types"] == 1).any() and ((batch["target_types"] == 0) & (batch["targets"] != IGNORE)).any()
    result = _run([batch], system)
    assert math.isfinite(result["type0"])
    assert math.isnan(result["type1"]) and math.isnan(result["total"])
