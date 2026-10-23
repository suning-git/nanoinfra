"""LossEvaluator / evaluate_loss on a real System (trunk + head), with core's metrics and a project-written one.

What is checked: the total is the mean CE over every valid target across batches (not a mean of batch means),
IGNORE targets count nowhere, per-type CE follows target_types, a metric with no data anywhere reads inf, a project
metric (weighted numerator, denominator from sample_meta) reads its ratio of sums, results come back in the order of
`metrics`, a second evaluation starts from zero, what a metric may return (non-integer and NaN denominators, an
integer above 2**24), the entry checks (steps, a short loader, type_ce's target_types), and that the CE comes from
the head INSTANCE's loss_per_token (where head_ce="compiled" and FSDP bind theirs). The expected values come from the
System's own logits by hand.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from core.evaluation.eval_loss import evaluate_loss
from core.evaluation.evaluator import LossEvaluator
from core.evaluation.metrics import mean_ce, type_ce
from core.model.heads import LMHead
from core.model.system import prepare_unpacked_AR_trunk_inputs
from core.tokenization.vocab_layout import VocabLayout

IGNORE = VocabLayout.IGNORE_INDEX
V, H = 16, 8


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


def _batches():
    """Two batches with different numbers of valid targets (a mean of batch means would differ), target types 0 / 1,
    no target of type 2, and a per-position weight plus a per-sample count for the project metric."""
    torch.manual_seed(1)
    out = []
    for n_ignored in (1, 4):
        idx = torch.randint(0, V, (2, 6))
        targets = torch.randint(0, V, (2, 6))
        targets.view(-1)[:n_ignored] = IGNORE
        target_types = torch.randint(0, 2, (2, 6))
        weight = torch.rand(2, 6) + 0.5
        out.append({"idx": idx, "targets": targets, "target_types": target_types, "weight": weight,
                    "sample_meta": [{"n_predicted": 7}, {"n_predicted": 3 + n_ignored}]})
    return out


def _by_hand(system, batches):
    ces, types, weights = [], [], []
    with torch.no_grad():
        for b in batches:
            logits = system.head(system.trunk(b["idx"]))
            y = b["targets"].reshape(-1)
            keep = y != IGNORE
            ces.append(F.cross_entropy(logits.reshape(-1, V)[keep], y[keep], reduction="none"))
            types.append(b["target_types"].reshape(-1)[keep])
            weights.append(b["weight"].reshape(-1)[keep])
    ce, tt, w = torch.cat(ces), torch.cat(types), torch.cat(weights)
    n_predicted = sum(m["n_predicted"] for b in batches for m in b["sample_meta"])
    return {"total": ce.mean().item(), 0: ce[tt == 0].mean().item(), 1: ce[tt == 1].mean().item(),
            "weighted": (ce * w).sum().item() / n_predicted}


def _weighted():
    """A project metric: per-position weights, a denominator the samples declare (block diffusion's 1/t and
    n_predicted have this form)."""
    def metric(batch, ce):
        sel = batch["targets"] != IGNORE
        return (ce[sel] * batch["weight"][sel]).sum(), sum(m["n_predicted"] for m in batch["sample_meta"])
    return metric


def _evaluate(system, batches, metrics):
    return LossEvaluator(batches, eval_steps=len(batches), metrics=metrics).evaluate(system, contextlib.nullcontext())


def test_core_metrics_match_the_hand_values():
    system, batches = _TinySystem(), _batches()
    want = _by_hand(system, batches)
    res = _evaluate(system, batches, {"val/total": mean_ce(), "val/type0": type_ce(0), "val/type1": type_ce(1),
                                      "val/type2": type_ce(2)})
    assert res["val/total"] == pytest.approx(want["total"], rel=1e-5)
    assert res["val/type0"] == pytest.approx(want[0], rel=1e-5)
    assert res["val/type1"] == pytest.approx(want[1], rel=1e-5)
    assert res["val/type2"] == float("inf")                  # no valid target of type 2


def test_total_is_not_a_mean_of_batch_means():
    system, batches = _TinySystem(), _batches()
    total = _by_hand(system, batches)["total"]
    means = [_by_hand(system, [b])["total"] for b in batches]
    assert abs(sum(means) / 2 - total) > 1e-3                 # the fixture can tell the two apart
    assert _evaluate(system, batches, {"val/total": mean_ce()})["val/total"] == pytest.approx(total, rel=1e-5)


def test_a_project_metric_reads_its_ratio_of_sums():
    system, batches = _TinySystem(), _batches()
    res = _evaluate(system, batches, {"val/weighted": _weighted()})
    assert res["val/weighted"] == pytest.approx(_by_hand(system, batches)["weighted"], rel=1e-5)


def test_results_come_back_in_the_order_of_metrics():
    res = _evaluate(_TinySystem(), _batches(), {"z/ce": mean_ce(), "a/type0": type_ce(0), "m/weighted": _weighted()})
    assert list(res) == ["z/ce", "a/type0", "m/weighted"]


def test_a_second_evaluation_starts_from_zero():
    """The second evaluation reads a different batch (as TextEvaluator hands a fresh loader each time): carried-over
    sums would mix the first batch in. (Re-reading the same data could not tell: 2N / 2D is N / D.)"""
    system, (a, b) = _TinySystem(), _batches()
    ev = LossEvaluator([a], eval_steps=1, metrics={"val/total": mean_ce()})
    first = ev.evaluate(system, contextlib.nullcontext())["val/total"]
    ev.dataloader = [b]
    second = ev.evaluate(system, contextlib.nullcontext())["val/total"]
    alone = _evaluate(system, [b], {"val/total": mean_ce()})["val/total"]
    assert abs(first - alone) > 1e-3                          # the two batches read apart
    assert second == alone


def test_the_scalar_contract():
    """What a metric may return: a non-integer denominator, a NaN denominator (propagates, not read as 0 -> inf), and
    an integer above 2**24 that float32 would round (the running totals are float64)."""
    def const(num, den):
        return lambda batch, ce: (num, den)
    res = evaluate_loss(_TinySystem(), _batches()[:1], 1, {
        "fraction": const(1.5, 0.5),
        "nan_den": const(1.0, float("nan")),
        "big": const(torch.tensor(2 ** 24 + 1, dtype=torch.int64), 1),
        "none": const(0, 0),
    })
    assert res["fraction"] == 3.0
    assert res["nan_den"] != res["nan_den"]                   # NaN
    assert res["big"] == 2 ** 24 + 1
    assert res["none"] == float("inf")


def test_steps_must_be_positive():
    with pytest.raises(ValueError, match="steps must be positive"):
        evaluate_loss(_TinySystem(), _batches(), 0, {"val/total": mean_ce()})


def test_a_short_loader_is_an_error_naming_the_batch():
    with pytest.raises(RuntimeError, match="ran out after 2 of 3 batches"):
        evaluate_loss(_TinySystem(), _batches(), 3, {"val/total": mean_ce()})


def test_type_ce_needs_target_types():
    batches = [{k: v for k, v in b.items() if k != "target_types"} for b in _batches()]
    with pytest.raises(ValueError, match="no 'target_types'"):
        _evaluate(_TinySystem(), batches, {"val/type0": type_ce(0)})


def test_type_ce_refuses_target_types_of_another_shape():
    # One element would broadcast to every target and put them all in one type.
    batches = [{**b, "target_types": torch.tensor([0])} for b in _batches()]
    with pytest.raises(ValueError, match="target_types has shape"):
        _evaluate(_TinySystem(), batches, {"val/type0": type_ce(0)})


def test_the_ce_comes_from_the_head_instance():
    """head_ce="compiled" binds a compiled loss_per_token on the instance and FSDP registers it there: evaluation
    must call that attribute, not the class's method."""
    system, calls = _TinySystem(), []
    plain = system.head.loss_per_token

    def spy(hidden, targets):
        calls.append(hidden.shape)
        return plain(hidden, targets)
    system.head.loss_per_token = spy
    _evaluate(system, _batches(), {"val/total": mean_ce()})
    assert len(calls) == 2
