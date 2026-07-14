"""Unit tests for core evaluation infrastructure."""

import math

import torch

from core.evaluation import evaluate_loss
from core.evaluation.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Evaluator scheduling (should_eval): periodic default, explicit eval_at set,
# subclass override — the Trainer asks should_eval(step) every step.
# ---------------------------------------------------------------------------

def test_should_eval_default_periodic():
    ev = Evaluator()
    ev.interval_steps = 250
    assert ev.should_eval(0)
    assert ev.should_eval(250)
    assert not ev.should_eval(251)


def test_should_eval_explicit_schedule_overrides_interval():
    ev = Evaluator()
    ev.interval_steps = 250
    ev.eval_at = {20, 27, 36}
    assert ev.should_eval(20)
    assert ev.should_eval(36)
    assert not ev.should_eval(250)  # interval ignored once eval_at is set


def test_should_eval_subclass_override():
    class EveryPowerOfTwo(Evaluator):
        def should_eval(self, step):
            return step > 0 and (step & (step - 1)) == 0

    ev = EveryPowerOfTwo()
    assert ev.should_eval(64)
    assert not ev.should_eval(65)


class MockModel(torch.nn.Module):
    """Minimal model that returns uniform per-token loss."""

    def __init__(self, loss_value=1.0):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.loss_value = loss_value

    def forward(self, x, token_types=None, targets=None, loss_reduction="none"):
        B, T = x.shape
        loss = torch.full((B, T), self.loss_value, device=x.device, dtype=torch.float32)
        if targets is not None:
            loss = torch.where(targets >= 0, loss, torch.zeros_like(loss))
        if loss_reduction == "none":
            return loss
        return loss.mean()


def test_evaluate_loss_ce_only():
    model = MockModel(loss_value=2.0)
    batches = [
        {"idx": torch.tensor([[1, 2, 3]]), "targets": torch.tensor([[2, 3, 4]])},
        {"idx": torch.tensor([[5, 6, 7]]), "targets": torch.tensor([[6, 7, 8]])},
    ]
    results = evaluate_loss(model, batches, steps=2)
    assert abs(results["total_loss"] - 2.0) < 1e-6
    assert "type_losses" not in results
    assert "bpb" not in results


def test_evaluate_loss_with_type_ids():
    model = MockModel(loss_value=1.0)
    batches = [
        {
            "idx": torch.tensor([[1, 2, 3, 4, 5]]),
            "targets": torch.tensor([[5, 12, 25, 3, 22]]),
            "target_types": torch.tensor([[0, 2, 1, 0, 1]]),
        },
    ]
    results = evaluate_loss(model, batches, steps=1, type_ids=[0, 1])
    assert abs(results["total_loss"] - 1.0) < 1e-6
    assert abs(results["type_losses"][0] - 1.0) < 1e-6
    assert abs(results["type_losses"][1] - 1.0) < 1e-6


def test_evaluate_loss_with_ignore_index():
    model = MockModel(loss_value=1.0)
    batches = [{"idx": torch.tensor([[1, 2, 3, 0, 0]]), "targets": torch.tensor([[2, 3, 4, -1, -1]])}]
    results = evaluate_loss(model, batches, steps=1)
    assert abs(results["total_loss"] - 1.0) < 1e-6


def test_evaluate_loss_with_bpb():
    model = MockModel(loss_value=1.5)
    token_bytes = torch.zeros(30, dtype=torch.int64)
    token_bytes[0:10] = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])

    batches = [
        {"idx": torch.tensor([[0, 1, 2, 3]]), "targets": torch.tensor([[1, 2, 3, 10]])},
        {"idx": torch.tensor([[4, 5, 6, 7]]), "targets": torch.tensor([[5, 6, 7, 8]])},
    ]
    results = evaluate_loss(model, batches, steps=2, token_bytes=token_bytes)
    expected_bpb = 10.5 / (math.log(2) * 15)
    assert abs(results["bpb"] - expected_bpb) < 1e-5


def test_evaluate_loss_with_loss_weights():
    model = MockModel(loss_value=1.0)
    batches = [
        {
            "idx": torch.tensor([[1, 2, 3]]),
            "targets": torch.tensor([[2, 3, 4]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 1.0]]),
        },
    ]
    results = evaluate_loss(model, batches, steps=1)
    assert abs(results["total_loss"] - 1.0) < 1e-6


def test_evaluate_loss_with_weighted_loss_weights():
    class PositionDependentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, token_types=None, targets=None, loss_reduction="none"):
            B, T = x.shape
            loss = torch.arange(1, T + 1, device=x.device, dtype=torch.float32).unsqueeze(0).expand(B, T)
            if loss_reduction == "none":
                return loss
            return loss.mean()

    model = PositionDependentModel()
    batches = [
        {
            "idx": torch.tensor([[1, 2, 3]]),
            "targets": torch.tensor([[2, 3, 4]]),
            "target_types": torch.tensor([[0, 1, 1]]),
            "loss_weights": torch.tensor([[1.0, 13.0, 13.0]]),
        },
    ]
    results = evaluate_loss(model, batches, steps=1, type_ids=[0, 1])
    assert abs(results["total_loss"] - (66.0 / 27.0)) < 1e-5
    assert abs(results["type_losses"][0] - 1.0) < 1e-5
    assert abs(results["type_losses"][1] - 2.5) < 1e-5
