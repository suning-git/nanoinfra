"""Nanoinfra evaluation components (mechanisms only — each modality's own
evaluator, e.g. the text modality's TextEvaluator, is injected via the
Trainer's evaluators list)."""

from .loss_eval import evaluate_loss
from .eval_loss import evaluate_loss_logits, evaluate_loss_fused
from .evaluator import (
    Evaluator,
    LossEvaluator,
    SourceLossEvaluator,
)

__all__ = [
    "evaluate_loss",
    "evaluate_loss_logits",
    "evaluate_loss_fused",
    "Evaluator",
    "LossEvaluator",
    "SourceLossEvaluator",
]
