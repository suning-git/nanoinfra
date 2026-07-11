"""
Evaluator objects for Trainer.

Each evaluator encapsulates its own data source, metrics, and eval budget.
Trainer just iterates evaluators and merges result dicts.
"""

import torch.distributed as dist

from core.evaluation.eval_loss import evaluate_loss_logits, evaluate_loss_fused
from core.evaluation.loss_eval import evaluate_loss as evaluate_loss_compat


def compute_eval_batches(eval_samples, dataset_size, device_batch_size):
    """Convert eval_samples config to number of batches per rank.

    Args:
        eval_samples: -1 for minimal (1 batch), None/0 for full dataset,
                      positive int for specific sample count
        dataset_size: Total samples in dataset
        device_batch_size: Batch size per device

    Returns:
        Number of batches each rank should run
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if eval_samples == -1:
        return 1
    elif eval_samples and eval_samples > 0:
        eval_samples = min(eval_samples, dataset_size)
        raw = eval_samples // (device_batch_size * world_size)
        if raw == 0:
            print(f"  WARNING: eval_samples={eval_samples} too small for "
                  f"batch_size={device_batch_size} * world_size={world_size}, using 1 batch")
        return max(1, raw)
    else:
        return max(1, (dataset_size // world_size) // device_batch_size)


class Evaluator:
    """Base evaluator interface.

    Scheduling: the Trainer asks `should_eval(step)` every step and calls
    evaluate() when it returns True (plus one forced full eval at the final
    step). The default policy is periodic — every `interval_steps` — unless
    `eval_at` holds an explicit set of steps (e.g. a log-spaced schedule
    computed by the orchestrator and passed through config; core never
    computes schedules). Subclasses may override should_eval() entirely.
    """

    interval_steps: int = 50        # periodic cadence, subclasses set from config
    eval_at: set | None = None      # explicit step schedule; overrides interval_steps

    def should_eval(self, step) -> bool:
        if self.eval_at is not None:
            return step in self.eval_at
        return step % self.interval_steps == 0

    def evaluate(self, model, autocast_ctx) -> dict[str, float]:
        raise NotImplementedError


class LossEvaluator(Evaluator):
    """Generic loss evaluator with switchable computation mode.

    Wraps the service layer (evaluate_loss_logits or evaluate_loss_fused)
    with dataloader, config, and metric naming.

    Args:
        dataloader: Any iterable yielding {idx, targets, target_types, ...}.
        eval_steps: Number of batches to evaluate.
        mode: 'logits' (head forward + F.cross_entropy) or 'fused' (head.loss, memory-safe).
        type_metrics: Optional {type_id: "metric/name"} for per-type loss reporting.
        token_bytes: Optional tensor for BPB computation.
        total_metric: Metric name for total loss (None = don't report).
        bpb_metric: Metric name for BPB (None = don't report).
    """

    def __init__(self, dataloader, eval_steps, mode='logits', type_metrics=None,
                 token_bytes=None, total_metric=None, bpb_metric=None):
        self.dataloader = dataloader
        self.eval_steps = eval_steps
        self.mode = mode
        self.type_metrics = type_metrics
        self.token_bytes = token_bytes
        self.total_metric = total_metric
        self.bpb_metric = bpb_metric
        self._eval_fn = evaluate_loss_fused if mode == 'fused' else evaluate_loss_logits

    def evaluate(self, model, _autocast_ctx):
        type_ids = list(self.type_metrics) if self.type_metrics else None
        # Real System (trunk + head) → the eval_loss service (logits/fused). A mock
        # per-token-loss model (unit tests) has neither → the compat helper.
        if hasattr(model, 'trunk') and hasattr(model, 'head'):
            eval_fn = self._eval_fn
        else:
            eval_fn = evaluate_loss_compat
        raw = eval_fn(
            model, self.dataloader, self.eval_steps,
            type_ids=type_ids, token_bytes=self.token_bytes,
        )
        results = {}
        if self.total_metric and "total_loss" in raw:
            results[self.total_metric] = raw["total_loss"]
        if self.bpb_metric and "bpb" in raw:
            results[self.bpb_metric] = raw["bpb"]
        if self.type_metrics and "type_losses" in raw:
            for tid, name in self.type_metrics.items():
                if tid in raw["type_losses"]:
                    results[name] = raw["type_losses"][tid]
        return results


# Backward compatibility alias
SourceLossEvaluator = LossEvaluator
