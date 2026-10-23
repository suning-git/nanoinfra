"""
Evaluator objects for Trainer.

Each evaluator encapsulates its own data source, metrics, and eval budget.
Trainer just iterates evaluators and merges result dicts.
"""

from core.evaluation.eval_loss import evaluate_loss


class Evaluator:
    """Base evaluator interface.

    Scheduling: the Trainer asks `should_eval(step)` every step and calls
    evaluate() when it returns True (plus one forced full eval at the final
    step). The default policy is periodic — every `interval_steps` — unless
    `eval_at` holds an explicit set of steps (e.g. a log-spaced schedule
    computed by the orchestrator and passed through config; core never
    computes schedules). Subclasses may override should_eval() entirely.

    Precision: the Trainer passes the autocast context it trains under, and the
    EVALUATOR enters it around whatever should run under that regime (it may keep
    loader construction or fp32 metric reduction outside). Every evaluate()
    implementation, this module's included, is responsible for entering it.
    Enter it ONCE: torch.amp.autocast keeps its restore state on the instance, so
    nesting the same object leaves autocast switched on after the outer exit. A
    wrapper that delegates to LossEvaluator must not enter the context itself.
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
    """Per-position CE metrics of a System on an evaluation stream: evaluate_loss (core/evaluation/eval_loss.py)
    with a dataloader, a batch budget and the metrics.

        LossEvaluator(loader, eval_steps, metrics={"val/ce": mean_ce(), "val/bpb": bits_per_byte(token_bytes)})

    Args:
        dataloader: any iterable yielding batches with `targets`, plus whatever the metrics read (`target_types`
            for type_ce).
        eval_steps: number of batches per evaluation, the same on every rank.
        metrics: {metric name: fn(batch, ce) -> (numerator, denominator)}. core's are in
            core/evaluation/metrics.py; a project writes its own the same way. The names are fixed here, at
            construction, which is what keeps them the same on every rank.
    """

    def __init__(self, dataloader, eval_steps, metrics):
        self.dataloader = dataloader
        self.eval_steps = eval_steps
        self.metrics = metrics

    def evaluate(self, model, autocast_ctx):
        # Enter the Trainer's precision regime here (base-class contract); evaluate_loss turns gradients off.
        with autocast_ctx:
            return evaluate_loss(model, self.dataloader, self.eval_steps, self.metrics)
