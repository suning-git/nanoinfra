"""Unit tests for core evaluation infrastructure."""

from core.evaluation.evaluator import Evaluator
from core.training.trainer import eval_is_due


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


# ---------------------------------------------------------------------------
# eval_is_due: the training loop's gate. Zero evaluators must mean no evaluation
# even at the last step — the loop used to take that branch regardless and print a
# bare `Step NNNNN |` with an empty result dict behind it.
# ---------------------------------------------------------------------------

def _every(n):
    ev = Evaluator()
    ev.interval_steps = n
    return ev


def test_eval_is_due_no_evaluators_is_never_due():
    for last in (False, True):
        assert not eval_is_due([], step=0, last_step=last)
        assert not eval_is_due(None, step=7, last_step=last)


def test_eval_is_due_follows_the_evaluators_cadence():
    evs = [_every(250)]
    assert eval_is_due(evs, step=0, last_step=False)      # step 0 (evaluated after its update)
    assert eval_is_due(evs, step=250, last_step=False)
    assert not eval_is_due(evs, step=251, last_step=False)


def test_eval_is_due_last_step_overrides_cadence():
    """A run that evaluates at all should end with a final number."""
    assert eval_is_due([_every(250)], step=251, last_step=True)


def test_eval_is_due_any_evaluator_is_enough():
    assert eval_is_due([_every(1000), _every(3)], step=3, last_step=False)
