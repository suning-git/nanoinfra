"""What `clip_gradients` adds on top of torch — and the one torch property it bets on.

torch already implements the abort (`clip_grad_norm_(..., error_if_nonfinite=True)`),
so testing that it aborts would be testing torch. These tests cover only:

  1. the property we DEPEND ON but torch does not promise (see below);
  2. the three behaviors that are ours: the step number, the fallback abort, and
     the fact that opting out is not silent.

(1) is the reason this file exists. torch documents `error_if_nonfinite` as "an error
is thrown if the total norm ... is nan, inf, or -inf" — and says NOTHING about the
gradients. Our whole design hangs on it throwing BEFORE the gradients are scaled, so
that `.grad` still holds raw per-parameter values and a post-mortem can find which
module produced the nan. That is an implementation detail, not a documented contract:
a torch release could move the check after the clip and every one of our runs would
quietly start destroying its own evidence.

It is also the mistake we already made. The first version of this guard checked the
norm AFTER `clip_grad_norm_` returned — which reads as equivalent, and is not: a nan
norm makes the scale factor nan, so every gradient in the model comes back nan.
Delete the ordering test and the next person rewrites it that way again.
"""

import math

import pytest
import torch

from core.training.trainer import clip_gradients


class _FakeSystem:
    """Only `parameters()` is used by clip_gradients."""

    def __init__(self, grads):
        self._params = []
        for g in grads:
            p = torch.nn.Parameter(torch.tensor([1.0]))
            p.grad = torch.tensor([g])
            self._params.append(p)

    def parameters(self):
        return self._params


def test_abort_leaves_the_gradients_raw():
    """THE reason this file exists — an undocumented torch property we rely on.

    Fails if torch ever moves the check after the clip, and fails if someone
    rewrites clip_gradients to check the returned norm instead.
    """
    system = _FakeSystem([float("nan"), 2.0, 3.0])
    with pytest.raises(RuntimeError):
        clip_gradients(system, max_grad_norm=1.0, step=722)
    survived = [p.grad.item() for p in system.parameters()]
    assert math.isnan(survived[0])
    assert survived[1:] == [2.0, 3.0], f"gradients were scaled before the abort: {survived}"


def test_message_carries_the_step():
    """Ours: torch's message has no step number, and the step is the first thing
    a post-mortem needs."""
    with pytest.raises(RuntimeError, match="step 722"):
        clip_gradients(_FakeSystem([float("nan"), 2.0]), max_grad_norm=1.0, step=722)


def test_aborts_even_if_torchs_check_does_not_fire(monkeypatch):
    """Ours: the guarantee is "the optimizer never steps on a non-finite gradient",
    and it must not depend on one upstream implementation detail holding."""
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_",
                        lambda *a, **k: torch.tensor(float("nan")))
    with pytest.raises(RuntimeError, match="did not catch it"):
        clip_gradients(_FakeSystem([1.0]), max_grad_norm=1.0, step=722)


def test_opting_out_is_not_silent(capsys):
    """Ours: opting out is legal; opting out quietly is how the original incident
    hid for 277 steps."""
    system = _FakeSystem([float("nan"), 2.0])
    norm = clip_gradients(system, max_grad_norm=1.0, step=722, abort_on_nonfinite=False)
    assert math.isnan(norm)
    out = capsys.readouterr().out
    assert "722" in out and "abort_on_nonfinite is off" in out
