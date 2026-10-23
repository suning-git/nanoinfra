"""Unit tests for pluggable heads: naive LMHead numerics + __class__ injection.

An end-to-end training run exercises whichever arm its recipe pins, and only on a box
that has that arm's package installed. These tests pin the head's tensor contract and
prove the __class__-injection pattern directly — no liger, no FSDP, no GPU — so the
mechanism stays covered on CPU everywhere, including where the fused kernel is absent.
"""

import torch
import torch.nn.functional as F

from core.model.heads import LMHead
from core.tokenization.vocab_layout import VocabLayout


def _ref_logits(head, hidden):
    logits = head.lm_head(hidden).float()
    return head.softcap * torch.tanh(logits / head.softcap)


def test_lmhead_forward_softcap():
    torch.manual_seed(0)
    head = LMHead(8, 16)
    hidden = torch.randn(2, 3, 8)
    logits = head(hidden)
    assert logits.shape == (2, 3, 16)
    assert torch.allclose(logits, _ref_logits(head, hidden))
    assert logits.abs().max() <= head.softcap + 1e-4  # softcapped


def test_lmhead_loss_matches_cross_entropy():
    torch.manual_seed(0)
    head = LMHead(8, 16)
    hidden = torch.randn(2, 3, 8)
    targets = torch.randint(0, 16, (2, 3))
    targets[0, 1] = VocabLayout.IGNORE_INDEX  # exercise ignore_index
    ref = F.cross_entropy(
        _ref_logits(head, hidden).reshape(-1, 16), targets.reshape(-1),
        ignore_index=VocabLayout.IGNORE_INDEX,
    )
    assert torch.allclose(head.loss(hidden, targets), ref)


def test_class_injection_preserves_params_and_swaps_behavior():
    """The setup() mechanism: swaps the method table via __class__, keeps __dict__."""
    torch.manual_seed(0)
    head = LMHead(8, 16)
    w_before = head.lm_head.weight.detach().clone()

    class DoubleLossHead(LMHead):
        @classmethod
        def setup(cls, h):
            h._marker = 123
            h.__class__ = cls

        def loss(self, hidden, targets):
            return 2.0 * LMHead.loss(self, hidden, targets)

    hidden = torch.randn(2, 3, 8)
    targets = torch.randint(0, 16, (2, 3))
    base = head.loss(hidden, targets).item()

    DoubleLossHead.setup(head)

    assert isinstance(head, DoubleLossHead)                 # class swapped
    assert head._marker == 123                              # __dict__ attr added
    assert torch.equal(head.lm_head.weight, w_before)       # params untouched
    assert abs(head.loss(hidden, targets).item() - 2 * base) < 1e-5   # new behavior active
    assert torch.allclose(head(hidden), _ref_logits(head, hidden))    # inherited forward intact


def test_liger_guard_survives_a_non_import_error():
    """The optional-liger guard must catch more than ImportError.

    `liger_kernel.transformers` runs @triton.autotune at import time, so on a machine
    with no Triton driver it raises RuntimeError, not ImportError. An ImportError-only
    guard lets that through and `import core.model.heads` dies — on a cluster login
    node, which is exactly where people run import-level checks.

    Reloading the module is the only way to exercise an import-time guard, so this
    restores it afterwards: later tests (and other modules holding LMHead) must keep
    seeing the same class objects.
    """
    import builtins
    import importlib
    import core.model.heads as heads_mod

    real_import = builtins.__import__

    def raising_import(name, *args, **kwargs):
        if name == "liger_kernel.transformers":
            raise RuntimeError("0 active drivers ([]). There should only be one.")
        return real_import(name, *args, **kwargs)

    try:
        builtins.__import__ = raising_import
        reloaded = importlib.reload(heads_mod)
        assert reloaded.LIGER_AVAILABLE is False
    finally:
        builtins.__import__ = real_import
        importlib.reload(heads_mod)          # put the real classes back


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
