"""The liger head with fp32-stored weights under bf16 autocast (2026-10).

Liger multiplies by weight.t(), a view, which autocast's weight-cast cache skips: an fp32-stored head weight was cast
to bf16 once per token chunk, and the buffer Liger sums the chunks' weight gradients into followed the weight's dtype.
LigerLMHead.loss now casts once and keeps that sum in fp32. Checked against the old call, `head.fused_ce(weight, ...)`,
which is all loss() used to be, with the loss scaled by 1/16 as the Trainer does for 16 micro-batches:

  - fp32 storage under bf16 autocast: the loss and the hidden-state gradient are bit for bit the old ones, the
    weight gradient is the old one rounded to bf16, and the weight is cast to bf16 once a call instead of once per
    chunk plus once; with a loss scale that is not a power of two (1/3), the rounded gradient is then scaled and
    rounded again in bf16 (what the comment in loss() says);
  - bf16 storage under autocast, fp32 storage without autocast, and fp32 storage under fp16 autocast: bit for bit
    the old call.
"""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from core.model.heads import LIGER_AVAILABLE, LigerLMHead, LMHead
from core.tokenization.vocab_layout import VocabLayout

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not LIGER_AVAILABLE, reason="needs CUDA + liger")

N, V, H = 512, 1024, 64          # liger cuts the N tokens into chunks of 32 (V/H = 16 of them)


def _head(dtype):
    torch.manual_seed(0)
    head = LMHead(H, V, softcap=15.0)
    with torch.no_grad():
        head.lm_head.weight.normal_(0, 0.5)      # not the zero init, under which every logit is the same
    head = head.cuda().to(dtype)
    LigerLMHead.setup(head)
    return head


def _run(head, call, autocast, hidden_dtype, scale=1 / 16, autocast_dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(4, N // 4, H, generator=g).to("cuda", hidden_dtype).requires_grad_(True)
    targets = torch.randint(0, V, (4, N // 4), generator=g).cuda()
    targets[0, :7] = VocabLayout.IGNORE_INDEX
    head.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast):
        loss = call(head, hidden, targets) * scale
    loss.backward()
    return loss.detach(), hidden.grad, head.lm_head.weight.grad


def _new(head, hidden, targets):
    return head.loss(hidden, targets)


def _old(head, hidden, targets):
    return head.fused_ce(head.lm_head.weight, hidden.reshape(-1, H), targets.reshape(-1))


class _CountWeightCasts(TorchDispatchMode):
    """bf16 copies made of the weight itself (or a view of it), not of a gradient the weight's size."""

    def __init__(self, weight):
        super().__init__()
        self.ptr, self.n = weight.untyped_storage().data_ptr(), 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if (func is torch.ops.aten._to_copy.default and out.dtype == torch.bfloat16
                and args[0].untyped_storage().data_ptr() == self.ptr):
            self.n += 1
        return out


@pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.bfloat16])
def test_fp32_storage_under_autocast_rounds_only_the_weight_gradient(hidden_dtype):
    head = _head(torch.float32)
    new_loss, new_dh, new_dw = _run(head, _new, True, hidden_dtype)
    old_loss, old_dh, old_dw = _run(head, _old, True, hidden_dtype)
    assert new_dw.dtype == torch.float32 and not torch.equal(new_dw, old_dw)
    assert torch.equal(new_loss, old_loss)
    assert torch.equal(new_dh, old_dh)
    assert torch.equal(new_dw, old_dw.bfloat16().float())


def test_fp32_storage_under_autocast_casts_the_weight_once():
    head = _head(torch.float32)
    counts = {}
    for name, call in (("new", _new), ("old", _old)):
        with _CountWeightCasts(head.lm_head.weight) as casts:
            _run(head, call, True, torch.bfloat16)
        counts[name] = casts.n
    assert counts == {"new": 1, "old": V // H + 1}, counts   # old: each chunk's weight.t(), and weight once (cached)


def test_a_scale_that_is_not_a_power_of_two_rounds_the_weight_gradient_twice():
    head = _head(torch.float32)
    _, _, unscaled = _run(head, _new, True, torch.bfloat16, scale=1.0)       # Liger skips its multiply at 1.0
    new_loss, new_dh, new_dw = _run(head, _new, True, torch.bfloat16, scale=1 / 3)
    old_loss, old_dh, _ = _run(head, _old, True, torch.bfloat16, scale=1 / 3)
    assert torch.equal(new_loss, old_loss) and torch.equal(new_dh, old_dh)
    assert torch.equal(new_dw, (unscaled * (1 / 3)).bfloat16().float())


@pytest.mark.parametrize("dtype, autocast, autocast_dtype", [(torch.bfloat16, True, torch.bfloat16),
                                                             (torch.float32, False, torch.bfloat16),
                                                             (torch.float32, True, torch.float16)])
def test_other_storage_and_autocast_combinations_are_unchanged(dtype, autocast, autocast_dtype):
    head = _head(dtype)
    hidden_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    new = _run(head, _new, autocast, hidden_dtype, autocast_dtype=autocast_dtype)
    old = _run(head, _old, autocast, hidden_dtype, autocast_dtype=autocast_dtype)
    for a, b in zip(new, old):
        assert a.dtype == b.dtype and torch.equal(a, b)
