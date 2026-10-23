"""LMHead.loss_per_token: unreduced CE with weighting left to the objective.

CPU tests cover the output contract and Liger's grad-mode gate (refused with gradient
recording, served without). GPU tests cover Liger's no-grad values, compiled values and
weighted gradients, peak-memory reduction at a fixed shape, and build_system's method
binding and dynamic N support. Peak allocation measures
the memory benefit; it does not identify which intermediate tensors materialize.
"""

import contextlib

import pytest
import torch
import torch._dynamo

from core.model.gpt import GPT, GPTConfig
from core.model.heads import LIGER_AVAILABLE, LigerLMHead, LMHead
from core.tokenization.vocab_layout import VocabLayout

IGNORE = VocabLayout.IGNORE_INDEX
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


# ---- the contract (CPU) ------------------------------------------------------------

def test_mean_over_supervised_positions_is_the_scalar_loss():
    torch.manual_seed(0)
    head = LMHead(8, 16)
    hidden = torch.randn(2, 3, 8)
    targets = torch.randint(0, 16, (2, 3))
    targets[0, 1] = IGNORE
    per = head.loss_per_token(hidden, targets)
    assert per.shape == (6,)                          # flattened, one per position
    assert per[1].item() == 0.0                       # the ignored position
    valid = targets.reshape(-1) != IGNORE
    torch.testing.assert_close(per[valid].mean(), head.loss(hidden, targets))


def _liger_head(dim=8, vocab=16):
    head = LMHead(dim, vocab)
    if LIGER_AVAILABLE:
        LigerLMHead.setup(head)
    else:
        head.__class__ = LigerLMHead                  # what setup() does after building fused_ce*
    return head


def test_liger_arm_refuses_with_gradient_recording():
    """Liger's reduction='none' backward would get per-token weights wrong, so a training call is refused —
    including one whose hidden needs no gradient (trunk frozen) while the head's weight does."""
    head = _liger_head()
    targets = torch.randint(0, 16, (2, 3))
    for hidden in (torch.randn(2, 3, 8, requires_grad=True), torch.randn(2, 3, 8)):
        with pytest.raises(NotImplementedError, match="head_ce='compiled'"):
            head.loss_per_token(hidden, targets)


def test_liger_arm_serves_without_gradient_recording_on_cpu():
    """Off CUDA the liger head falls back to the naive per-token CE, behind the same gate."""
    head = _liger_head()
    hidden, targets = torch.randn(2, 3, 8), torch.randint(0, 16, (2, 3))
    with torch.no_grad():
        torch.testing.assert_close(head.loss_per_token(hidden, targets), LMHead.loss_per_token(head, hidden, targets))


@gpu
@pytest.mark.skipif(not LIGER_AVAILABLE, reason="needs liger")
@pytest.mark.parametrize("dtype,autocast", [(torch.float32, True), (torch.float32, False), (torch.bfloat16, True)])
def test_liger_arm_without_gradients_matches_naive(dtype, autocast):
    """Evaluation's entry on the liger head: the fused kernel's per-token CE (no gradient computed) against the
    naive one, IGNORE positions zero, a whole-IGNORE batch all zeros, and a hidden that was made with
    requires_grad=True giving the same values (detached inside, so Liger computes no input gradient)."""
    torch.manual_seed(0)
    naive = LMHead(256, 4096).cuda().to(dtype)
    with torch.no_grad():
        naive.lm_head.weight.normal_(0, 0.05)
    liger = LMHead(256, 4096).cuda().to(dtype)
    liger.load_state_dict(naive.state_dict())
    LigerLMHead.setup(liger)
    # Values cannot tell whether Liger also computed gradients (it computes them in its forward whenever its input
    # requires one), so look at what reaches the kernel.
    reached = []
    liger.fused_ce_per_token.register_forward_pre_hook(lambda module, args: reached.append(args[1].requires_grad))
    hidden = torch.randn(4, 64, 256, device="cuda", dtype=dtype)
    targets = torch.randint(0, 4096, (4, 64), device="cuda")
    targets[0, :10] = IGNORE
    ctx = torch.autocast("cuda", torch.bfloat16) if autocast else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        want = naive.loss_per_token(hidden, targets).float()
        got = liger.loss_per_token(hidden, targets).float()
        from_grad_hidden = liger.loss_per_token(hidden.clone().requires_grad_(True), targets).float()
        all_ignored = liger.loss_per_token(hidden, torch.full_like(targets, IGNORE))
    assert got.shape == want.shape == (4 * 64,)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert (got[:10] == 0).all()
    assert torch.equal(from_grad_hidden, got)
    assert reached == [False, False, False]             # the requires_grad=True hidden was detached on the way
    assert (all_ignored == 0).all()


# ---- the compiled arm (GPU) --------------------------------------------------------

N, V, H = 8192, 32768, 512        # a realistic vocabulary exposes peak-memory regressions


def _pair():
    """Two heads with the same weights: eager, and compiled the way model_setup binds it."""
    torch.manual_seed(0)
    eager = LMHead(H, V).cuda()
    with torch.no_grad():
        eager.lm_head.weight.normal_(0, 0.02)
    compiled = LMHead(H, V).cuda()
    compiled.load_state_dict(eager.state_dict())
    compiled.loss_per_token = torch.compile(compiled.loss_per_token, dynamic=True)
    return eager, compiled


def _weighted_backward(head, hidden, targets, w, autocast):
    """The objective's construction: weights times the vector, summed, backward."""
    hidden = hidden.clone().requires_grad_(True)
    head.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
        per = head.loss_per_token(hidden, targets)
    (per * w).sum().backward()
    return per.detach(), head.lm_head.weight.grad.clone(), hidden.grad.clone()


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


@gpu
@pytest.mark.parametrize("autocast,grad_tol", [(False, 1e-5), (True, 1e-2)])
def test_compiled_arm_agrees_with_eager(autocast, grad_tol):
    """Compare values and nonuniformly weighted gradients, allowing dtype rounding."""
    torch._dynamo.reset()
    eager, compiled = _pair()
    hidden = torch.randn(N, H, device="cuda")
    targets = torch.randint(0, V, (N,), device="cuda")
    w = torch.rand(N, device="cuda") * 4 + 0.5
    per_e, gw_e, gh_e = _weighted_backward(eager, hidden, targets, w, autocast)
    per_c, gw_c, gh_c = _weighted_backward(compiled, hidden, targets, w, autocast)
    assert _rel(per_c, per_e) < 1e-5
    assert _rel(gw_c, gw_e) < grad_tol, "weight gradient"
    assert _rel(gh_c, gh_e) < grad_tol, "hidden gradient"


@gpu
def test_compiled_arm_reduces_peak_memory():
    """Check the warmed-up forward/backward peak at this shape and dtype.

    This guards a memory benefit that numeric agreement alone cannot establish.
    It does not assert a particular intermediate allocation or compiler strategy.
    """
    torch._dynamo.reset()
    eager, compiled = _pair()
    hidden = torch.randn(N, H, device="cuda")
    targets = torch.randint(0, V, (N,), device="cuda")
    w = torch.ones(N, device="cuda")

    def peak(head):
        _weighted_backward(head, hidden, targets, w, autocast=True)     # warm-up
        head.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _weighted_backward(head, hidden, targets, w, autocast=True)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    assert peak(compiled) < peak(eager) / 3


# ---- the binding model_setup performs (GPU) ----------------------------------------

TINY = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2,
                 n_head=2, n_kv_head=2, n_embd=16, n_token_types=2)


@contextlib.contextmanager
def _assembled(head_ce):
    """The head build_system hands out for a tiny GPT on this one device, the way an
    orchestrator gets it."""
    from core.training.model_setup import build_system
    yield build_system(GPT, TINY, head_ce=head_ce, parallel="ddp")["system"].head


@gpu
def test_build_system_binds_the_arm():
    """"compiled" binds loss_per_token as a dynamo wrapper next to loss; "naive" leaves
    the class method alone; "liger" hands out a head that refuses it with gradient recording."""
    torch._dynamo.reset()
    hidden = torch.randn(3, TINY.n_embd, device="cuda")
    targets = torch.randint(0, TINY.vocab_size, (3,), device="cuda")
    autocast = torch.autocast("cuda", torch.bfloat16)      # how a training step calls it

    with _assembled("compiled") as head:
        assert hasattr(head.loss_per_token, "_torchdynamo_orig_callable")
        with autocast:
            assert head.loss_per_token(hidden, targets).shape == (3,)

    with _assembled("naive") as head:
        assert "loss_per_token" not in head.__dict__       # the plain class method
        with autocast:
            assert head.loss_per_token(hidden, targets).shape == (3,)

    if LIGER_AVAILABLE:
        with _assembled("liger") as head:
            with pytest.raises(NotImplementedError), autocast:
                head.loss_per_token(hidden, targets)


@gpu
def test_shipped_binding_takes_a_new_n_without_recompiling():
    """Check dynamic N support on the actual build_system binding. Block diffusion's
    masked position count varies per batch; these two counts must share a graph.
    """
    torch._dynamo.reset()
    with _assembled("compiled") as head:

        def call(n):
            hidden = torch.randn(n, TINY.n_embd, device="cuda", requires_grad=True)
            targets = torch.randint(0, TINY.vocab_size, (n,), device="cuda")
            with torch.autocast("cuda", torch.bfloat16):
                head.loss_per_token(hidden, targets).sum().backward()

        call(64)
        with torch._dynamo.config.patch(error_on_recompile=True):
            call(23)
