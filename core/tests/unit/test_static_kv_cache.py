"""
StaticKVCache: does the graph-friendly decode path compute what the ordinary one does?

The static path exists to be fast, and it is only worth having if it is also right.
Two things are checked, and neither needs a trained model — both are about kernels and
indexing, which do not care what the weights contain:

  1. Logit equivalence against KVCache, teacher-forced. Both caches consume the SAME
     token sequence, so a disagreement is pure numerics rather than one path having
     drifted into a different context.

     The two do NOT agree bitwise, and should not be expected to: they reach the same
     arithmetic by different kernels (sdpa against an explicit masked softmax) in
     bf16. The contract is therefore relative error plus argmax, and the tolerance is
     justified by measurement rather than taste — a mask that is off by ONE position
     scores 3.6e-1, thirty-five times the 1e-2 the correct path scores, and breaks the
     argmax. `test_a_wrong_mask_is_caught` keeps that honest: a tolerance nothing can
     fail is not a test.
  2. The pieces that a CUDA graph actually depends on: the returned buffers keep their
     shape and their address as the position advances, and the position lives on the
     device rather than in Python.

Run: pytest core/tests/unit/test_static_kv_cache.py
"""

import pytest
import torch

from core.model.gpt import GPT, GPTConfig
from core.model.kv_cache import STATIC, KVCache, StaticKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CFG = dict(sequence_len=256, vocab_size=512, n_layer=3, n_head=4, n_embd=128,
           n_token_types=1)


def _tokens(T, seed=1):
    """Deterministic input. An unseeded draw makes the argmax check flaky — some token
    sequences produce near-ties that bf16 flips — and a flaky check gets loosened until
    it means nothing rather than fixed."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, CFG["vocab_size"], (1, T), device="cuda", generator=g)


def _rel(a, b):
    """Relative error, so the bound means the same thing at any logit magnitude."""
    return float((a.float() - b.float()).abs().max() / a.float().abs().max())


def assert_argmax_agrees(ref, other, tie=1e-3):
    """Same argmax everywhere except at ties the reference itself cannot resolve.

    Demanding 100% is the wrong contract, and loosening it to a hit RATE is worse —
    that hides real disagreements among tolerated ones. What makes a disagreement
    acceptable is the MARGIN: if the reference's own top-1 and top-2 are within bf16
    resolution of each other, either answer is as correct as the other, and which one
    a kernel returns says nothing. Measured here: the correct path's only disagreement
    sits at a margin of exactly 0.0, while an off-by-one mask disagrees at margins up
    to 0.07 (see `test_a_wrong_mask_is_caught`).
    """
    ref, other = ref.float()[0], other.float()[0]
    bad = (ref.argmax(-1) != other.argmax(-1)).nonzero().flatten()
    if not len(bad):
        return
    top2 = ref.topk(2, -1).values
    margin = (top2[:, 0] - top2[:, 1]) / ref.abs().max()
    real = [int(i) for i in bad if float(margin[i]) > tie]
    assert not real, (f"argmax differs at {real} with margins "
                      f"{[round(float(margin[i]), 5) for i in real]} — above the "
                      f"{tie} tie threshold, so this is not rounding")


def _static(model, max_len=None):
    """Attach a fresh static cache and hand it back. Two steps, not one argument —
    see GPT.attach_kv_cache for why that is load-bearing rather than stylistic."""
    c = StaticKVCache.for_model(model.config, 1, max_len or CFG["sequence_len"])
    model.attach_kv_cache(c)
    return c


def _model(n_kv_head):
    torch.manual_seed(0)
    cfg = GPTConfig(n_kv_head=n_kv_head, **CFG)
    m = GPT(cfg).cuda().to(torch.bfloat16).eval()
    m.cos, m.sin = m.cos.cuda(), m.sin.cuda()
    return m


@torch.no_grad()
@pytest.mark.parametrize("n_kv_head", [4, 2])          # plain attention, then GQA
def test_matches_dynamic_cache(n_kv_head):
    model = _model(n_kv_head)
    idx = _tokens(40)

    dyn = model(idx, kv_cache=KVCache.for_model(model.config, 1, CFG["sequence_len"]))
    st = (_static(model), model(idx, kv_cache=STATIC))[1]
    assert dyn.shape == st.shape
    assert _rel(dyn, st) < 5e-2, f"relative error {_rel(dyn, st):.2e}"
    assert_argmax_agrees(dyn, st)


@torch.no_grad()
def test_error_does_not_accumulate():
    """Rounding stays put as the sequence grows; a logic error compounds. Measured
    flat at ~1e-2 from 8 to 200 positions, which is what says the gap is the kernels."""
    model = _model(4)
    for T in (8, 40, 120, 200):
        idx = _tokens(T)
        d = model(idx, kv_cache=KVCache.for_model(model.config, 1, CFG["sequence_len"]))
        s = (_static(model), model(idx, kv_cache=STATIC))[1]
        assert _rel(d, s) < 5e-2, f"T={T}: rel {_rel(d, s):.2e}"
        assert_argmax_agrees(d, s)


@torch.no_grad()
def test_a_wrong_mask_is_caught():
    """Positive control. Let each query see ONE slot too many — the smallest wrong
    mask there is — and the checks above must fail. Without this, a tolerance chosen
    to make the test pass proves nothing."""
    model = _model(4)
    idx = _tokens(40)
    bad = StaticKVCache.for_model(model.config, 1, CFG["sequence_len"])
    bad.attn_mask = lambda T, device: (
        bad.arange[None, :] <= (bad.pos + torch.arange(T, device=device))[:, None] + 1
    )[None, None]

    d = model(idx, kv_cache=KVCache.for_model(model.config, 1, CFG["sequence_len"]))
    _static(model)
    assert _rel(d, model(idx, kv_cache=STATIC)) < 5e-2        # the control's control
    model.attach_kv_cache(bad)
    assert _rel(d, model(idx, kv_cache=STATIC)) > 1e-1, \
        "an off-by-one mask slipped through — the tolerance is too loose to mean anything"


@torch.no_grad()
def test_incremental_decode_matches():
    """Prefill then one token at a time — the path an interactive session takes, and
    where an off-by-one in the write position would show up."""
    model = _model(4)
    idx = _tokens(24)
    dyn = KVCache.for_model(model.config, 1, CFG["sequence_len"])
    st = _static(model)

    model(idx, kv_cache=dyn)
    model(idx, kv_cache=STATIC)
    for _ in range(6):
        nxt = _tokens(1)
        d, s = model(nxt, kv_cache=dyn), model(nxt, kv_cache=STATIC)
        assert _rel(d, s) < 5e-2, f"diverged at pos {int(st.pos)}: rel {_rel(d, s):.2e}"
        assert_argmax_agrees(d, s)


@torch.no_grad()
def test_shapes_and_addresses_are_fixed():
    """The three properties a CUDA graph needs. Without these the graph records one
    position's buffers and replays them forever."""
    model = _model(4)
    st = _static(model)

    assert isinstance(st.pos, torch.Tensor) and st.pos.is_cuda, \
        "position must live on the device — a Python int is baked into the graph"

    model(_tokens(8), kv_cache=STATIC)
    k0, shape0 = st.k.data_ptr(), st.k.shape
    for _ in range(4):
        model(_tokens(1), kv_cache=STATIC)
        assert st.k.shape == shape0, "buffer shape changed"
        assert st.k.data_ptr() == k0, "buffer was reallocated"
    assert int(st.pos) == 12

    st.rewind(4)
    assert int(st.pos) == 8
    st.reset()
    assert int(st.pos) == 0 and st.k.data_ptr() == k0, "reset must not reallocate"


@torch.no_grad()
def test_bidirectional_mask_sees_the_whole_chunk():
    """Block diffusion denoises a chunk in place: every query in it must see the whole
    chunk, not just its own causal prefix."""
    model = _model(4)
    st = _static(model)
    model(_tokens(10), kv_cache=STATIC)

    st.bidir = True
    mask = st.attn_mask(6, "cuda")
    assert mask.shape == (1, 1, 1, CFG["sequence_len"]), "bidir mask is one row"
    assert int(mask.sum()) == 16, "every query sees prefix (10) + the whole chunk (6)"

    st.bidir = False
    causal = st.attn_mask(6, "cuda")
    assert causal.shape == (1, 1, 6, CFG["sequence_len"])
    assert [int(r.sum()) for r in causal[0, 0]] == [11, 12, 13, 14, 15, 16]
