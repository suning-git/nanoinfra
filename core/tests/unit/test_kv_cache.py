"""
KVCache mechanics + the engine's core guarantee: cached generation reproduces
uncached generation (verified by a greedy cross-check).
Modality-free: random-weight trunk+head, integer stop ids.
"""

import numpy as np
import pytest
import torch

from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.inference import autoregressive_generate, sample_tokens
from core.model.kv_cache import KVCache
from core.model.system import LMSystem

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def small_system(seq_len=64, vocab=128, n_layer=2, n_embd=64, n_head=2, n_kv_head=None):
    torch.manual_seed(1234)
    config = GPTConfig(sequence_len=seq_len, vocab_size=vocab, n_layer=n_layer,
                       n_head=n_head, n_kv_head=n_kv_head or n_head, n_embd=n_embd,
                       n_token_types=3)
    trunk = GPT(config).to(DEVICE)
    head = LMHead(n_embd, vocab).to(DEVICE)   # default init (NOT zeroed) -> real logits
    return LMSystem(trunk, head).eval()


def _generate_no_cache(system, prompt_ids, prompt_types, max_new_tokens,
                       gen_token_type, stop_token=None):
    """Reference: full re-forward every step."""
    ids, types = prompt_ids, prompt_types
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
    out = []
    for _ in range(max_new_tokens):
        hidden = system.trunk(ids, token_types=types)
        logits = system.head(hidden[:, -1:]).squeeze(1)
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if stop_token is not None:
            finished |= (nxt.squeeze(-1) == stop_token)
            nxt[finished] = stop_token
        out.append(nxt)
        ids = torch.cat([ids, nxt], dim=1)
        types = torch.cat([types, torch.full_like(nxt, gen_token_type)], dim=1)
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# cache mechanics
# ---------------------------------------------------------------------------

def test_cache_pos_advances_on_last_layer_only():
    cache = KVCache(n_layer=2, batch_size=1, n_kv_head=2, head_dim=4, max_len=8)
    k = torch.randn(1, 2, 3, 4)
    fk, fv = cache.insert_kv(0, k, k)
    assert cache.get_pos() == 0 and fk.shape[2] == 3
    cache.insert_kv(1, k, k)
    assert cache.get_pos() == 3
    fk, _ = cache.insert_kv(0, k[:, :, :1], k[:, :, :1])
    assert fk.shape[2] == 4                      # 3 cached + 1 new

    cache.reset()
    assert cache.get_pos() == 0


def test_cache_overflow_and_shape_asserts():
    cache = KVCache(n_layer=1, batch_size=1, n_kv_head=2, head_dim=4, max_len=2)
    k = torch.randn(1, 2, 3, 4)
    with pytest.raises(AssertionError, match="overflow"):
        cache.insert_kv(0, k, k)
    with pytest.raises(AssertionError, match="shape mismatch"):
        cache.insert_kv(0, torch.randn(1, 3, 1, 4), torch.randn(1, 3, 1, 4))


# ---------------------------------------------------------------------------
# trunk equivalence: prefill + decode == one full forward
# ---------------------------------------------------------------------------

def test_trunk_hidden_matches_full_forward():
    system = small_system()
    trunk = system.trunk
    B, T = 2, 12
    torch.manual_seed(7)
    ids = torch.randint(0, 128, (B, T), device=DEVICE)
    types = torch.zeros_like(ids)

    full = trunk(ids, token_types=types)                     # [B, T, H]

    cache = KVCache.for_model(trunk.config, B, T)
    prefix = 5
    h_pre = trunk(ids[:, :prefix], token_types=types[:, :prefix], kv_cache=cache)
    steps = [h_pre[:, -1]]
    for t in range(prefix, T):
        h = trunk(ids[:, t:t+1], token_types=types[:, t:t+1], kv_cache=cache)
        steps.append(h[:, -1])
    stepped = torch.stack(steps, dim=1)                      # [B, T-prefix+1, H]

    assert torch.allclose(full[:, prefix-1:], stepped, atol=2e-3, rtol=1e-3), \
        f"max diff {(full[:, prefix-1:] - stepped).abs().max().item()}"


# ---------------------------------------------------------------------------
# generation: greedy cached == greedy uncached; stop semantics; reproducibility
# ---------------------------------------------------------------------------

def test_greedy_cached_matches_uncached():
    system = small_system()
    B, T, NEW = 3, 10, 20
    torch.manual_seed(42)
    prompt = torch.randint(0, 128, (B, T), device=DEVICE)
    types = torch.zeros_like(prompt)

    ref = _generate_no_cache(system, prompt, types, NEW, gen_token_type=1)
    got = autoregressive_generate(system, prompt, types, NEW,
                                  gen_token_type=1, temperature=0)
    assert torch.equal(ref, got), "cached greedy stream diverged from uncached"


def test_stop_token_and_early_stop():
    system = small_system()
    prompt = torch.randint(0, 128, (1, 8), device=DEVICE)
    types = torch.zeros_like(prompt)

    first = autoregressive_generate(system, prompt, types, 16,
                                    gen_token_type=1, temperature=0)
    stop = int(first[0, 0])
    out = autoregressive_generate(system, prompt, types, 16, gen_token_type=1,
                                  stop_token=stop, temperature=0)
    assert out.shape == (1, 1) and int(out[0, 0]) == stop   # early exit at step 1
    out2 = autoregressive_generate(system, prompt, types, 16, gen_token_type=1,
                                   stop_token=stop, temperature=0, early_stop=False)
    assert out2.shape == (1, 16)                             # budget run, stop-filled
    assert (out2 == stop).all()


def test_sampling_reproducible_with_generator():
    system = small_system()
    prompt = torch.randint(0, 128, (2, 6), device=DEVICE)
    types = torch.zeros_like(prompt)

    def run():
        g = torch.Generator(device=DEVICE).manual_seed(9)
        return autoregressive_generate(system, prompt, types, 12, gen_token_type=1,
                                       temperature=1.0, top_k=20, generator=g)
    assert torch.equal(run(), run())


def test_sample_tokens_greedy_and_topk():
    logits = torch.tensor([[0.1, 3.0, -1.0, 0.5]], device=DEVICE)
    assert int(sample_tokens(logits, temperature=0)[0, 0]) == 1
    g = torch.Generator(device=DEVICE).manual_seed(0)
    tok = sample_tokens(logits, temperature=1.0, top_k=1, generator=g)
    assert int(tok[0, 0]) == 1                               # top-1 == argmax


# ---------------------------------------------------------------------------
# rewind, and what a forward may see (causal=)
# ---------------------------------------------------------------------------

def _ids(B, T, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randint(0, 128, (B, T), device=DEVICE, generator=g)


def _poke(ids, pos):
    """The same ids with one position changed — the probe for who sees what."""
    out = ids.clone()
    out[:, pos] = (out[:, pos] + 1) % 128
    return out


def test_rewind_forgets_bitwise():
    """rewind undoes history exactly: after junk is written and rewound, a forward sees
    what it would have seen had the junk never been written. Bitwise, because both sides
    run the identical path — this is a state test, not a kernels test. Batch of 2; junk
    written causally and bidirectionally; rewind(0); an append of a different length
    after the rewind; a rewind all the way to empty."""
    trunk = small_system().trunk
    prefix, junk, nxt = _ids(2, 12, 1), _ids(2, 7, 2), _ids(2, 5, 3)

    clean = KVCache.for_model(trunk.config, 2, 64)
    trunk(prefix, kv_cache=clean)
    direct = trunk(nxt, kv_cache=clean)

    for causal in (True, False):
        dirty = KVCache.for_model(trunk.config, 2, 64)
        trunk(prefix, kv_cache=dirty)
        trunk(junk, kv_cache=dirty, causal=causal)
        dirty.rewind(7)
        dirty.rewind(0)
        assert dirty.get_pos() == 12
        assert torch.equal(direct, trunk(nxt, kv_cache=dirty)), \
            f"rewound cache (junk written causal={causal}) is not bitwise a clean one"

    dirty.rewind(dirty.get_pos())
    assert dirty.get_pos() == 0
    fresh = KVCache.for_model(trunk.config, 2, 64)
    assert torch.equal(trunk(prefix, kv_cache=fresh), trunk(prefix, kv_cache=dirty))


def test_rewind_at_capacity():
    """Fill to max_len, rewind, refill: the freed slots are usable again, and one
    position past capacity still overflows."""
    trunk = small_system().trunk
    cache = KVCache.for_model(trunk.config, 1, 16)
    trunk(_ids(1, 16, 9), kv_cache=cache)
    with pytest.raises(AssertionError, match="overflow"):
        trunk(_ids(1, 1, 10), kv_cache=cache)
    cache.rewind(5)
    trunk(_ids(1, 5, 11), kv_cache=cache)
    assert cache.get_pos() == 16
    with pytest.raises(AssertionError, match="overflow"):
        trunk(_ids(1, 1, 12), kv_cache=cache)


def test_rewind_rejects_what_it_cannot_undo():
    cache = KVCache(n_layer=1, batch_size=1, n_kv_head=2, head_dim=4, max_len=8)
    k = torch.randn(1, 2, 3, 4)
    cache.insert_kv(0, k, k)                                 # pos 3
    for n in (-1, 4):
        with pytest.raises(ValueError, match="rewind"):
            cache.rewind(n)
    with pytest.raises(TypeError):
        cache.rewind(1.0)                                    # would leave pos a float
    for n in (True, torch.tensor(1), torch.tensor([1])):     # a flag; tensors (a sync on GPU)
        with pytest.raises(TypeError, match="host int"):
            cache.rewind(n)
    assert cache.get_pos() == 3
    cache.rewind(np.int64(1))                                # NumPy integers are counts
    assert cache.get_pos() == 2 and type(cache.get_pos()) is int
    cache.rewind(2)
    assert cache.get_pos() == 0


def test_who_sees_what():
    """Visibility read off the outputs: change one input token and see which outputs
    move. Exact — an output that must not move must be bitwise unchanged — and blind to
    which kernel computed the attention. P = 6 cached tokens, then a chunk of T = 5."""
    trunk = small_system().trunk
    prefix, chunk = _ids(1, 6, 1), _ids(1, 5, 2)

    def run(chunk_ids, causal, prefix_ids=prefix):
        cache = KVCache.for_model(trunk.config, 1, 64)
        trunk(prefix_ids, kv_cache=cache)
        return trunk(chunk_ids, kv_cache=cache, causal=causal)

    def moved(a, b):
        return [not torch.equal(a[:, i], b[:, i]) for i in range(a.size(1))]

    # The chunk's LAST token: causal hides it from every earlier query; bidirectional
    # shows it to all of them.
    assert moved(run(chunk, True), run(_poke(chunk, 4), True)) == [False] * 4 + [True]
    assert moved(run(chunk, False), run(_poke(chunk, 4), False)) == [True] * 5
    # The prefix's FIRST token is seen by every query either way.
    for causal in (True, False):
        assert moved(run(chunk, causal), run(chunk, causal, _poke(prefix, 0))) == [True] * 5


@pytest.mark.parametrize("n_kv_head", [2, 1])          # plain attention, then GQA
def test_bidirectional_without_a_prefix(n_kv_head):
    """causal=False on an EMPTY cache (and with no cache) is plain bidirectional
    attention — it must not be taken for a causal full-prefix pass because Tq == Tk."""
    trunk = small_system(n_kv_head=n_kv_head).trunk
    x = _ids(1, 9, 4)
    for kv in (None, "empty"):
        def run(ids):
            cache = KVCache.for_model(trunk.config, 1, 64) if kv else None
            return trunk(ids, kv_cache=cache, causal=False)
        a, b = run(x), run(_poke(x, 8))
        assert not torch.equal(a[:, 0], b[:, 0]), \
            f"kv_cache={kv}: the first query does not see the last token"


def test_causal_is_the_default():
    trunk = small_system().trunk
    x, nxt = _ids(1, 8, 5), _ids(1, 3, 6)
    assert torch.equal(trunk(x), trunk(x, causal=True))
    a, b = KVCache.for_model(trunk.config, 1, 64), KVCache.for_model(trunk.config, 1, 64)
    assert torch.equal(trunk(x, kv_cache=a), trunk(x, kv_cache=b, causal=True))
    assert torch.equal(trunk(nxt, kv_cache=a), trunk(nxt, kv_cache=b, causal=True))


def test_visibility_is_checked_before_the_cache_is_written():
    """A refused call leaves the cache exactly as it was: the checks run before any
    layer inserts."""
    trunk = small_system().trunk
    cache = KVCache.for_model(trunk.config, 1, 64)
    trunk(_ids(1, 4, 7), kv_cache=cache)
    before = [t.clone() for t in cache.k + cache.v]
    x = _ids(1, 3, 8)
    for kw, error, msg in ((dict(causal=1), TypeError, "causal must be"),
                           (dict(causal="no"), TypeError, "causal must be"),
                           (dict(causal=True, block_mask=object()), ValueError, "not both"),
                           (dict(causal=False, block_mask=object()), ValueError, "not both")):
        with pytest.raises(error, match=msg):
            trunk(x, kv_cache=cache, **kw)
        assert cache.get_pos() == 4, f"the refused call {kw} wrote the cache"
        # Bytes, not values: the unwritten tail of a torch.empty buffer can hold NaN bit
        # patterns, and NaN != NaN would fail an untouched cache.
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                   for a, b in zip(before, cache.k + cache.v)), \
            f"the refused call {kw} wrote the cache"
    with pytest.raises(TypeError):
        trunk(x, None, cache, None, False)                   # causal is keyword-only


# ---------------------------------------------------------------------------
# chunked continuation: several tokens appended to a non-empty cache
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, pytest.param(torch.bfloat16, marks=pytest.mark.skipif(
    DEVICE != "cuda", reason="bf16 on CUDA: the flash kernel inference runs"))])
@pytest.mark.parametrize("n_kv_head", [2, 1])          # plain attention, then GQA
def test_chunked_continuation_matches_full_forward(n_kv_head, dtype):
    """Appending in chunks of several tokens computes what one forward over the whole
    sequence computes: each chunk is causal with the cached prefix in front. Chunks of
    5, 1, 7 and 3 after a prefix of 8, batch of 2. bf16 on CUDA reaches the kernel the
    model actually runs (fp32 does not); its bound is bf16-sized."""
    trunk = small_system(n_kv_head=n_kv_head).trunk
    if dtype != torch.float32:
        trunk.to(dtype)                    # the rotary tables are bf16 already and must stay so
    ids = _ids(2, 24, 21)
    full = trunk(ids)
    cache = KVCache.for_model(trunk.config, 2, 64)
    parts, lo = [], 0
    for n in (8, 5, 1, 7, 3):
        parts.append(trunk(ids[:, lo:lo + n], kv_cache=cache))
        lo += n
    got = torch.cat(parts, 1)
    atol = 2e-3 if dtype == torch.float32 else 6e-2
    assert torch.allclose(got.float(), full.float(), atol=atol, rtol=1e-3), \
        f"chunked continuation diverged from the full forward: max diff {(got.float() - full.float()).abs().max().item()}"


@pytest.mark.parametrize("n_kv_head", [2, 1])
def test_chunked_continuation_under_autocast(n_kv_head):
    """Under CUDA autocast the QK-norm runs in fp32 while v stays bf16; the chunk branch
    must take mixed dtypes (causal_lower_right alone raises on them) and still match the
    full forward."""
    trunk = small_system(n_kv_head=n_kv_head).trunk
    ids = _ids(2, 13, 21)
    cache = KVCache.for_model(trunk.config, 2, 64)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        full = trunk(ids)
        got = torch.cat([trunk(ids[:, :8], kv_cache=cache), trunk(ids[:, 8:], kv_cache=cache)], 1)
    assert torch.allclose(got.float(), full.float(), atol=5e-2), \
        f"chunked continuation under autocast diverged: max diff {(got.float() - full.float()).abs().max().item()}"


def test_chunked_continuation_compiles_in_one_graph():
    """Inside torch.compile the chunk branch must not build causal_lower_right's tensor
    subclass, which dynamo cannot trace (torch 2.12): a fullgraph compile of a chunked
    append succeeds, and agrees with eager."""
    trunk = small_system().trunk
    cache = KVCache.for_model(trunk.config, 1, 64)
    trunk(_ids(1, 8, 22), kv_cache=cache)
    chunk = _ids(1, 5, 23)
    eager = trunk(chunk, kv_cache=cache)
    cache.rewind(5)
    got = torch.compile(trunk, backend="eager", fullgraph=True)(chunk, kv_cache=cache)
    assert torch.allclose(got, eager, atol=2e-3, rtol=1e-3), \
        f"compiled chunk differs from eager: max diff {(got - eager).abs().max().item()}"
