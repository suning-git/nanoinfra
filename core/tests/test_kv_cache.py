"""
KVCache mechanics + the engine's core guarantee: cached generation reproduces
uncached generation (verified by a greedy cross-check).
Modality-free: random-weight trunk+head, integer stop ids.
"""

import pytest
import torch

from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.inference import autoregressive_generate, sample_tokens
from core.model.kv_cache import KVCache
from core.model.system import LMSystem

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def small_system(seq_len=64, vocab=128, n_layer=2, n_embd=64, n_head=2):
    torch.manual_seed(1234)
    config = GPTConfig(sequence_len=seq_len, vocab_size=vocab, n_layer=n_layer,
                       n_head=n_head, n_kv_head=n_head, n_embd=n_embd,
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
