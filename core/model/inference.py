"""
Autoregressive generation — the KV-cache engine.

Operates on the System surface: `system.trunk` (ids -> hidden, kv_cache-aware)
+ `system.head` (hidden -> softcapped logits). The whole prompt is prefilled
through the cache in one forward, then one `[B, 1]` token per step — so cost
is O(prompt + new) instead of O((prompt + new)^2).

Contracts (deliberately narrow):
- Prompts are SAME-LENGTH `[B, T]`. No padding / attention-mask support; a
  variable-length consumer earns that complexity when it exists.
- `stop_token` is a GLOBAL id. Name -> id resolution (e.g.
  `control_resolver.resolve("eos")`) happens at the CALL SITE; core stays
  zero-protocol.
- Distributed-safe early stop: each step all-reduces (MIN) the local
  "everything finished" flag, so all ranks break together and FSDP forward
  counts stay equal. Return shape is `[B, <=max_new_tokens]`.
- The caller owns `system.eval()` and any autocast context.

Verified by a greedy cross-check: cached streams bit-identical to a full-reforward
reference (core/tests/test_kv_cache.py).
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from core.model.kv_cache import KVCache


def sample_tokens(logits, temperature=1.0, top_k=None, generator=None):
    """
    Sample next tokens from logits.

    Args:
        logits: [B, vocab_size] raw logits
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering (None = no filtering)
        generator: Optional torch.Generator for reproducible sampling

    Returns:
        [B, 1] sampled token ids
    """
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float('Inf'))

    if temperature > 0:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)
    else:
        return torch.argmax(logits, dim=-1, keepdim=True)


def _all_finished(finished) -> bool:
    """True iff every sequence on EVERY rank is finished (rank-synced break)."""
    done = finished.all()
    if dist.is_available() and dist.is_initialized():
        done = done.to(torch.int32)
        dist.all_reduce(done, op=dist.ReduceOp.MIN)
    return bool(done)


@torch.no_grad()
def autoregressive_generate(
    system,
    prompt_ids,        # [B, T]  same-length prompt token ids
    prompt_types,      # [B, T]  token types for the prompt (e.g. layout.classify)
    max_new_tokens,    # int     maximum tokens to generate
    gen_token_type,    # int     token type for all generated tokens (modality band)
    stop_token=None,   # int     GLOBAL id; a sequence emitting it is finished
    temperature=1.0,
    top_k=None,
    early_stop=True,   # break (rank-synced) once every sequence is finished
    generator=None,    # optional torch.Generator for reproducible sampling
):
    """
    Batched autoregressive generation through a KV cache.

    Returns:
        [B, S] generated token ids (prompt excluded), S <= max_new_tokens.
        After a sequence emits stop_token, its remaining positions are filled
        with stop_token (it keeps stepping until all sequences finish or the
        budget runs out).
    """
    B, T = prompt_ids.size()
    trunk = system.trunk
    rotary_len = trunk.cos.size(1)
    assert T + max_new_tokens <= rotary_len, \
        f"prompt {T} + max_new_tokens {max_new_tokens} exceeds rotary cache {rotary_len}"

    cache = KVCache.for_model(trunk.config, B, T + max_new_tokens)

    # Prefill: one causal forward over the whole prompt; head on the LAST
    # position only (no [B, T, V] materialization).
    hidden = trunk(prompt_ids, token_types=prompt_types, kv_cache=cache)
    logits = system.head(hidden[:, -1:]).squeeze(1)          # [B, V]

    finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
    generated = []

    for _ in range(max_new_tokens):
        next_tokens = sample_tokens(logits, temperature, top_k, generator)  # [B, 1]

        if stop_token is not None:
            finished |= (next_tokens.squeeze(-1) == stop_token)
            next_tokens[finished] = stop_token

        generated.append(next_tokens)
        if len(generated) == max_new_tokens:
            break                                            # budget spent — no extra forward
        if early_stop and stop_token is not None and _all_finished(finished):
            break

        # Decode step: one new token through the cache.
        types = torch.full_like(next_tokens, gen_token_type)
        hidden = trunk(next_tokens, token_types=types, kv_cache=cache)
        logits = system.head(hidden[:, -1:]).squeeze(1)

    return torch.cat(generated, dim=1)                       # [B, <=max_new_tokens]
