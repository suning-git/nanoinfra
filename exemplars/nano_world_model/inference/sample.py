"""
sample.py — band-masked autoregressive sampling, in two reference implementations.

Over the shared vocabulary the model could emit any id, but after `video_start` only
video-band ids are legal. Masking the logits to the band enforces the grammar the row
layout trained, rather than hoping the model learned to stay inside it.

Two samplers, and the difference between them is the whole cost story:

    band          recompute the entire prefix for every token. O(n^2), no cache, no
                  state. It is here as the reference the other two are checked against.
    band_cached   core's KVCache: prefill the prefix once, then one-token forwards.
                  Same distribution, ~10x faster at interactive prefix lengths.

    band_static   the same again on core's StaticKVCache: every shape pinned, so
                  `torch.compile(mode="reduce-overhead")` captures CUDA graphs.
                  Measured 3.3x the eager static path, 2.4x the dynamic one.

All three sample identically by construction: they draw with the same generator over
the same category layout, so a given seed produces the same codes.
"""

import torch

from core.model.kv_cache import STATIC, KVCache, StaticKVCache


@torch.no_grad()
def band(system, layout, type_id, prefix_ids, stop_id, *, seq_len,
         temperature=1.0, top_k=40, seed=0, device="cuda", fixed_len=None):
    """Autoregress `type_id`-band ids after the prefix -> LOCAL code ids.

    fixed_len: emit EXACTLY this many codes and never sample the stop token — right
    for a fixed-shape codec, where a latent frame is always the same number of codes.
    Otherwise run until stop_id.
    """
    system.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    lo, hi = layout.ranges[type_id]
    allow_stop = stop_id is not None and fixed_len is None
    n = fixed_len if fixed_len is not None else seq_len - len(prefix_ids)
    seq, out = list(prefix_ids), []
    for _ in range(n):
        toks = torch.tensor([seq], dtype=torch.long, device=device)
        types = layout.classify_token_types(toks)
        logits = system.head(system.trunk(toks, token_types=types))[0, -1].float()
        nxt = _draw(logits, lo, hi, stop_id if allow_stop else None, temperature, top_k, g)
        if allow_stop and nxt == stop_id:
            break
        seq.append(nxt)
        out.append(nxt - lo)
    return out


@torch.no_grad()
def band_cached(system, layout, type_id, prefix_ids, stop_id, *, seq_len,
                temperature=1.0, top_k=40, seed=0, device="cuda", fixed_len=None,
                cache=None, collect_logits=None):
    """KV-cached `band`. Returns (codes, cache).

    `cache=None` starts fresh and prefills with prefix_ids. Passing a cache back in —
    together with ONLY the ids added since the last call — continues the sequence
    without re-reading the history, which is what interactive use needs.
    """
    system.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    lo, hi = layout.ranges[type_id]
    allow_stop = stop_id is not None and fixed_len is None
    n = fixed_len if fixed_len is not None else seq_len - len(prefix_ids)

    def fwd(ids):
        toks = torch.tensor([ids], dtype=torch.long, device=device)
        types = layout.classify_token_types(toks)
        return system.head(system.trunk(toks, token_types=types, kv_cache=cache))[0, -1].float()

    if cache is None:
        cache = KVCache.for_model(system.trunk.config, 1, seq_len)
    logits = fwd(list(prefix_ids))                      # prefill, or append new prefix
    out = []
    for _ in range(n):
        if collect_logits is not None:
            collect_logits.append(logits.detach().clone())
        nxt = _draw(logits, lo, hi, stop_id if allow_stop else None, temperature, top_k, g)
        if allow_stop and nxt == stop_id:
            break
        out.append(nxt - lo)
        logits = fwd([nxt])                             # one-token forward
    return out, cache


def _draw(logits, lo, hi, stop_id, temperature, top_k, g):
    """Mask to the band (plus the stop id, if allowed), then temperature + top-k."""
    masked = torch.full_like(logits, float("-inf"))
    masked[lo:hi] = logits[lo:hi]
    if stop_id is not None:
        masked[stop_id] = logits[stop_id]
    masked = masked / max(temperature, 1e-6)
    if top_k:
        v, _ = torch.topk(masked, min(top_k, int((masked > float("-inf")).sum())))
        masked[masked < v[-1]] = float("-inf")
    return int(torch.multinomial(torch.softmax(masked, -1), 1, generator=g))


@torch.no_grad()
def band_static(system, layout, type_id, prefix_ids, *, seq_len, temperature=1.0,
                       top_k=40, seed=0, device="cuda", fixed_len=256, cache=None,
                       collect_logits=None):
    """The static-path twin of `band_cached`, using core's StaticKVCache.

    fixed_len only, and no stop token: a world model emits a fixed number of codes per
    latent frame, so free-running until a stop id would only invite early-stop artifacts
    — and a data-dependent stop is a host sync per token, which is the overhead this
    whole path exists to remove.

    Returns (local code ids, cache). Hand the cache back in with only the NEW prefix
    tokens to continue interactively — that is how the engine appends the player's
    actions and decodes the next frame without re-reading the whole history.
    """
    system.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    lo, hi = layout.ranges[type_id]
    if cache is None:
        cache = StaticKVCache.for_model(system.trunk.config, 1, seq_len)
    system.trunk.attach_kv_cache(cache)
    tok_buf = torch.empty((1, 1), dtype=torch.long, device=device)   # fixed address

    def fwd(toks):
        types = layout.classify_token_types(toks)
        return system.head(system.trunk(toks, token_types=types, kv_cache=STATIC))[0, -1].float()

    logits = fwd(torch.tensor([list(prefix_ids)], dtype=torch.long, device=device))
    out = []
    for i in range(fixed_len):
        if collect_logits is not None:
            collect_logits.append(logits.detach().clone())
        # sampling ops REPLICATE band_cached exactly — multinomial over the same
        # category layout, so the same seed draws the same codes (acceptance a). Two
        # differences that change no math: top-k count is the constant min(top_k, band)
        # (the dynamic path's int(sum()) recomputes the same number with a GPU->CPU sync
        # per token), and the drawn token stays a GPU tensor until the segment ends (one
        # sync per segment instead of 256).
        mask = torch.full_like(logits, float("-inf"))
        mask[lo:hi] = logits[lo:hi]
        logits = mask / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, hi - lo))
            logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)   # [1], on GPU
        out.append(nxt)
        tok_buf.copy_(nxt.view(1, 1))
        logits = fwd(tok_buf)
    return [int(t) - lo for t in torch.cat(out).cpu()], cache
