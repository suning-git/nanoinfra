"""
Key/value caches for autoregressive decoding.

    KVCache        the ordinary one. Returns a GROWING slice of its buffers, so the
                   shape itself says how much is valid. Right for anything that is not
                   chasing per-token latency.

    StaticKVCache  every shape and every address fixed across steps, which is what
                   CUDA graphs require. Costs an attention mask; buys the replay of a
                   recorded launch sequence instead of re-launching every kernel. At
                   batch-1 decode that is most of the time — measured 3.5x on a 12-layer
                   768-dim model.

KVCache implements the contract core/model/gpt.py speaks (CausalSelfAttention.forward):
    get_pos()                  -> #tokens already cached (the trunk reads this ONCE
                                  per forward, before the block loop, as its rotary
                                  offset)
    insert_kv(layer_idx, k, v) -> the full cached view so far [B, n_kv_head, pos+T, hd]

Within one trunk forward every layer inserts the same T positions in order
0..n_layer-1, so the position advances exactly once per forward: on the LAST
layer's insert. Buffers are allocated lazily from the first insert's dtype/device
(so the cache follows autocast) .
"""

import torch

# Sentinel passed as `kv_cache` to select the static path. The cache OBJECT must not
# travel through forward arguments: dynamo classifies its tensors as mutated graph
# inputs and refuses to use CUDA graphs (measured: 0.833 -> 1.124 ms/token, and the
# capture is silently skipped rather than failing). Modules reach it through an
# attribute instead, which dynamo treats as a static address — see GPT.attach_kv_cache.
STATIC = "static"


class KVCache:
    def __init__(self, n_layer, batch_size, n_kv_head, head_dim, max_len):
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.max_len = max_len
        self.k = [None] * n_layer
        self.v = [None] * n_layer
        self.pos = 0

    @classmethod
    def for_model(cls, config, batch_size, max_len):
        """Size a cache from a GPTConfig."""
        return cls(config.n_layer, batch_size, config.n_kv_head,
                   config.n_embd // config.n_head, max_len)

    def get_pos(self) -> int:
        return self.pos

    def reset(self) -> None:
        """Rewind to empty; buffers are kept and overwritten."""
        self.pos = 0

    def insert_kv(self, layer_idx, k, v):
        """k, v: [B, n_kv_head, T, head_dim] for this forward's T new positions.
        Returns the full cached (k, v) view up to pos+T."""
        B, H, T, D = k.shape
        assert B == self.batch_size and H == self.n_kv_head and D == self.head_dim, \
            f"cache shape mismatch: got [{B},{H},·,{D}], cache is " \
            f"[{self.batch_size},{self.n_kv_head},·,{self.head_dim}]"
        assert self.pos + T <= self.max_len, \
            f"KVCache overflow: pos {self.pos} + T {T} > max_len {self.max_len}"

        if self.k[layer_idx] is None:
            shape = (self.batch_size, self.n_kv_head, self.max_len, self.head_dim)
            self.k[layer_idx] = torch.empty(shape, dtype=k.dtype, device=k.device)
            self.v[layer_idx] = torch.empty(shape, dtype=v.dtype, device=v.device)

        self.k[layer_idx][:, :, self.pos:self.pos + T] = k
        self.v[layer_idx][:, :, self.pos:self.pos + T] = v
        full_k = self.k[layer_idx][:, :, :self.pos + T]
        full_v = self.v[layer_idx][:, :, :self.pos + T]

        if layer_idx == self.n_layer - 1:
            self.pos += T
        return full_k, full_v


class StaticKVCache:
    """A cache with every shape and address pinned, so CUDA graphs can replay it.

    A graph records one sequence of kernel launches and replays it: same kernels, same
    addresses, same shapes, only the CONTENTS of memory may differ. Three things in the
    ordinary decode path violate that, and each of the three differences below removes
    one:

      1. THE RETURN IS THE WHOLE BUFFER, never a growing slice, and the write position
         is a 0-dim tensor on the GPU advanced in place. A Python int would be baked
         into the recorded graph, so every replay would write to the slot it held at
         record time.
      2. VALIDITY IS A MASK (`attn_mask`) rather than a shrunken k/v. Shapes must be
         fixed; mask CONTENTS may vary, and this one does.
      3. THE ROTARY LOOKUP IS AN index_select on that position tensor. Python slicing
         resolves at trace time — same problem as (1).

    Buffers are allocated eagerly: a lazily allocated one would take a new address on
    the first insert after a reset, and an address is exactly what a graph may not
    change.

    `bidir` switches the mask from causal decode to "every query sees the whole prefix
    plus the current chunk, bidirectionally", which is what a block-diffusion denoise
    step needs. A plain attribute rather than a constructor argument because it changes
    between steps of the same sequence, on the same cache.
    """

    def __init__(self, n_layer, batch_size, n_kv_head, head_dim, max_len,
                 device="cuda", dtype=torch.bfloat16):
        shape = (n_layer, batch_size, n_kv_head, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.pos = torch.zeros((), dtype=torch.long, device=device)   # on the GPU   (1)
        self.arange = torch.arange(max_len, device=device)
        self.n_layer, self.max_len = n_layer, max_len
        self.bidir = False

    @classmethod
    def for_model(cls, config, batch_size, max_len, **kw):
        """Size a cache from a GPTConfig."""
        return cls(config.n_layer, batch_size, config.n_kv_head,
                   config.n_embd // config.n_head, max_len, **kw)

    def reset(self):
        """Rewind to empty. In place, so the buffers keep their addresses."""
        self.pos.zero_()

    def rewind(self, n):
        """Forget the last n cached positions, GPU-side so a graph replay stays valid.

        A block-diffusion denoise loop inserts the noisy block's keys and values on
        every step and must forget them before the next one, and before the final clean
        insert — otherwise the block attends to its own earlier guesses.
        """
        self.pos.sub_(n)
        # Same justification as insert_kv's overflow assert: unchecked, a rewind past
        # zero surfaces later as an anonymous device-side index assert that poisons
        # the CUDA context. Async, so it costs no sync.
        torch._assert_async((self.pos >= 0).all(),
                            "StaticKVCache underflow: rewind past position 0")

    def insert_kv(self, layer_idx, k, v):
        # Overflow is checked on the DEVICE. KVCache can afford a Python assert because
        # its position is a Python int; reading this one would sync the GPU on every
        # insert, which is the cost this class exists to avoid. Without the check an
        # overflow still fails — index_copy_ traps it — but as a device-side assert
        # that poisons the CUDA context and names neither the cache nor the limit.
        torch._assert_async((self.pos + k.size(2) <= self.max_len).all(),
                            "StaticKVCache overflow: pos + T > max_len")
        idx = self.pos + torch.arange(k.size(2), device=k.device)
        self.k[layer_idx].index_copy_(2, idx, k)
        self.v[layer_idx].index_copy_(2, idx, v)
        if layer_idx == self.n_layer - 1:      # once per trunk forward, as KVCache does
            self.pos.add_(k.size(2))
        return self.k[layer_idx], self.v[layer_idx]            # FULL buffers        (1)

    def rotary(self, cos, sin, T):
        """The rotary tables for this forward's T positions, tensor-indexed.     (3)"""
        positions = self.pos + torch.arange(T, device=cos.device)
        return cos.index_select(1, positions), sin.index_select(1, positions)

    def attn_mask(self, T, device):
        """[1, 1, T, max_len] bool — shape fixed, contents follow `pos`.         (2)

        Built ONCE per trunk forward, by GPT.forward before the block loop, and handed
        to every block. Not once per layer: `pos` advances on the last layer's insert,
        so a mask built inside the attention would be right for every layer but the last.
        """
        if self.bidir:
            # Denoise chunk: every query attends to slots < pos + T, i.e. the clean
            # prefix plus the whole current block, bidirectionally.
            return (self.arange[None, :] < (self.pos + T))[None, None]
        # Causal decode: the query at global position p attends to slots <= p.
        positions = self.pos + torch.arange(T, device=device)
        return (self.arange[None, :] <= positions[:, None])[None, None]
