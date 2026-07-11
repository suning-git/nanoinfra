"""
KVCache — per-layer key/value cache for autoregressive decoding.

Implements the contract core/model/gpt.py speaks (CausalSelfAttention.forward):
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
