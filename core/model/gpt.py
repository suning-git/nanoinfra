"""
GPT model for Nanoinfra
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- FlexAttention support for flexible attention patterns (reserved for future use)
- Token type embeddings for optional typed-token conditioning
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# FlexAttention for flexible attention patterns (prefix-LM, document masking, etc.)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    n_token_types: int = 3  # Number of token types (0=text, 1=motion, 2=control)


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out


def create_flex_attention_mask(batch_size, seq_len, prefix_lengths, device):
    """
    Create a FlexAttention block mask for prefix-LM style attention.
    Reserved for future use (e.g., block-wise causal attention).

    Args:
        batch_size: Batch size
        seq_len: Sequence length (must be multiple of 128 for FlexAttention)
        prefix_lengths: Tensor of shape [B] with prefix length for each sample.
                       Tokens within prefix use full (bidirectional) attention.
                       Tokens after prefix use causal attention but can attend to prefix.
        device: Device to create the mask on

    Returns:
        BlockMask for use with flex_attention

    Attention pattern visualization (prefix_length=3, seq_len=6):
         kv: 0  1  2  3  4  5
    q:     ┌──────────────────
      0    │ 1  1  1  0  0  0   ← full attention (prefix)
      1    │ 1  1  1  0  0  0   ← full attention (prefix)
      2    │ 1  1  1  0  0  0   ← full attention (prefix)
      3    │ 1  1  1  1  0  0   ← attend prefix + causal
      4    │ 1  1  1  1  1  0   ← attend prefix + causal
      5    │ 1  1  1  1  1  1   ← attend prefix + causal
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention not available. Requires PyTorch 2.5+")

    def mask_mod(b, h, q_idx, kv_idx):
        prefix_len = prefix_lengths[b]
        # Within prefix: full bidirectional attention
        q_in_prefix = q_idx < prefix_len
        kv_in_prefix = kv_idx < prefix_len
        # After prefix: causal attention, but can attend to prefix
        causal = q_idx >= kv_idx
        # Combine: if both in prefix -> full attention; otherwise -> causal + can see prefix
        return (q_in_prefix & kv_in_prefix) | (kv_in_prefix) | causal

    return create_block_mask(mask_mod, B=batch_size, H=None,
                            Q_LEN=seq_len, KV_LEN=seq_len, device=device)


def create_full_attention_mask(batch_size, seq_len, device):
    """
    Create a FlexAttention block mask for full bidirectional attention.
    Reserved for future use (e.g., encoding motion inputs into embeddings).

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to create the mask on

    Returns:
        BlockMask for use with flex_attention
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention not available. Requires PyTorch 2.5+")

    def mask_mod(b, h, q_idx, kv_idx):
        return True  # All positions attend to all positions

    return create_block_mask(mask_mod, B=batch_size, H=None,
                            Q_LEN=seq_len, KV_LEN=seq_len, device=device)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Group Query Attention — a MODULE CONSTANT, branched on in forward so the
        # sdpa call passes a literal (passing a runtime bool becomes a SymBool
        # under compile+FSDP and graph-breaks; measured -5-8% on 2-GPU compiled).
        self.enable_gqa = self.n_kv_head != self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache, block_mask=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values. Multiple modes supported.
        # self.enable_gqa is branched on (not passed as a value) so each sdpa call
        # sees a literal — keeps torch.compile to ONE graph under FSDP.
        if block_mask is not None:
            # FlexAttention mode: use custom attention pattern (e.g., prefix-LM)
            # FlexAttention doesn't natively support GQA yet, so we expand KV heads
            if self.enable_gqa:
                n_rep = self.n_head // self.n_kv_head
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            y = flex_attention(q, k, v, block_mask=block_mask)
        elif kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            if self.enable_gqa:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            if self.enable_gqa:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            if self.enable_gqa:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, block_mask=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, block_mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    # Trunk contract, part 1: the config class this trunk is built from.
    # Loaders recover a blueprint from a checkpoint via trunk_cls.Config
    # (config_from_meta) without core naming any concrete config type.
    Config = GPTConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        # Token type embeddings for optional typed-token conditioning
        self.type_emb = nn.Embedding(config.n_token_types, config.n_embd)

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # zero out type embeddings (start with no type bias)
        torch.nn.init.zeros_(self.type_emb.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            self.type_emb.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # original: use a range of frequencies from high to low
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # use only the lowest frequency (highest channel index gives lowest freq)
        # lowest_freq = 1.0 / (base ** ((head_dim - 2) / head_dim))
        # inv_freq = lowest_freq * torch.ones(head_dim // 2, dtype=torch.float32, device=device)
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    @property
    def blocks(self):
        """Trunk contract, part 2: the per-layer modules, one FSDP shard group
        each (wavefront unshard). Where they live internally (transformer.h)
        is this class's private layout; assembly code only sees `blocks`."""
        return self.transformer.h

    def estimate_flops(self):
        """Trunk contract, part 3: FLOPs/token of the trunk (for MFU).

        6 * matmul-params + attention term; wte is a lookup, not a matmul,
        so its params are excluded (type_emb kept for parity with the
        historical system-level formula).
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h = self.config.n_layer, self.config.n_head
        q, t = self.config.n_embd // self.config.n_head, self.config.sequence_len
        return 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, token_types=None, kv_cache=None, block_mask=None):
        """
        Transformer body only (the "trunk") — returns hidden_states [B, T, H].

        Post-forward processing (loss, logits) lives in a separate head module
        (core/model/heads.py), assembled with the trunk by LMSystem.

        Args:
            idx: [B, T] token ids
            token_types: [B, T] token type ids (optional)
            kv_cache: KV cache for inference
            block_mask: FlexAttention mask (optional)
        """
        B, T = idx.size()
        device = idx.device

        x = self.transformer.wte(idx)

        if token_types is not None:
            x = x + self.type_emb(token_types)

        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        assert device == self.cos.device, f"Device mismatch: {device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be bfloat16"

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache, block_mask)
        x = norm(x)
        return x

