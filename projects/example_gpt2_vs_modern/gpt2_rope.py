"""GPT-2 + RoPE — the single-variable ablation rung.

This is `GPT2Trunk` (gpt2.py) with EXACTLY ONE change: the learned absolute
position embedding (`wpe`) is replaced by rotary position embedding (RoPE) in
attention. Everything else stays 2019-GPT-2:

    LayerNorm (learnable weight + bias)   ·   tanh-GELU MLP
    biases in every Linear                ·   plain multi-head (no QK-norm, no GQA)

So the three-way comparison reads as a ladder:

    gpt2  (learned-abs-pos)  --+RoPE-->  gpt2_rope  --...-->  modern

The gpt2 vs gpt2_rope gap isolates the *positional scheme* alone; gpt2_rope vs
modern is everything else the modern trunk accumulates (RMSNorm, ReLU^2, no-bias,
QK-norm). RoPE is a *replacement* for learned-abs-pos, not an addition — modern
has no `wpe` either — so this is the faithful first rung, not a doubly-positioned
model.

The RoPE math (`apply_rotary_emb`) and the cos/sin precompute are borrowed
verbatim from the modern core GPT (core/model/gpt.py) so the ONLY thing that
differs between this arm and a hypothetical "gpt2 with core's exact RoPE" is
nothing — the encoding is bit-for-bit the modern one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import GPTConfig, apply_rotary_emb

# reuse the classic block-halves unchanged; only attention gains RoPE
from gpt2 import gelu, GPT2MLP


class GPT2RoPEAttention(nn.Module):
    """Classic causal multi-head attention (biased QKV / projection, no QK-norm),
    with RoPE applied to q and k — the single delta from GPT2Attention."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # bias=True (GPT-2)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # bias=True

    def forward(self, x, cos_sin):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        hd = C // self.n_head
        # keep the (B, T, n_head, head_dim) layout so RoPE broadcasts over heads,
        # then transpose to (B, n_head, T, head_dim) for sdpa
        q = q.view(B, T, self.n_head, hd)
        k = k.view(B, T, self.n_head, hd)
        v = v.view(B, T, self.n_head, hd)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)   # <-- the ONE change
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # causal flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class GPT2RoPEBlock(nn.Module):
    """Pre-LN GPT-2 block, RoPE-attention variant. Identical to GPT2Block except
    the attention takes cos_sin and applies rotary."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPT2RoPEAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)

    def forward(self, x, cos_sin=None, kv_cache=None, block_mask=None):
        x = x + self.attn(self.ln_1(x), cos_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2RoPETrunk(nn.Module):
    """GPT-2 (minGPT gpt-nano) with learned-abs-pos swapped for RoPE.

    A strict single-variable delta from GPT2Trunk: no `wpe`, RoPE in attention,
    everything else classic GPT-2. Trains only (no KV-cache inference path — the
    comparison is training curves)."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),      # token embedding
            # NO wpe — position comes entirely from RoPE
            h=nn.ModuleList([GPT2RoPEBlock(config, i) for i in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                        # final norm (trunk owns it)
        ))
        # Rotary embeddings — same precompute as the modern core GPT.
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def blocks(self):
        return self.transformer.h

    def init_weights(self):
        # Same meta-init contract as GPT2Trunk: init EVERY param (LayerNorm too),
        # AND recompute the rotary buffers (to_empty leaves them garbage).
        self.apply(self._init_weights)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)   # GPT-2 / minGPT init
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # verbatim from core/model/gpt.py so the RoPE is bit-identical to modern's
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]   # (1, T, 1, head_dim//2) for broadcast
        return cos, sin

    def estimate_flops(self):
        """FLOPs/token for MFU. wte is a lookup (excluded); no wpe to exclude now."""
        nparams = sum(p.numel() for p in self.parameters())
        lookup = self.transformer.wte.weight.numel()
        l, h = self.config.n_layer, self.config.n_head
        q, t = self.config.n_embd // self.config.n_head, self.config.sequence_len
        return 6 * (nparams - lookup) + 12 * l * h * q * t

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, token_types=None, kv_cache=None, block_mask=None):
        B, T = idx.shape
        x = self.transformer.wte(idx)
        # GPT-2 has no token-type embedding — token_types is ignored on purpose.
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = self.transformer.ln_f(x)
        return x
