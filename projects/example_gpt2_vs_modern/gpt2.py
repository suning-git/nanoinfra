"""GPT-2 (minGPT "gpt-nano") architecture as a nanoinfra trunk — the 2019-era
baseline for the modern-vs-classic comparison.

Re-expresses the classic GPT-2 block at real scale in the trunk-contract shape
(forward -> hidden, `blocks`, `estimate_flops`, `Config`), so it trains through
the SAME orchestrator as the modern GPT via `model.trunk_class`. The architecture
is borrowed from the course's minimal_gpt.py (minGPT gpt-nano).

What differs from the modern core GPT (core/model/gpt.py):

  GPT-2 style (this trunk)                 modern core GPT
  ------------------------------------     -------------------------------------
  learned absolute position emb (wpe)      RoPE (no position embedding)
  LayerNorm (learnable weight + bias)      RMSNorm (no learnable params)
  tanh-GELU MLP                            ReLU^2 MLP
  biases in every Linear                   no biases
  plain multi-head attention              + QK-norm, GQA-capable

Both return normalized hidden (the trunk owns the final norm — modern GPT ends in
`norm(x)`, this ends in `ln_f`) and both feed the SAME untied LMHead, so the two
curves isolate the trunk architecture. (minGPT-nano itself does not tie the head,
so the shared untied head is faithful to the classic design too.)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import GPTConfig


def gelu(x):
    # minGPT's tanh approximation of GELU — the 2019 default.
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class GPT2Attention(nn.Module):
    """Classic causal multi-head attention: biased QKV / projection, no RoPE, no QK-norm."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # bias=True (GPT-2)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # bias=True

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # causal flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)      # bias=True
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)    # bias=True

    def forward(self, x):
        return self.c_proj(gelu(self.c_fc(x)))


class GPT2Block(nn.Module):
    """Pre-LN GPT-2 block: LN -> attn -> residual ; LN -> GELU-MLP -> residual."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)

    # signature matches the modern Block; the classic arch ignores rope / kv-cache / mask
    def forward(self, x, cos_sin=None, kv_cache=None, block_mask=None):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Trunk(nn.Module):
    """The GPT-2 (minGPT gpt-nano) architecture as a nanoinfra trunk.

    Trains only (no KV-cache inference path — the comparison is training curves).
    Config-compatible with the modern GPT (same GPTConfig), so `build_system`
    assembles it and the same LMHead un-embeds it.
    """

    Config = GPTConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),      # token embedding
            wpe=nn.Embedding(config.sequence_len, config.n_embd),    # learned abs positions
            h=nn.ModuleList([GPT2Block(config, i) for i in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                        # final norm (trunk owns it)
        ))

    @property
    def blocks(self):
        """Trunk contract: the per-layer FSDP shard-group modules."""
        return self.transformer.h

    def init_weights(self):
        # The framework builds the trunk on the `meta` device and calls this AFTER
        # `to_empty(device)` — so nothing is initialized by default and init_weights
        # must set EVERY parameter, LayerNorm included (a default nn.LayerNorm would
        # only get weight=1 / bias=0 at construction, which meta-init skips).
        self.apply(self._init_weights)
        # match the modern trunk: bf16 embeddings on CUDA (memory; optim tolerates it)
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            self.transformer.wpe.to(dtype=torch.bfloat16)

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

    def estimate_flops(self):
        """FLOPs/token for MFU. 6*matmul-params + attention term; the wte and wpe
        lookups are excluded (they are gathers, not matmuls)."""
        nparams = sum(p.numel() for p in self.parameters())
        lookup = self.transformer.wte.weight.numel() + self.transformer.wpe.weight.numel()
        l, h = self.config.n_layer, self.config.n_head
        q, t = self.config.n_embd // self.config.n_head, self.config.sequence_len
        return 6 * (nparams - lookup) + 12 * l * h * q * t

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, token_types=None, kv_cache=None, block_mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        # GPT-2 has no token-type embedding — token_types is ignored on purpose.
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x
