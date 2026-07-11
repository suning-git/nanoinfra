"""A residual-free GPT — the architecture change this ablation studies.

Subclasses the reference GPT and swaps in blocks that DROP the identity path:
the transformer body becomes  x = attn(norm(x)); x = mlp(norm(x))  — no `x +`.
Everything else (attention, MLP, RoPE, the trunk contract, the FLOPs estimate)
is inherited. This is the whole architecture edit — no fork of core.

One necessary partner change. The reference GPT zero-inits every block's output
projection (`c_proj`) so each block STARTS as the identity x + 0 = x — a trick
that only makes sense WITH a residual path. Without residuals a zeroed c_proj
makes every block output 0: a dead, gradient-free network that never trains. So
we re-init those projections normally. That keeps this a clean single-variable
ablation — the only thing that differs from the baseline is the residual itself.
"""
import torch.nn as nn

from core.model.gpt import GPT, GPTConfig, Block, norm


class NoResidualBlock(Block):
    def forward(self, x, cos_sin, kv_cache, block_mask=None):
        # baseline Block:  x = x + attn(norm(x)) ;  x = x + mlp(norm(x))
        # ablated: drop the residual (identity) path — no `x +`.
        x = self.attn(norm(x), cos_sin, kv_cache, block_mask)
        x = self.mlp(norm(x))
        return x


class NoResidualGPT(GPT):
    """The reference GPT with residual connections removed. Config-compatible
    with GPT (same GPTConfig), so it drives through the same orchestrator."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer.h = nn.ModuleList(
            [NoResidualBlock(config, i) for i in range(config.n_layer)])

    def init_weights(self):
        super().init_weights()          # standard init, but zeros every c_proj
        # undo the residual-era zero-init (see module docstring) so the net trains
        for block in self.transformer.h:
            self._init_weights(block.attn.c_proj)
            self._init_weights(block.mlp.c_proj)
