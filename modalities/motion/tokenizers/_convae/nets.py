"""
convae.nets — the fixed conv-autoencoder stack (T2M-GPT-style, arXiv 2301.06052).

The ResConv1d / Encoder / Decoder primitives shared by every tokenizer in the
conv-autoencoder FAMILY (both quantizer bottlenecks reuse them verbatim). Stable
physical machinery — the topology has not changed across any of this work;
only width / downsample are knobs. A genuinely different architecture (a
transformer or diffusion codec) is a SIBLING family folder, not an edit here.

    features [B, D_feat, T] --Encoder--> z_e [B, code_dim, T/down]
    z_q [B, code_dim, T/down] --Decoder--> recon [B, D_feat, T]
"""

import torch.nn as nn
import torch.nn.functional as F


class ResConv1d(nn.Module):
    """Dilated residual block (T2M-GPT uses dilations 9,3,1)."""

    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(width, width, 1),
        )

    def forward(self, x):
        return x + self.net(x)


def res_stack(width: int):
    return nn.Sequential(*[ResConv1d(width, d) for d in (9, 3, 1)])


class Encoder(nn.Module):
    def __init__(self, d_feat: int, width: int, code_dim: int, n_down: int = 2):
        super().__init__()
        self.inp = nn.Conv1d(d_feat, width, 3, padding=1)
        # n_down stride-2 downsamples = rate 2**n_down (default 2 -> rate 4)
        self.down = nn.ModuleList([
            nn.Sequential(nn.ReLU(), nn.Conv1d(width, width, 4, stride=2, padding=1), res_stack(width))
            for _ in range(n_down)
        ])
        self.out = nn.Conv1d(width, code_dim, 3, padding=1)

    def forward(self, x):
        x = self.inp(x)
        for blk in self.down:
            x = blk(x)
        return self.out(x)


class Decoder(nn.Module):
    def __init__(self, d_feat: int, width: int, code_dim: int, n_down: int = 2):
        super().__init__()
        self.inp = nn.Conv1d(code_dim, width, 3, padding=1)
        self.up = nn.ModuleList([
            nn.Sequential(res_stack(width), nn.ReLU(), nn.Conv1d(width, width, 3, padding=1))
            for _ in range(n_down)
        ])
        self.out = nn.Conv1d(width, d_feat, 3, padding=1)

    def forward(self, x):
        x = self.inp(x)
        for blk in self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = blk(x)
        return self.out(x)
