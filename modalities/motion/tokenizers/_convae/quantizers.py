"""
convae.quantizers — the two proven bottleneck options for the conv-AE family.

  - VectorQuantizerEMA — EMA-updated learned codebook + dead-code reset
    (used by every VQ tokenizer here).
  - FSQ2 — finite scalar quantization with a clean, bijective level<->index map
    (the bijection fix matters for AR round-trips —
    the original t10 FSQ had rounding collisions).

(These are conceptually orthogonal to the conv backbone — a future family could
reuse them. They live here for now because they were developed with the conv AE;
hoist to a shared quantizer lib only IF a second family actually reuses them.)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FSQ_LEVELS = [8, 8, 8]   # 512 codes (the shelf's operating point)


class VectorQuantizerEMA(nn.Module):
    """EMA-updated codebook with dead-code reset.

    Codebook is a buffer (not a Parameter) — updated by EMA, not gradients. The
    encoder gets gradient via the straight-through estimator; commitment loss pulls
    encoder outputs toward their assigned codes.
    """

    def __init__(self, n_codes: int, code_dim: int, decay: float = 0.99,
                 eps: float = 1e-5, reset_threshold: float = 1.0):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.reset_threshold = reset_threshold  # cluster_size below this -> dead -> reset

        embed = torch.randn(n_codes, code_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", embed.clone())

    def _quantize_indices(self, flat):
        # flat: [N, C] -> nearest code index per row
        # ||x - e||^2 = ||x||^2 - 2 x·e + ||e||^2
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )
        return dist.argmin(1)

    def lookup(self, indices):
        """codes [B, T] -> z_q [B, C, T]."""
        z = F.embedding(indices, self.embed)        # [B, T, C]
        return z.transpose(1, 2).contiguous()

    def forward(self, z_e):
        # z_e: [B, C, T]
        B, C, T = z_e.shape
        flat = z_e.transpose(1, 2).reshape(-1, C)   # [B*T, C]
        idx = self._quantize_indices(flat)           # [B*T]
        idx_bt = idx.view(B, T)
        z_q = self.lookup(idx_bt)                    # [B, C, T]

        if self.training:
            self._ema_update(flat, idx)

        # losses: commitment (encoder -> code). Codebook moved by EMA, not loss.
        commit = F.mse_loss(z_q.detach(), z_e)
        # straight-through: copy gradient from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        # perplexity (codebook usage diagnostic)
        with torch.no_grad():
            probs = torch.bincount(idx, minlength=self.n_codes).float()
            probs = probs / probs.sum()
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        return z_q_st, idx_bt, commit, perplexity

    @torch.no_grad()
    def _ema_update(self, flat, idx):
        onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)   # [N, K]
        n = onehot.sum(0)                                         # [K] count per code
        dw = onehot.t() @ flat                                    # [K, C] sum of assigned vecs

        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(dw, alpha=1 - self.decay)

        # Laplace smoothing so empty clusters don't blow up the normalization
        total = self.cluster_size.sum()
        cluster = (self.cluster_size + self.eps) / (total + self.n_codes * self.eps) * total
        self.embed.copy_(self.embed_avg / cluster.unsqueeze(1))

        # dead-code reset: any code used < threshold gets reseeded from a random
        # live encoder output in this batch (keeps the codebook fully utilized)
        dead = self.cluster_size < self.reset_threshold
        n_dead = int(dead.sum())
        if n_dead > 0 and flat.shape[0] >= n_dead:
            pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
            self.embed[dead] = flat[pick]
            self.embed_avg[dead] = flat[pick]
            self.cluster_size[dead] = 1.0


class FSQ2(nn.Module):
    """Finite Scalar Quantization with a clean L-level bijection (even L via a 0.5 shift).

    The original t10 FSQ computed `idx = round(zq + (L-1)/2)`; with (L-1)/2 = 3.5 that
    rounds half-integers (banker's rounding -> level collisions) and, when tanh saturates
    to ±1.0 in fp32, overflows past L-1. Harmless for continuous-latent reconstruction,
    fatal for an AR that round-trips through discrete codes. Here: for even L, integer
    levels {-L/2, …, L/2-1} via a 0.5 shift, so `idx ∈ [0, L-1]` bijectively (exactly L
    codes) and encode/decode are consistent.
    """

    def __init__(self, levels):
        super().__init__()
        L = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", L)
        self.register_buffer("hlv", L // 2)                 # e.g. 4 for L=8
        self.register_buffer("scale", (L // 2) - 0.5)        # 3.5 for L=8 -> symmetric [-1,1]
        basis = torch.cumprod(torch.tensor([1] + list(levels[:-1]), dtype=torch.long), 0)
        self.register_buffer("basis", basis)
        self.d = len(levels)
        self.n_codes = int(np.prod(levels))

    def _levels_int(self):
        return self.levels.long()

    def forward(self, z):
        """z [B,d,T] -> (zqn [B,d,T] in [-1,1], codes [B,T] in [0, n_codes))."""
        z = z.transpose(1, 2)                                # [B,T,d]
        zb = torch.tanh(z) * self.scale                      # (-scale, scale)
        lvl = torch.round(zb - 0.5)                          # integer level in {-L/2, …, L/2-1}
        q = lvl + 0.5                                        # half-integer, symmetric about 0
        zqn_hard = q / self.scale                            # in [-1, 1], L levels
        cont = zb / self.scale
        zqn = cont + (zqn_hard - cont).detach()              # straight-through
        idx = torch.minimum((lvl.long() + self.hlv.long()).clamp_(min=0), self._levels_int() - 1)
        codes = (idx * self.basis).sum(-1)                   # [B,T]
        return zqn.transpose(1, 2), codes

    def codes_to_zqn(self, codes):
        """[.,N] codes -> [.,d,N] normalized latents (exact inverse of forward's grid)."""
        idx = torch.div(codes.unsqueeze(-1), self.basis, rounding_mode="floor") % self._levels_int()
        lvl = idx.float() - self.hlv                        # {-L/2, …, L/2-1}
        zqn = (lvl + 0.5) / self.scale                       # [-1,1]
        return zqn.transpose(1, 2)
