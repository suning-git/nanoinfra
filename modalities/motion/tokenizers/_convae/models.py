"""
convae.models — the two conv-AE assemblies (backbone + a quantizer bottleneck).

  MotionVQVAE  = Encoder -> VectorQuantizerEMA -> Decoder
  MotionFSQ2   = Encoder -> proj_in -> FSQ2 -> proj_out -> Decoder

Both expose the same surface (forward -> dict, encode -> codes, decode -> feats)
so MotionCodec can drive either from a self-describing checkpoint.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modalities.motion.tokenizers._convae.nets import Decoder, Encoder
from modalities.motion.tokenizers._convae.quantizers import FSQ_LEVELS, FSQ2, VectorQuantizerEMA


class MotionVQVAE(nn.Module):
    def __init__(self, d_feat: int, n_codes: int = 512, code_dim: int = 512,
                 width: int = 512, commit_weight: float = 0.25, decay: float = 0.99,
                 downsample: int = 4):
        super().__init__()
        n_down = round(math.log2(downsample))
        assert 2 ** n_down == downsample, f"downsample must be a power of 2, got {downsample}"
        self.d_feat = d_feat
        self.n_codes = n_codes
        self.downsample = downsample          # temporal rate = 2**n_down (frames per code)
        self.commit_weight = commit_weight
        self.encoder = Encoder(d_feat, width, code_dim, n_down)
        self.vq = VectorQuantizerEMA(n_codes, code_dim, decay=decay)
        self.decoder = Decoder(d_feat, width, code_dim, n_down)

    def forward(self, x, target=None):
        """x: [B, T, D] -> dict(recon, codes, loss, ...). target defaults to x; pass a
        clean target with a noised x for denoising regularization (keeps grad intact)."""
        target = x if target is None else target
        x = x.transpose(1, 2)                        # [B, D, T]
        z_e = self.encoder(x)
        z_q, codes, commit, ppl = self.vq(z_e)
        recon = self.decoder(z_q).transpose(1, 2)    # [B, T, D]
        recon_loss = F.mse_loss(recon, target)
        loss = recon_loss + self.commit_weight * commit
        return {
            "recon": recon, "codes": codes, "loss": loss,
            "recon_loss": recon_loss.detach(), "commit": commit.detach(),
            "perplexity": ppl.detach(),
        }

    @torch.no_grad()
    def encode(self, x):
        """[B, T, D] -> codes [B, T/down] (local indices in [0, n_codes))."""
        self.eval()
        z_e = self.encoder(x.transpose(1, 2))
        flat = z_e.transpose(1, 2).reshape(-1, z_e.shape[1])
        idx = self.vq._quantize_indices(flat)
        return idx.view(x.shape[0], -1)

    @torch.no_grad()
    def decode(self, codes):
        """codes [B, T/down] -> features [B, T, D]."""
        self.eval()
        z_q = self.vq.lookup(codes)
        return self.decoder(z_q).transpose(1, 2)


class MotionFSQ2(nn.Module):
    def __init__(self, d_feat, width=512, code_dim=512, downsample=4, levels=FSQ_LEVELS):
        super().__init__()
        n_down = round(math.log2(downsample))
        self.encoder = Encoder(d_feat, width, code_dim, n_down)
        self.proj_in = nn.Conv1d(code_dim, len(levels), 1)
        self.fsq = FSQ2(levels)
        self.proj_out = nn.Conv1d(len(levels), code_dim, 1)
        self.decoder = Decoder(d_feat, width, code_dim, n_down)
        self.d_feat = d_feat
        self.downsample = downsample
        self.n_codes = self.fsq.n_codes

    def forward(self, x, target=None):
        """x: [B, T, D] -> dict(recon, codes, loss, ...) — same surface as MotionVQVAE."""
        target = x if target is None else target
        x = x.transpose(1, 2)
        z_e = self.proj_in(self.encoder(x))
        z_q, codes = self.fsq(z_e)
        recon = self.decoder(self.proj_out(z_q)).transpose(1, 2)
        recon_loss = F.mse_loss(recon, target)
        with torch.no_grad():
            probs = torch.bincount(codes.flatten(), minlength=self.n_codes).float()
            probs = probs / probs.sum()
            ppl = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        return {"recon": recon, "codes": codes, "loss": recon_loss,
                "recon_loss": recon_loss.detach(), "perplexity": ppl.detach()}

    @torch.no_grad()
    def encode(self, x):
        """x [B,T,D] normalized -> codes [B,T/down]."""
        self.eval()
        z_e = self.proj_in(self.encoder(x.transpose(1, 2)))
        _, codes = self.fsq(z_e)
        return codes

    @torch.no_grad()
    def decode(self, codes):
        """codes [B,N] -> recon [B,N*down,D] normalized."""
        self.eval()
        zqn = self.fsq.codes_to_zqn(codes)
        return self.decoder(self.proj_out(zqn)).transpose(1, 2)
