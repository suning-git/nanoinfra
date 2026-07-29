"""
MotionCodec — the motion modality's LOCAL-ID producer (the codec artifact loader).

The contract the assembler/recipes expect (mirrors text's codec surface and the
earlier MotionTokenizer/MotionTokenizerFSQ adapters):

    .vocab_size            -> int   (code count = the motion band's width)
    .downsample            -> int   (frames per code)
    .d_feat                -> int   (native feature dim; rot139 -> 139)
    .rep                   -> str   (native representation name, e.g. "rot139")
    encode(features)       -> list[int]  local codes    ([T, d_feat] raw features)
    decode(codes)          -> np.ndarray [T', d_feat]   raw features

Which network the checkpoint carries is decided by its self-describing config
(quantizer choice is artifact DATA, not code structure):
    config["levels"] present -> MotionFSQ2   (FSQ2 grid quantizer)
    otherwise                -> MotionVQVAE  (EMA codebook)

Checkpoint formats accepted (two exist on disk):
    {"model", "config", "normalizer": {"mean","std"}}       (VQ studies)
    {"model", "config", "norm_mean", "norm_std"}            (FSQ studies)
save_checkpoint() writes the canonical (second) form.
"""

import numpy as np
import torch

from modalities.motion.tokenizers._convae.models import MotionFSQ2, MotionVQVAE


def _build_net(cfg):
    if "levels" in cfg:
        return MotionFSQ2(cfg["d_feat"], width=cfg["width"], code_dim=cfg["code_dim"],
                          downsample=cfg.get("downsample", 4), levels=list(cfg["levels"]))
    return MotionVQVAE(d_feat=cfg["d_feat"], n_codes=cfg["n_codes"],
                       code_dim=cfg["code_dim"], width=cfg["width"],
                       commit_weight=cfg.get("commit_weight", 0.25),
                       downsample=cfg.get("downsample", 4))


class MotionCodec:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ck["config"]
        self.device = device
        self.model = _build_net(cfg).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        if "normalizer" in ck:
            mean, std = ck["normalizer"]["mean"], ck["normalizer"]["std"]
        else:
            mean, std = ck["norm_mean"], ck["norm_std"]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
        self.rep = cfg.get("rep", "rot139")
        self.d_feat = cfg["d_feat"]
        self.downsample = cfg.get("downsample", 4)
        self.vocab_size = int(cfg["n_codes"]) if "n_codes" in cfg else self.model.n_codes
        self._cfg = cfg

    @torch.no_grad()
    def encode(self, features) -> list:
        """features [T, d_feat] raw (un-normalized) -> list[int] local codes (length T/down).

        T is truncated to a multiple of `downsample`."""
        x = np.asarray(features, dtype=np.float32)
        T = (x.shape[0] // self.downsample) * self.downsample
        x = (x[:T] - self.mean) / self.std
        xt = torch.tensor(x[None], dtype=torch.float32, device=self.device)
        return self.model.encode(xt)[0].cpu().tolist()

    @torch.no_grad()
    def decode(self, codes):
        """local codes [N] -> raw features [N*down, d_feat]."""
        c = torch.tensor(np.asarray(codes, dtype=np.int64)[None], device=self.device)
        # .float(): a caller may decode under a bf16 autocast (e.g. the energy
        # evaluator inside the trainer) — numpy() rejects BFloat16.
        feats = self.model.decode(c)[0].float().cpu().numpy()
        return feats * self.std + self.mean


def save_checkpoint(model, config: dict, norm_mean, norm_std, path: str):
    """Write a MotionCodec-loadable checkpoint (the canonical format).

    `config` must self-describe the net: d_feat/width/code_dim/downsample plus
    `levels` (FSQ2) or `n_codes` (VQ), and `rep` (native representation name)."""
    if "levels" in config:
        config = dict(config, n_codes=int(np.prod(config["levels"])))
    torch.save({"model": model.state_dict(), "config": config,
                "norm_mean": np.asarray(norm_mean, dtype=np.float32),
                "norm_std": np.asarray(norm_std, dtype=np.float32)}, path)
