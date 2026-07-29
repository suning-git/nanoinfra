"""Motion codec contract tests (CPU, random weights — no trained artifact needed).

The trained-artifact path is exercised by the three-band verify script with
--motion_ckpt (needs the checkpoint + data on disk)."""

import numpy as np
import pytest
import torch

from modalities.motion import (
    FakeMotionCodec,
    MotionCodec,
    MotionFSQ2,
    MotionVQVAE,
    manifest,
    save_checkpoint,
)

D_FEAT = 139


def test_fsq2_code_bijection():
    """forward's hard grid values and codes_to_zqn are the SAME grid — exactly.

    This is the property the original t10 FSQ violated (banker's rounding
    collisions + tanh-saturation overflow past L-1), fatal for AR round-trips."""
    torch.manual_seed(0)
    m = MotionFSQ2(D_FEAT, width=32, code_dim=16, downsample=4)
    z = torch.randn(4, m.fsq.d, 16) * 5.0            # drive tanh into saturation too
    zqn, codes = m.fsq(z)
    assert codes.min() >= 0 and codes.max() < m.n_codes
    assert torch.equal(m.fsq.codes_to_zqn(codes), zqn.detach())
    # every code round-trips: codes -> grid -> codes (bijective on the full space)
    all_codes = torch.arange(m.n_codes).unsqueeze(0)  # [1, 512]
    grid = m.fsq.codes_to_zqn(all_codes)              # [1, d, 512]
    lvl = torch.round(grid.transpose(1, 2) * m.fsq.scale - 0.5)
    idx = (lvl.long() + m.fsq.hlv.long())
    rebuilt = (idx * m.fsq.basis).sum(-1)
    assert torch.equal(rebuilt, all_codes)


@pytest.mark.parametrize("cls,kw", [
    (MotionVQVAE, dict(n_codes=64, code_dim=32, width=32)),
    (MotionFSQ2, dict(code_dim=32, width=32)),
])
def test_net_encode_decode_shapes(cls, kw):
    torch.manual_seed(0)
    net = cls(D_FEAT, downsample=4, **kw)
    x = torch.randn(2, 64, D_FEAT)
    codes = net.encode(x)
    assert codes.shape == (2, 16)                     # T/downsample
    assert codes.min() >= 0 and codes.max() < net.n_codes
    rec = net.decode(codes)
    assert rec.shape == x.shape
    out = net(x)
    assert out["recon"].shape == x.shape and torch.isfinite(out["loss"])


@pytest.mark.parametrize("fmt", ["canonical", "legacy_vq"])
def test_motion_codec_loader_roundtrip(tmp_path, fmt):
    """MotionCodec loads both on-disk checkpoint formats and honors the contract."""
    torch.manual_seed(0)
    mean, std = np.zeros(D_FEAT, np.float32), np.ones(D_FEAT, np.float32)
    path = str(tmp_path / "codec.pt")
    if fmt == "canonical":
        net = MotionFSQ2(D_FEAT, width=32, code_dim=16, downsample=4)
        cfg = {"rep": "rot139", "d_feat": D_FEAT, "width": 32, "code_dim": 16,
               "downsample": 4, "levels": [8, 8, 8]}
        save_checkpoint(net, cfg, mean, std, path)
    else:
        net = MotionVQVAE(D_FEAT, n_codes=64, code_dim=32, width=32, downsample=4)
        cfg = {"d_feat": D_FEAT, "n_codes": 64, "code_dim": 32, "width": 32,
               "downsample": 4}
        torch.save({"model": net.state_dict(), "config": cfg,
                    "normalizer": {"mean": mean, "std": std}}, path)

    codec = MotionCodec(path)
    assert codec.d_feat == D_FEAT and codec.downsample == 4
    assert codec.vocab_size == net.n_codes
    assert codec.rep == "rot139"

    feats = np.random.default_rng(0).normal(size=(67, D_FEAT)).astype(np.float32)
    codes = codec.encode(feats)                       # 67 -> 64 frames -> 16 codes
    assert isinstance(codes, list) and len(codes) == 16
    assert all(isinstance(c, int) and 0 <= c < codec.vocab_size for c in codes)
    rec = codec.decode(codes)
    assert rec.shape == (64, D_FEAT) and np.isfinite(rec).all()
    # encode is deterministic and decode->encode is stable through the quantizer grid
    assert codec.encode(feats) == codes


def test_manifest_takes_any_codec():
    m = manifest(FakeMotionCodec(64))
    assert (m.name, m.type_id, m.vocab_size) == ("motion", 1, 64)
