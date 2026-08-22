"""
rope3d.py — 3D rotary position encoding for the interleaved world-model rows.

WHY 3D AT ALL: 1D RoPE sees the row as a flat string — spatial grid neighbors sit
1 or 16 apart, and the same grid cell one latent frame later sits ~260 apart, so
the copy-mostly-with-local-edits structure of video is scrambled before the model
ever sees it. True (t, y, x) coordinates fix that, and because the row layout is
fixed per run, 3D RoPE is a TABLE SWAP: build cos/sin [1, S, 1, head_dim/2] and
overwrite the GPT's non-persistent rotary buffers. apply_rotary_emb, KV-cache
slicing, everything else untouched; zero core changes. This is the ONE position
regime of this exemplar — there is no 1D fallback knob, because a knob would
imply an open question and this one is settled (the research twin's entire
long-window line trains on it).

WHERE tokens sit comes from RowLayout's slot arrays — this file only says WHAT
coordinate each kind of slot gets:

    code slot, latent k    t = td*k (the last game frame the latent covers;
                           latent 0 = the given observation at t=0)
    y,x                    code j in its latent -> (j//side, j%side)
    action slot j of
    predicted block k      t = td*(k-1) + j + 0.5 — its real place on the
                           timeline, half-offset so it never collides with a
                           code token; spatially global -> (0,0)
    bos / video_start      t = -1 (before everything)
    video_end / eos / pad  t = frames + 1 (after everything)

The bases are FIXED per axis and held constant across window lengths, so a model
warm-starts cleanly 17f -> 129f -> longer under ONE rope regime (a base derived
from the current window would change under the model's feet and re-break the
warm-start). Both are far below the 1D-LLM standard 10000, which would waste most
channels on our small coordinate ranges (spatial 0-15, time 0-256):
    time base 500  -> covers ~1065 frames; spatial base 32 -> covers a 65-side grid.
Channel budget: half the rotation pairs go to time, a quarter each to y and x
(head_dim 64 -> 32 pairs -> 16/8/8). Time gets the biggest share because its
coordinate range is the one that grows — 17 frames in the quickstart, 129 in the
long-window run, while y and x stay 0..15 whatever the window.

Self-test:
    python -m exemplars.nano_world_model.rope3d
"""
import math

import numpy as np
import torch

BASE_T = 500.0
BASE_S = 32.0


def coords_from_layout(rows, seq_len=None):
    """(t,y,x) per flat position of a RowLayout's row. Returns float32 [S, 3]."""
    contract = rows.contract
    td, cpf = rows.td, rows.cpf
    side = int(round(math.sqrt(cpf)))
    end_t = float(contract["frames"] + 1)

    t = np.full(rows.row_len, end_t, dtype=np.float32)   # vend/eos/pad default
    y = np.zeros(rows.row_len, dtype=np.float32)
    x = np.zeros(rows.row_len, dtype=np.float32)
    t[0] = t[1] = -1.0                                   # bos, video_start

    j = np.arange(cpf)
    for k in range(rows.n_lat):                          # code slots, latent by latent
        sl = rows.code_slots[k * cpf:(k + 1) * cpf]
        t[sl] = float(td) * k
        y[sl] = j // side
        x[sl] = j % side
    for b in range(rows.n_blocks):                       # actions driving block b+1
        sl = rows.action_slots[b * td:(b + 1) * td]
        t[sl] = float(td) * b + np.arange(td) + 0.5

    coords = np.stack([t, y, x], axis=1)
    if seq_len is not None and seq_len != len(coords):   # pad (mirror overrides anyway)
        pad = np.tile(np.array([[end_t, 0.0, 0.0]], dtype=np.float32),
                      (max(0, seq_len - len(coords)), 1))
        coords = np.concatenate([coords, pad])[:seq_len]
    return torch.from_numpy(coords)


def axis_pairs(head_dim):
    """(t, y, x) rotation-pair counts for a head width: half to time, a quarter
    each to y and x, any remainder to time (the header says why time)."""
    n = head_dim // 2
    y = x = n // 4
    return (n - y - x, y, x)


def cos_sin_3d(coords, head_dim, pairs=None):
    """cos/sin [1,S,1,head_dim/2] matching core GPT's buffer layout: rotation-pair
    channel i gets angle coord_axis * inv_freq_axis, axes laid out [t.., y.., x..]."""
    pairs = pairs or axis_pairs(head_dim)
    assert sum(pairs) == head_dim // 2, (pairs, head_dim)
    assert min(pairs) > 0, f"head_dim {head_dim} too small to split three ways"
    chunks = []
    for axis, (n_pairs, b) in enumerate(zip(pairs, (BASE_T, BASE_S, BASE_S))):
        inv_freq = 1.0 / (b ** (torch.arange(n_pairs, dtype=torch.float32) / n_pairs))
        chunks.append(torch.outer(coords[:, axis], inv_freq))      # [S, n_pairs]
    freqs = torch.cat(chunks, dim=-1)                              # [S, head_dim/2]
    cos, sin = freqs.cos(), freqs.sin()
    return (cos[None, :, None, :].bfloat16(), sin[None, :, None, :].bfloat16())


def install_rope3d(trunk, rows):
    """Overwrite a core GPT's rotary tables with the 3D ones. The buffers are
    non-persistent, so this must run AFTER build (and after a checkpoint load,
    which leaves them alone — a freshly loaded model still carries 1D tables
    until this runs). For the diffusion objective, run BEFORE
    RowLayout.install_mirror_rope — the mirror reindexes whatever is installed."""
    cfg = trunk.config
    coords = coords_from_layout(rows, seq_len=cfg.sequence_len)
    cos, sin = cos_sin_3d(coords, cfg.n_embd // cfg.n_head)
    dev = trunk.cos.device
    trunk.cos, trunk.sin = cos.to(dev), sin.to(dev)
    return coords


if __name__ == "__main__":
    from exemplars.nano_world_model import spec
    from exemplars.nano_world_model.row_layout import RowLayout

    def _rows(frames):
        contract = spec.shape_contract(frames)
        return RowLayout(contract, video_offset=0, action_offset=0,
                         control_ids={"bos": 1, "eos": 2, "video_start": 3,
                                      "video_end": 4}, n_actions=spec.N_ACTIONS)

    # 17f spot checks (positions per row_layout: [bos, vstart, L0, a0..a3, L1, ...])
    c = coords_from_layout(_rows(17), seq_len=1408)
    assert tuple(c[0]) == (-1, 0, 0) and tuple(c[1]) == (-1, 0, 0)
    assert tuple(c[2]) == (0, 0, 0) and tuple(c[257]) == (0, 15, 15)     # L0 first/last
    assert tuple(c[258]) == (0.5, 0, 0) and tuple(c[261]) == (3.5, 0, 0)  # a0..a3
    assert tuple(c[262]) == (4, 0, 0) and tuple(c[517]) == (4, 15, 15)    # L1
    assert tuple(c[518]) == (4.5, 0, 0)                                   # a4
    assert tuple(c[1297]) == (16, 15, 15)                                 # L4 last code
    assert tuple(c[1298]) == (18, 0, 0) and tuple(c[1407]) == (18, 0, 0)  # vend..pad
    cos, sin = cos_sin_3d(c, 64)
    assert cos.shape == (1, 1408, 1, 32) and cos.dtype == torch.bfloat16
    # the split follows head width, so a differently-shaped model still works
    assert axis_pairs(64) == (16, 8, 8) and axis_pairs(128) == (32, 16, 16)
    assert cos_sin_3d(c, 128)[0].shape == (1, 1408, 1, 64)
    # norm preservation on a random rotation (bf16 tolerance)
    xx = torch.randn(1, 1408, 1, 64)
    x1, x2 = xx[..., :32], xx[..., 32:]
    y1 = x1 * cos.float() + x2 * sin.float()
    y2 = -x1 * sin.float() + x2 * cos.float()
    assert torch.allclose(xx.norm(dim=-1), torch.cat([y1, y2], -1).norm(dim=-1), atol=2e-2)
    # a 129f row gets coherent coordinates too (warm-start target)
    c129 = coords_from_layout(_rows(129))
    assert tuple(c129[2]) == (0, 0, 0) and float(c129[:, 0].max()) == 130.0
    print("rope3d self-test OK")
