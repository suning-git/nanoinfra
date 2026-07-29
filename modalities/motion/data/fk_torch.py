"""
Differentiable (torch) SMPL forward kinematics for rot139 features -> root-relative joint
positions. A torch port of the numpy FK (smpl_body.smpl_fk + sixd_to_matrix), so a
POSITION-space loss can be backpropagated into a VQ-VAE that is otherwise trained on rotation MSE.

Why root-relative: pose quality (the stuck ~11.5 cm root-rel MPJPE) depends only on the local
rotations + skeleton, not the root trajectory. FK with trans=0 and subtract the root -> the exact
quantity recon_eval reports as "root-relative MPJPE". (Drift is a separate, trajectory problem.)

Validated against the numpy recon_eval path (smpl_to_rot139.features_to_smpl + smpl_body.smpl_fk).

Returns positions in METERS (SMPL native); x100 for cm.
"""

import os
import sys

import torch

_PKG = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_PKG, "paths.py")) and _PKG != os.path.dirname(_PKG):
    _PKG = os.path.dirname(_PKG)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import paths  # noqa: F401,E402  (registers package sub-dirs on sys.path)
import smpl_body as B  # noqa: E402

_CONSTS = {}


def _consts(device):
    """Cache (J_rest [22,3], parents list) on the given device."""
    key = str(device)
    if key not in _CONSTS:
        J, parents = B.load_body_model("neutral")
        _CONSTS[key] = (torch.tensor(J, dtype=torch.float32, device=device),
                        [int(p) for p in parents])
    return _CONSTS[key]


def sixd_to_matrix(d6):
    """[...,6] -> [...,3,3] via Gram-Schmidt (Zhou et al.). Differentiable."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / (a2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def fk_rootrel(local_R, J_rest, parents):
    """local_R [...,22,3,3] (trans=0) -> root-relative global positions [...,22,3] (meters)."""
    n = local_R.shape[-3]
    gR = [None] * n
    gp = [None] * n
    gR[0] = local_R[..., 0, :, :]
    gp[0] = J_rest[0].expand(local_R.shape[:-3] + (3,))     # absolute value cancels (root-rel)
    for j in range(1, n):
        p = parents[j]
        gR[j] = gR[p] @ local_R[..., j, :, :]
        offset = J_rest[j] - J_rest[p]                      # [3]
        gp[j] = gp[p] + torch.einsum("...ij,j->...i", gR[p], offset)
    pos = torch.stack(gp, dim=-2)                           # [...,22,3]
    return pos - pos[..., 0:1, :]                           # subtract root -> root-relative


def features_to_rootrel(feats):
    """rot139 features [...,139] (UN-normalized) -> root-relative joint positions [...,22,3] (m)."""
    J_rest, parents = _consts(feats.device)
    d6 = feats[..., :132].reshape(feats.shape[:-1] + (22, 6))
    R = sixd_to_matrix(d6)
    return fk_rootrel(R, J_rest, parents)


# --------------------------------------------------------------- validation
if __name__ == "__main__":
    import numpy as np
    import dataset as md
    import smpl_to_rot139 as conv

    J_rest_np, parents_np = B.load_body_model("neutral")
    clips, _ = md.load_or_build("val", "amass", verbose=False)
    win = 64
    raw = []
    for c in clips[:50]:
        for s in range(0, len(c) - win + 1, win):
            raw.append(c[s:s + win])
    raw = np.stack(raw[:200]).astype(np.float32)            # [N,64,139]

    # numpy reference (recon_eval path): per window, features_to_smpl -> smpl_fk -> root-relative
    ref = []
    for w in raw:
        R, t = conv.features_to_smpl(w, np.zeros(3, np.float32))
        gp, _ = B.smpl_fk(R, t * 0, J_rest_np, parents_np)  # trans=0
        ref.append(gp - gp[:, 0:1])
    ref = np.stack(ref)                                     # [N,64,22,3]

    # torch path
    with torch.no_grad():
        tp = features_to_rootrel(torch.tensor(raw)).numpy()

    err_cm = np.linalg.norm(tp - ref, axis=-1).mean() * 100
    print(f"torch-FK vs numpy-FK root-relative position diff: {err_cm:.6f} cm  "
          f"({'PASS' if err_cm < 1e-2 else 'FAIL'})")

    # gradient sanity: position loss must backprop into the features
    f = torch.tensor(raw[:4], requires_grad=True)
    loss = (features_to_rootrel(f) ** 2).mean()
    loss.backward()
    g = f.grad[..., :132].abs().mean().item()
    print(f"grad wrt 6D-rotation features: {g:.3e}  ({'PASS' if g > 0 else 'FAIL (no grad)'})")
