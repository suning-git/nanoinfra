"""
Shared SMPL kinematics — used by every SMPL-based feature spec (rot139 now, hml263 later).

Provides: the body model load (rest joints + kinematic tree), axis-angle -> matrix
(Rodrigues), rigid-bone forward kinematics, foot-contact detection, and the up-axis
convention. The body model is rendered on the NEUTRAL skeleton; per-subject shape (betas)
is intentionally discarded — the representation is body-agnostic.

⚠ AMASS is Z-UP (verified empirically: head z≈1.54, foot z≈0.08, standing pelvis z≈0.92 m).
Up-axis = 2, horizontal plane = the two non-up axes. (Validated, not assumed.)

(Was the kinematics half of the old data/smpl.py.)
"""

import os
import sys

import numpy as np

from modalities.motion.data import paths  # noqa: E402

N_BODY_JOINTS = 22
SMPL_LFOOT = [7, 10]   # L_ankle, L_foot
SMPL_RFOOT = [8, 11]   # R_ankle, R_foot
UP_AXIS = 2            # AMASS is Z-up
HORIZ_AXES = [i for i in range(3) if i != UP_AXIS]   # Z-up -> [0,1] = x,y


def load_body_model(gender: str = "neutral"):
    """Return (J_rest [22,3], parents [22]) from the SMPL+H body model (neutral by default)."""
    path = os.path.join(paths.AMASS_BODY_MODELS, gender, "model.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SMPL+H body model not found at {path}. Place smplh models under "
            f"{paths.AMASS_BODY_MODELS}/<gender>/model.npz (see README).")
    d = np.load(path, allow_pickle=True)
    J = np.asarray(d["J"], dtype=np.float64)[:N_BODY_JOINTS]      # [22,3]
    parents = np.asarray(d["kintree_table"], dtype=np.int64)[0, :N_BODY_JOINTS].copy()
    parents[0] = -1
    return J, parents


def axis_angle_to_matrix(aa):
    """[..., 3] axis-angle -> [..., 3, 3] (Rodrigues, vectorized)."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / (theta + 1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -kz, ky
    K[..., 1, 0], K[..., 1, 2] = kz, -kx
    K[..., 2, 0], K[..., 2, 1] = -ky, kx
    s = np.sin(theta)[..., 0][..., None, None]
    c = np.cos(theta)[..., 0][..., None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def smpl_fk(local_R, trans, J_rest, parents):
    """local_R [T,J,3,3] + trans [T,3] -> global pos [T,J,3], global R [T,J,3,3] (rigid bone)."""
    T, J = local_R.shape[0], local_R.shape[1]
    gR = np.zeros((T, J, 3, 3))
    gp = np.zeros((T, J, 3))
    gR[:, 0] = local_R[:, 0]
    gp[:, 0] = trans + J_rest[0]
    for j in range(1, J):
        p = parents[j]
        gR[:, j] = gR[:, p] @ local_R[:, j]
        offset = (J_rest[j] - J_rest[p])
        gp[:, j] = gp[:, p] + np.einsum("tij,j->ti", gR[:, p], offset)
    return gp, gR


def foot_contacts(gp, vel_thresh=0.01):
    """[T,J,3] global positions -> [T,4] binary-ish contact flags for SMPL feet."""
    idx = SMPL_LFOOT + SMPL_RFOOT
    pos = gp[:, idx]
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    speed = np.linalg.norm(vel, axis=-1)
    return (speed < vel_thresh).astype(np.float64)
