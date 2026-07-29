"""
Converter: LAFAN1 BVH animation <-> the rot139 feature spec.

    rot139 = [ 22 joints x 6D local rotation (132) | root dx,dz world (2) | root height (1)
               | foot contacts L/R foot+toe (4) ]   => D = 139

Forward (extract_features) and inverse (features_to_anim) both live here, since both are
specific to (BVH skeleton, rot139). Shared rotation math comes from geometry; the LAFAN1
parser `utils` (quat_fk, extract_feet_contacts) is passed in by the loader.

(Was the feature half of the old data/features.py.)
"""

import os
import sys

import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_PKG, "paths.py")) and _PKG != os.path.dirname(_PKG):
    _PKG = os.path.dirname(_PKG)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import paths  # noqa: E402  (registers package sub-dirs on sys.path)
import geometry as G  # noqa: E402

# LAFAN1 skeleton (22 joints): foot joints for contact detection
LFOOT_IDX = [3, 4]   # LeftFoot, LeftToe
RFOOT_IDX = [7, 8]   # RightFoot, RightToe
N_JOINTS = 22
FEATURE_DIM = N_JOINTS * 6 + 2 + 1 + 4   # 139


def extract_features(anim, utils):
    """Anim -> (features [T, 139], root0 [3]). root0 = initial root world xz,y for replay."""
    quats = anim.quats.astype(np.float64)        # [T, J, 4] local
    lpos = anim.pos.astype(np.float64)           # [T, J, 3] local (root has world translation)
    parents = anim.parents

    R = G.quat_to_matrix(quats)                  # [T, J, 3, 3]
    rot6d = G.matrix_to_6d(R).reshape(quats.shape[0], -1)   # [T, J*6]

    root = lpos[:, 0, :]                         # [T, 3] world root pos
    disp = np.zeros_like(root[:, [0, 2]])        # [T, 2]
    disp[1:] = root[1:, [0, 2]] - root[:-1, [0, 2]]
    height = root[:, 1:2]                        # [T, 1]

    _, gp = utils.quat_fk(quats, lpos, parents)  # global pos [T, J, 3]
    cl, cr = utils.extract_feet_contacts(gp, LFOOT_IDX, RFOOT_IDX, velfactor=0.02 * 100)
    contacts = np.concatenate([cl, cr], axis=-1).astype(np.float64)   # [T, 4]

    feats = np.concatenate([rot6d, disp, height, contacts], axis=-1).astype(np.float32)
    return feats, root[0].astype(np.float32)


def features_to_anim(feats, root0, ref_anim, utils):
    """features [T,139] -> (local quats [T,J,4], local pos [T,J,3]) reconstructing the Anim.

    Uses ref_anim for the static skeleton (offsets/parents). Root xz integrated from
    displacements starting at root0; height read directly.
    """
    T = feats.shape[0]
    J = N_JOINTS
    rot6d = feats[:, :J * 6].reshape(T, J, 6)
    disp = feats[:, J * 6:J * 6 + 2]
    height = feats[:, J * 6 + 2]

    R = G.sixd_to_matrix(rot6d.astype(np.float64))
    quats = G.matrix_to_quat(R)                  # [T, J, 4]

    xz = np.zeros((T, 2))
    xz[0] = root0[[0, 2]]
    for t in range(1, T):
        xz[t] = xz[t - 1] + disp[t]
    lpos = np.tile(ref_anim.pos[0:1].copy(), (T, 1, 1)).astype(np.float64)  # static offsets
    lpos[:, 0, 0] = xz[:, 0]
    lpos[:, 0, 1] = height
    lpos[:, 0, 2] = xz[:, 1]
    return quats, lpos


def global_positions(quats, lpos, parents, utils):
    _, gp = utils.quat_fk(quats, lpos, parents)
    return gp
