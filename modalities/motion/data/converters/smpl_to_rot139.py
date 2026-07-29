"""
Converter: SMPL(+H) motion parameters <-> the rot139 feature spec.

    rot139 = [ 22 joints x 6D local rotation (132) | root horiz disp (2) | root height (1)
               | foot contacts (4) ]   => D = 139   (identical layout to the BVH rot139)

Local joint rotations come straight from the SMPL pose params (axis-angle -> 6D). Global
positions (for contacts / root) come from the shared SMPL FK on the neutral skeleton.
Forward (smpl_to_features) and inverse (features_to_smpl) both live here — both are specific
to (SMPL, rot139). Adding a new spec = a new smpl_to_<spec>.py, no edits here.

(Was the feature half of the old data/smpl.py.)
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
import geometry as G    # noqa: E402
import smpl_body as B   # noqa: E402  (load_body_model, FK, axis_angle_to_matrix, UP_AXIS, …)

N_BODY_JOINTS = B.N_BODY_JOINTS


def smpl_to_features(poses, trans, J_rest, parents, fps=None):
    """SMPL poses [T,>=66] axis-angle + trans [T,3] -> (features [T,139], root0 [3])."""
    poses = np.asarray(poses, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    aa = poses[:, :N_BODY_JOINTS * 3].reshape(-1, N_BODY_JOINTS, 3)
    local_R = B.axis_angle_to_matrix(aa)
    rot6d = G.matrix_to_6d(local_R).reshape(poses.shape[0], -1)        # [T,132]

    root = trans
    disp = np.zeros((poses.shape[0], 2))
    disp[1:] = root[1:][:, B.HORIZ_AXES] - root[:-1][:, B.HORIZ_AXES]  # horizontal disp
    height = root[:, B.UP_AXIS:B.UP_AXIS + 1]                          # up-axis height

    gp, _ = B.smpl_fk(local_R, trans, J_rest, parents)
    contacts = B.foot_contacts(gp)

    feats = np.concatenate([rot6d, disp, height, contacts], axis=-1).astype(np.float32)
    return feats, root[0].astype(np.float32)


def features_to_smpl(feats, root0):
    """Inverse: features [T,139] -> (local_R [T,22,3,3], trans [T,3]) for FK-based eval."""
    T = feats.shape[0]
    rot6d = feats[:, :N_BODY_JOINTS * 6].reshape(T, N_BODY_JOINTS, 6)
    disp = feats[:, N_BODY_JOINTS * 6:N_BODY_JOINTS * 6 + 2]
    height = feats[:, N_BODY_JOINTS * 6 + 2]
    local_R = G.sixd_to_matrix(rot6d.astype(np.float64))
    trans = np.zeros((T, 3))
    trans[0, B.HORIZ_AXES] = root0[B.HORIZ_AXES]
    for t in range(1, T):
        trans[t, B.HORIZ_AXES] = trans[t - 1, B.HORIZ_AXES] + disp[t]
    trans[:, B.UP_AXIS] = height
    return local_R, trans


if __name__ == "__main__":
    J, parents = B.load_body_model("neutral")
    print(f"SMPL+H: {N_BODY_JOINTS} body joints")
    rng = np.random.default_rng(0)
    T = 5
    poses = 0.3 * rng.standard_normal((T, N_BODY_JOINTS * 3))
    trans = np.cumsum(0.02 * rng.standard_normal((T, 3)), axis=0)
    feats, root0 = smpl_to_features(poses, trans, J, parents)
    R_in = B.axis_angle_to_matrix(poses.reshape(T, N_BODY_JOINTS, 3))
    gp_in, _ = B.smpl_fk(R_in, trans, J, parents)
    R_rec, trans_rec = features_to_smpl(feats, root0)
    gp_rec, _ = B.smpl_fk(R_rec, trans_rec, J, parents)
    mpjpe = np.linalg.norm(gp_rec - gp_in, axis=-1).mean() * 100
    print(f"[{'PASS' if mpjpe < 0.1 else 'FAIL'}] rot139 round-trip MPJPE = {mpjpe:.4f} cm "
          f"(up-axis={B.UP_AXIS}, dim={feats.shape[1]})")
