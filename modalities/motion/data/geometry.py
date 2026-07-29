"""
Rotation math shared by all converters (skeleton- and spec-agnostic): quaternion <->
rotation matrix <-> 6-D representation. The 6-D form (Zhou et al. 2019) is the continuous,
neural-net-friendly rotation encoding used by every feature spec.

(Extracted from the old data/features.py so converters for different feature specs share
one copy.)
"""

import numpy as np


def quat_to_matrix(q):
    """[..., 4] (w,x,y,z) -> [..., 3, 3]."""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(R):
    """[..., 3, 3] -> [..., 4] (w,x,y,z). Stable branch-free-ish."""
    m = R.reshape(-1, 3, 3)
    w = np.sqrt(np.maximum(0.0, 1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2])) / 2
    x = np.sqrt(np.maximum(0.0, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2])) / 2
    y = np.sqrt(np.maximum(0.0, 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2])) / 2
    z = np.sqrt(np.maximum(0.0, 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2])) / 2
    x = np.copysign(x, m[:, 2, 1] - m[:, 1, 2])
    y = np.copysign(y, m[:, 0, 2] - m[:, 2, 0])
    z = np.copysign(z, m[:, 1, 0] - m[:, 0, 1])
    q = np.stack([w, x, y, z], axis=-1)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    return q.reshape(R.shape[:-2] + (4,))


def matrix_to_6d(R):
    """[..., 3, 3] -> [..., 6] = first two columns concatenated [col0(3), col1(3)].

    (Must concatenate columns, NOT row-major flatten of R[...,:,:2] — that interleaves
    them and breaks the Gram-Schmidt inverse.)
    """
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def sixd_to_matrix(d6):
    """[..., 6] -> [..., 3, 3] via Gram-Schmidt (Zhou et al.)."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)
