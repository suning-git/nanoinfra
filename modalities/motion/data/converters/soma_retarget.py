"""
SOMA (Bones-SEED, 78-joint BVH) -> SMPL rot139 retarget.

Ported verbatim from the research implementation this line grew out of: only the
import header changed, so the features it produces are bit-identical to the ones
the shipped codecs were trained on.

Pipeline (position-based retarget, convention-free):
  1. Parse the SOMA BVH (hierarchy + MOTION) — ZYX-Euler channels, cm, Y-up, 120 fps.
  2. FK the SOMA skeleton -> per-frame GLOBAL positions of the 22 mapped joints.
  3. Convert cm->m and Y-up->Z-up (our UP_AXIS=2 convention).
  4. Solve SMPL LOCAL rotations from those global positions: per joint, align the SMPL
     REST bone directions to the observed target directions (Kabsch for the 2 multi-child
     anchors pelvis/spine3, minimal-twist swing for single-child limbs, identity for leaves).
     This is convention-free — it never reads SOMA's (odd, +X-torso) rest pose.
  5. FK those local rotations on the SMPL neutral skeleton -> rot139 features via the SAME
     assembly as smpl_to_rot139 (rot6d | root horiz disp | root height | foot contacts).

Downsamples 120 -> 30 fps (step 4). Retargets onto SMPL proportions (scale-free rotations),
so the output plugs straight into the existing tokenizers / ARs / energy eval — zero new
tokenizer/AR code. Root height is scaled by the SMPL/SOMA leg-length ratio so feet stay ~floor.
"""
import os
import re
import sys

import numpy as np

from modalities.motion.data import geometry as G
from modalities.motion.data.converters import smpl_body as B

TARGET_FPS = 30.0
SRC_FPS = 120.0

# SMPL body-22 kinematic tree (parents) and the child sets used by the retarget.
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPL_NAMES = ["pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2", "L_ankle",
              "R_ankle", "spine3", "L_foot", "R_foot", "neck", "L_collar", "R_collar", "head",
              "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist"]

# SMPL joint -> SOMA joint name.  pelvis is anchored at SOMA *Hips* (Hips carries the 101cm
# pelvis height; Root sits at the floor), so its child directions all emanate from Hips.
SMPL_TO_SOMA = {
    "pelvis": "Hips", "L_hip": "LeftLeg", "R_hip": "RightLeg", "spine1": "Spine1",
    "L_knee": "LeftShin", "R_knee": "RightShin", "spine2": "Spine2", "L_ankle": "LeftFoot",
    "R_ankle": "RightFoot", "spine3": "Chest", "L_foot": "LeftToeBase", "R_foot": "RightToeBase",
    "neck": "Neck1", "L_collar": "LeftShoulder", "R_collar": "RightShoulder", "head": "Head",
    "L_shoulder": "LeftArm", "R_shoulder": "RightArm", "L_elbow": "LeftForeArm",
    "R_elbow": "RightForeArm", "L_wrist": "LeftHand", "R_wrist": "RightHand",
}

_CHILDREN = {j: [c for c in range(22) if SMPL_PARENTS[c] == j] for j in range(22)}


# ---------------------------------------------------------------------------- BVH parse + FK

class Bvh:
    __slots__ = ("names", "parents", "offsets", "chan_names", "chan_slices", "n_chan",
                 "frames", "fps")


def parse_bvh(path):
    """Parse a BVH file -> Bvh (hierarchy + [T, total_channels] motion frames)."""
    with open(path) as fh:
        text = fh.read()
    head, _, motion = text.partition("MOTION")

    names, parents, offsets, chan_names = [], [], [], []
    stack = [-1]
    tokens = head.replace("{", " { ").replace("}", " } ").split("\n")
    for line in tokens:
        s = line.strip()
        if s.startswith("ROOT") or s.startswith("JOINT"):
            names.append(s.split()[1])
            parents.append(stack[-1])
            offsets.append(None)
            chan_names.append([])
            stack.append(len(names) - 1)
        elif s.startswith("End"):
            stack.append(-2)                       # End Site: consume its OFFSET/braces, no joint
        elif s.startswith("OFFSET"):
            if stack[-1] >= 0 and offsets[stack[-1]] is None:
                offsets[stack[-1]] = [float(x) for x in s.split()[1:4]]
        elif s.startswith("CHANNELS"):
            if stack[-1] >= 0:
                chan_names[stack[-1]] = s.split()[2:]
        elif s == "}":
            stack.pop()

    n = len(names)
    b = Bvh()
    b.names = names
    b.parents = np.array(parents, dtype=np.int64)
    b.offsets = np.array(offsets, dtype=np.float64)          # [J,3]
    b.chan_names = chan_names
    slices, k = [], 0
    for ch in chan_names:
        slices.append((k, k + len(ch)))
        k += len(ch)
    b.chan_slices = slices
    b.n_chan = k

    mlines = [ln for ln in motion.split("\n")]
    fps = SRC_FPS
    data_rows = []
    for ln in mlines:
        s = ln.strip()
        if s.startswith("Frame Time"):
            fps = 1.0 / float(s.split(":")[1])
        elif s and not s.startswith("Frames"):
            row = s.split()
            if len(row) >= k:
                data_rows.append([float(x) for x in row[:k]])
    b.frames = np.array(data_rows, dtype=np.float64)         # [T, n_chan]
    b.fps = fps
    return b


def _euler_channel_matrix(chan_names, chan_vals):
    """Rotation matrix from a joint's rotation channels, applied in listed order (deg).

    BVH convention: R = R_first @ R_second @ R_third for the channel order as written
    (SOMA is 'Zrotation Yrotation Xrotation' -> R = Rz @ Ry @ Rx).
    Returns (R [T,3,3], root_translation_channels [T,3] or None).
    """
    T = chan_vals.shape[0]
    R = np.tile(np.eye(3), (T, 1, 1))
    pos = None
    axis_idx = {"X": 0, "Y": 1, "Z": 2}
    for i, name in enumerate(chan_names):
        col = chan_vals[:, i]
        if name.endswith("position"):
            if pos is None:
                pos = np.zeros((T, 3))
            pos[:, axis_idx[name[0]]] = col
        else:  # rotation channel, degrees
            a = np.deg2rad(col)
            c, s = np.cos(a), np.sin(a)
            Ri = np.tile(np.eye(3), (T, 1, 1))
            ax = name[0]
            if ax == "X":
                Ri[:, 1, 1], Ri[:, 1, 2], Ri[:, 2, 1], Ri[:, 2, 2] = c, -s, s, c
            elif ax == "Y":
                Ri[:, 0, 0], Ri[:, 0, 2], Ri[:, 2, 0], Ri[:, 2, 2] = c, s, -s, c
            else:  # Z
                Ri[:, 0, 0], Ri[:, 0, 1], Ri[:, 1, 0], Ri[:, 1, 1] = c, -s, s, c
            R = R @ Ri
    return R, pos


def bvh_global_positions(b):
    """FK the SOMA BVH -> global joint positions [T, J, 3] in BVH units (cm, Y-up)."""
    T = b.frames.shape[0]
    J = len(b.names)
    Rloc = np.zeros((T, J, 3, 3))
    tloc = np.zeros((T, J, 3))                    # local translation = offset (+ pos channels)
    for j in range(J):
        s0, s1 = b.chan_slices[j]
        R, pos = _euler_channel_matrix(b.chan_names[j], b.frames[:, s0:s1])
        Rloc[:, j] = R
        tloc[:, j] = b.offsets[j]
        if pos is not None:
            tloc[:, j] += pos
    gp = np.zeros((T, J, 3))
    gR = np.zeros((T, J, 3, 3))
    for j in range(J):
        p = b.parents[j]
        if p < 0:
            gR[:, j] = Rloc[:, j]
            gp[:, j] = tloc[:, j]
        else:
            gR[:, j] = gR[:, p] @ Rloc[:, j]
            gp[:, j] = gp[:, p] + np.einsum("tij,tj->ti", gR[:, p], tloc[:, j])
    return gp


# ------------------------------------------------------------------ positions -> SMPL local R

def _swing(a_unit, b_unit):
    """Minimal-twist rotation [T,3,3] mapping fixed unit a -> unit b[T,3] (Rodrigues)."""
    T = b_unit.shape[0]
    a = a_unit / (np.linalg.norm(a_unit) + 1e-12)
    b = b_unit / (np.linalg.norm(b_unit, axis=-1, keepdims=True) + 1e-12)
    v = np.cross(np.broadcast_to(a, b.shape), b)             # [T,3]
    c = b @ a                                                # [T] = cos angle
    s2 = (v * v).sum(-1)                                     # sin^2
    K = np.zeros((T, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    coef = ((1.0 - c) / (s2 + 1e-12))[:, None, None]
    R = np.tile(np.eye(3), (T, 1, 1)) + K + coef * (K @ K)
    # near-antiparallel (c ~ -1): rotate 180 about any axis perpendicular to a
    bad = c < -0.9999
    if bad.any():
        perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
        Rp = 2.0 * np.outer(perp, perp) - np.eye(3)
        R[bad] = Rp
    return R


def _kabsch(rest_dirs, tgt_dirs):
    """Best-fit rotation [T,3,3] aligning fixed rest_dirs [n,3] -> tgt_dirs [T,n,3]."""
    T = tgt_dirs.shape[0]
    rest = rest_dirs / (np.linalg.norm(rest_dirs, axis=-1, keepdims=True) + 1e-12)   # [n,3]
    tgt = tgt_dirs / (np.linalg.norm(tgt_dirs, axis=-1, keepdims=True) + 1e-12)      # [T,n,3]
    H = np.einsum("ni,tnj->tij", rest, tgt)                  # [T,3,3]
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    D = np.tile(np.eye(3), (T, 1, 1))
    D[:, 2, 2] = d
    R = np.einsum("tij,tjk,tkl->til", Vt.transpose(0, 2, 1), D, U.transpose(0, 2, 1))
    return R


def positions_to_smpl_local(gp_smpl, J_rest):
    """gp_smpl [T,22,3] target global positions -> SMPL local rotations [T,22,3,3].

    J_rest [22,3] = SMPL neutral rest joints. Per joint we align the rest child-offset
    direction(s) to the observed child direction(s): Kabsch when >1 child (pelvis, spine3),
    minimal-twist swing when exactly 1 child, inherit-parent for leaves.
    """
    T = gp_smpl.shape[0]
    rest_off = J_rest - J_rest[SMPL_PARENTS]                 # [22,3] rest bone offsets
    gR = np.zeros((T, 22, 3, 3))
    for j in range(22):                                       # parents precede children (SMPL order)
        ch = _CHILDREN[j]
        if len(ch) == 0:
            gR[:, j] = gR[:, SMPL_PARENTS[j]]                 # leaf: inherit parent orientation
        elif len(ch) == 1:
            c = ch[0]
            tgt = gp_smpl[:, c] - gp_smpl[:, j]
            gR[:, j] = _swing(rest_off[c], tgt)
        else:
            rest_dirs = rest_off[ch]                          # [nc,3]
            tgt = gp_smpl[:, ch] - gp_smpl[:, j:j + 1]        # [T,nc,3]
            gR[:, j] = _kabsch(rest_dirs, tgt)
    local = np.zeros((T, 22, 3, 3))
    local[:, 0] = gR[:, 0]
    for j in range(1, 22):
        p = SMPL_PARENTS[j]
        local[:, j] = np.einsum("tji,tjk->tik", gR[:, p], gR[:, j])   # R_p^T @ R_j
    return local


# ---------------------------------------------------------------------------- full converter

_SCALE = None   # SMPL/SOMA leg-length ratio (rest), computed lazily from the first BVH


def _yup_cm_to_zup_m(p):
    """(X,Y,Z) Y-up cm -> (x,y,z) Z-up m : +90deg about X, then /100."""
    out = np.empty_like(p)
    out[..., 0] = p[..., 0]
    out[..., 1] = -p[..., 2]
    out[..., 2] = p[..., 1]
    return out / 100.0


def _resample(a, src_fps):
    step = max(1, int(round(src_fps / TARGET_FPS)))
    return a[::step]


def soma_bvh_to_rot139(path, J_rest, parents):
    """One SOMA BVH -> (features [T,139], root0 [3]); None if too short/malformed."""
    b = parse_bvh(path)
    if b.frames.shape[0] < 16 * int(round(b.fps / TARGET_FPS)):
        return None
    gp_all = bvh_global_positions(b)                          # [T, 78, 3] cm, Y-up

    name_idx = {n: i for i, n in enumerate(b.names)}
    soma_cols = [name_idx[SMPL_TO_SOMA[SMPL_NAMES[j]]] for j in range(22)]
    gp22 = gp_all[:, soma_cols, :]                            # [T,22,3] cm Y-up
    gp22 = _yup_cm_to_zup_m(gp22)                             # [T,22,3] m Z-up

    # leg-length scale (SMPL rest / SOMA rest) so retargeted feet stay ~floor
    soma_leg = (np.linalg.norm(b.offsets[name_idx["LeftShin"]])
                + np.linalg.norm(b.offsets[name_idx["LeftFoot"]])) / 100.0
    smpl_leg = (np.linalg.norm(J_rest[4] - J_rest[1]) + np.linalg.norm(J_rest[7] - J_rest[4]))
    scale = smpl_leg / max(soma_leg, 1e-6)

    local_R = positions_to_smpl_local(gp22, J_rest)          # [T,22,3,3]

    # root translation: SOMA pelvis (Hips) position, scaled to SMPL size
    trans = gp22[:, 0, :] * scale                            # [T,3] Z-up m
    # put mean-min foot on the floor (z=0) using retargeted skeleton
    gp_fk, _ = B.smpl_fk(local_R, trans, J_rest, parents)
    foot_z = gp_fk[:, [10, 11], 2].min(axis=1)
    trans[:, 2] -= np.median(foot_z)

    local_R = _resample(local_R, b.fps)
    trans = _resample(trans, b.fps)

    rot6d = G.matrix_to_6d(local_R).reshape(local_R.shape[0], -1)      # [T,132]
    disp = np.zeros((local_R.shape[0], 2))
    disp[1:] = trans[1:, B.HORIZ_AXES] - trans[:-1, B.HORIZ_AXES]
    height = trans[:, B.UP_AXIS:B.UP_AXIS + 1]
    gp_fk, _ = B.smpl_fk(local_R, trans, J_rest, parents)
    contacts = B.foot_contacts(gp_fk)
    feats = np.concatenate([rot6d, disp, height, contacts], axis=-1).astype(np.float32)
    return feats, trans[0].astype(np.float32)


if __name__ == "__main__":
    J, parents = B.load_body_model("neutral")
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _mr.mr.__dict__.get("DATASETS", "datasets"),
        "..", "datasets", "bones_seed", "soma_uniform", "bvh", "230315",
        "screaming_001__A274.bvh")
    out = soma_bvh_to_rot139(path, J, parents)
    if out is None:
        print("too short / malformed"); sys.exit(1)
    feats, root0 = out
    h = feats[:, 132 + 2]
    print(f"feats {feats.shape}  height(z) {h.min():.2f}..{h.max():.2f} m "
          f"(standing ~0.9 expected)  root0={root0}")
    # sanity: FK back, check bone lengths + travel
    local_R, trans = __import__("smpl_to_rot139").features_to_smpl(feats, root0)
    gp, _ = B.smpl_fk(local_R, trans, J, parents)
    head_z = gp[:, 15, 2]
    print(f"head z {head_z.min():.2f}..{head_z.max():.2f} m ; "
          f"foot z min {gp[:, [10, 11], 2].min():.3f} ; "
          f"horiz travel {np.linalg.norm(gp[-1, 0, :2] - gp[0, 0, :2]):.2f} m")
