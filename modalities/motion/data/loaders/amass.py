"""
AMASS loader: SMPL+H .npz sequences -> 139-dim feature clips (PLAN: the diversity dataset).

AMASS official downloads extract per-dataset as <Dataset>/<Subject>/<seq>_poses.npz, each
holding: poses [T, 156] (axis-angle, SMPL+H), trans [T,3], betas, gender, mocap_framerate.
We take the 22 body joints, resample to 30 fps (AMASS is often 60/100/120), and convert
via data/conv.smpl_to_features -> the same 139-dim feature LAFAN1 produces.

Subject-diversity split: whole subjects are held out for val (the clean test of "more
subjects -> better generalization"), unlike LAFAN1 where only subject5 was held out.

⚠ Validation pending Ning's AMASS download: verify the up-axis (B.UP_AXIS), the .npz
key names, and the directory layout against the real files (this loader is written to the
documented AMASS format but untested on real data yet).
"""

import glob
import os
import sys

import numpy as np

from modalities.motion.data import paths  # noqa: E402
from modalities.motion.data.converters import smpl_body as B  # noqa: E402  (load_body_model, UP_AXIS)
from modalities.motion.data.converters import smpl_to_rot139 as conv  # noqa: E402  (smpl_to_features)

TARGET_FPS = 30.0


def list_amass_files(subsets=None):
    """All AMASS .npz under <base>/datasets/amass (excluding body_models)."""
    root = paths.AMASS_DIR
    files = []
    for f in glob.glob(os.path.join(root, "**", "*.npz"), recursive=True):
        if "body_models" in f:
            continue
        if subsets and not any(s.lower() in f.lower() for s in subsets):
            continue
        files.append(f)
    return sorted(files)


def _subject_of(path):
    """Subject id, namespaced by dataset: "Dataset/Subject" (AMASS layout Dataset/Subject/seq).

    Namespacing avoids cross-dataset id collisions (e.g. CMU/8 vs KIT/8 are different
    people) so the held-out-subject split counts and disjointness stay honest.
    """
    rel = os.path.relpath(path, paths.AMASS_DIR)
    parts = rel.split(os.sep)
    if len(parts) >= 3:                       # Dataset/Subject/seq.npz
        return f"{parts[0]}/{parts[-2]}"
    return os.path.basename(os.path.dirname(path))


def _resample(arr, src_fps, tgt_fps=TARGET_FPS):
    if src_fps <= tgt_fps + 1e-3:
        return arr
    step = int(round(src_fps / tgt_fps))
    return arr[::max(step, 1)]


def load_one(path, J, parents):
    """One .npz -> features [T,139] (resampled to 30fps), or None if too short / malformed."""
    d = np.load(path, allow_pickle=True)
    if "poses" not in d or "trans" not in d:
        return None
    fps = float(d["mocap_framerate"]) if "mocap_framerate" in d else TARGET_FPS
    poses = _resample(np.asarray(d["poses"]), fps)
    trans = _resample(np.asarray(d["trans"]), fps)
    if len(poses) < 16:
        return None
    feats, _ = conv.smpl_to_features(poses, trans, J, parents)
    return feats


def load_feature_clips(split="train", val_subject_frac=0.15, subsets=None,
                       gender="neutral", seed=0, verbose=True):
    """Parse AMASS -> list of [T,139] clips, split by SUBJECT.

    val = a held-out fraction of subjects (whole subjects, never seen in train).
    """
    J, parents = B.load_body_model(gender)
    files = list_amass_files(subsets)
    if not files:
        raise FileNotFoundError(
            f"No AMASS .npz under {paths.AMASS_DIR}. Download per-dataset SMPL+H G into "
            f"{paths.AMASS_DIR}/<Dataset>/ first.")

    subjects = sorted({_subject_of(f) for f in files})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(subjects))
    n_val = max(1, int(val_subject_frac * len(subjects)))
    val_subjects = {subjects[i] for i in perm[:n_val]}

    clips = []
    kept_subjects = set()
    for i, f in enumerate(files):
        subj = _subject_of(f)
        in_val = subj in val_subjects
        if (split == "val") != in_val:
            continue
        feats = load_one(f, J, parents)
        if feats is None:
            continue
        clips.append(feats)
        kept_subjects.add(subj)
        if verbose and i % 200 == 0:
            print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}: {feats.shape}")
    total = sum(len(c) for c in clips)
    if verbose:
        print(f"AMASS split={split}: {len(clips)} clips from {len(kept_subjects)} subjects, "
              f"{total} frames (~{total/TARGET_FPS/60:.1f} min), D=139")
    return clips, [np.zeros(3, np.float32)] * len(clips), None


if __name__ == "__main__":
    files = list_amass_files()
    print(f"AMASS .npz found: {len(files)}")
    if files:
        subjects = sorted({_subject_of(f) for f in files})
        print(f"subjects: {len(subjects)} (e.g. {subjects[:5]})")
        J, parents = B.load_body_model("neutral")
        feats = load_one(files[0], J, parents)
        print(f"sample {os.path.basename(files[0])}: {None if feats is None else feats.shape}")
        if feats is not None:
            h = feats[:, 132 + 2]
            print(f"  height(up-axis {B.UP_AXIS}) range {h.min():.2f}..{h.max():.2f} m "
                  f"(expect ~0.9 standing; if ~0 or negative, fix UP_AXIS)")
    else:
        print("(no data yet — run after AMASS downloads)")
