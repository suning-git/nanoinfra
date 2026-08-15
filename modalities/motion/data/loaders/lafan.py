"""
Loader for the raw LAFAN1 dataset (BVH). Imports the downloaded LAFAN1 parser package and
turns BVH files into rot139 feature clips (via the bvh_to_rot139 converter), with the
LAFAN1 held-out split (subject5 = val).

(Was the dataset-assembly half of the old data/features.py.)
"""

import glob
import os
import sys
import warnings

from modalities.motion.data import paths  # noqa: E402
from modalities.motion.data.converters import bvh_to_rot139 as conv  # noqa: E402

LAFAN_DIR = paths.LAFAN1_DIR
FPS = 30.0


def load_lafan():
    """Import the downloaded LAFAN1 parser package (extract, utils)."""
    if LAFAN_DIR not in sys.path:
        sys.path.insert(0, LAFAN_DIR)
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    from lafan_pkg import extract, utils  # noqa
    return extract, utils


def list_bvh(split="all"):
    files = sorted(glob.glob(os.path.join(LAFAN_DIR, "bvh", "*.bvh")))
    if split == "all":
        return files
    # held-out: subject5 for val (LAFAN1 convention uses subject5 as test)
    if split == "train":
        return [f for f in files if "subject5" not in os.path.basename(f)]
    if split == "val":
        return [f for f in files if "subject5" in os.path.basename(f)]
    raise ValueError(split)


def load_feature_clips(split="all", limit=None, verbose=True):
    """Parse BVH files -> list of [T,139] feature clips (+ parallel root0 list, ref_anim)."""
    extract, utils = load_lafan()
    files = list_bvh(split)
    if limit:
        files = files[:limit]
    clips, root0s = [], []
    ref_anim = None
    for i, f in enumerate(files):
        anim = extract.read_bvh(f)
        if ref_anim is None:
            ref_anim = anim
        feats, r0 = conv.extract_features(anim, utils)
        clips.append(feats)
        root0s.append(r0)
        if verbose and (i % 10 == 0 or i == len(files) - 1):
            print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}: {feats.shape}")
    total = sum(len(c) for c in clips)
    if verbose:
        print(f"split={split}: {len(clips)} clips, {total} frames "
              f"(~{total/FPS/60:.1f} min), D={clips[0].shape[1]}")
    return clips, root0s, ref_anim


if __name__ == "__main__":
    import numpy as np
    extract, utils = load_lafan()
    f = list_bvh("all")[0]
    anim = extract.read_bvh(f)
    feats, r0 = conv.extract_features(anim, utils)
    print(f"{os.path.basename(f)}: feats {feats.shape}, expected D={conv.FEATURE_DIM}")
    _, gp_orig = utils.quat_fk(anim.quats.astype(np.float64), anim.pos.astype(np.float64), anim.parents)
    q_rec, p_rec = conv.features_to_anim(feats, r0, anim, utils)
    gp_rec = conv.global_positions(q_rec, p_rec, anim.parents, utils)
    mpjpe = np.linalg.norm(gp_rec - gp_orig, axis=-1).mean()
    print(f"round-trip MPJPE = {mpjpe:.3f} cm (should be ~0)")
