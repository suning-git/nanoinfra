"""
Bones-SEED loader: the soma_uniform BVH corpus -> [T,139] rot139 feature clips.

The published dataset (bones.studio, license required) ships RAW capture — G1
robot packages and SOMA-processed 78-joint BVH — not our features. This loader
runs the SOMA->SMPL retarget (converters/soma_retarget.py: position-based,
convention-free, 120->30fps) over `datasets/bones_seed/soma_uniform/bvh/`.

Split is by ACTOR (the `__A###` filename suffix): whole actors held out for val,
like AMASS's held-out-subject split. The actor partition is a fixed permutation
(rng seed 0 over the sorted actor list), so train/val membership is reproducible
from the file list alone. Clip order within a split is sorted-glob order — the
same deterministic order the original batch converter used, so clips align with
its `rot139/{split}_names.npz` sidecars bitwise.

Ported from the research batch driver this line grew out of; the conversion
itself is converters/soma_retarget.py, ported verbatim.
"""

import glob
import os
import re
from multiprocessing import Pool

import numpy as np

from modalities.motion.data import paths
from modalities.motion.data.converters import smpl_body as B
from modalities.motion.data.converters import soma_retarget as S

VAL_ACTOR_FRAC = 0.10
MIN_FRAMES_30 = 64                     # drop fragments too short to matter (< ~2.1s)
_ACTOR_RE = re.compile(r"__A(\d+)")

_J = _PAR = None


def _init():
    global _J, _PAR
    _J, _PAR = B.load_body_model("neutral")


def _actor(path):
    m = _ACTOR_RE.search(os.path.basename(path))
    return m.group(1) if m else "unk"


def _convert(path):
    try:
        out = S.soma_bvh_to_rot139(path, _J, _PAR)
    except Exception:
        return None
    if out is None:
        return None
    feats, root0 = out
    if feats.shape[0] < MIN_FRAMES_30:
        return None
    return feats, root0, path


def load_feature_clips(split="train", verbose=True, limit=0, workers=32):
    """Parse+retarget Bones-SEED BVH -> (clips, root0s, names) for one split.

    Same contract as amass.load_feature_clips. Heavy (the full corpus is ~129k
    clips); dataset.load_or_build caches the result as the rot139 format version,
    so this runs once per machine.
    """
    bvh_root = os.path.join(paths.BONES_SEED_DIR, "soma_uniform", "bvh")
    files = sorted(glob.glob(os.path.join(bvh_root, "**", "*.bvh"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"no BVH under {bvh_root} — Bones-SEED is a separate download "
            f"(https://bones.studio, license required); unpack soma_uniform there. "
            f"See exemplars/nano_motion/data/README.md.")

    # actor split over the FULL corpus (must not depend on `limit`, or a smoke run
    # would silently use a different membership than the real build)
    actors = sorted({_actor(f) for f in files})
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(actors))
    n_val = max(1, int(VAL_ACTOR_FRAC * len(actors)))
    val_actors = {actors[i] for i in perm[:n_val]}
    want_val = split == "val"
    mine = [f for f in files if (_actor(f) in val_actors) == want_val]
    if limit:
        mine = mine[:limit]
    if verbose:
        print(f"[bones_seed/{split}] {len(mine)} of {len(files)} BVH files "
              f"({len(actors)} actors, {len(val_actors)} held out for val)")

    clips, roots, names = [], [], []
    with Pool(workers, initializer=_init) as pool:
        # ordered imap: clip order == sorted-glob order (deterministic)
        for i, res in enumerate(pool.imap(_convert, mine, chunksize=32)):
            if res is not None:
                feats, root0, path = res
                clips.append(feats)
                roots.append(root0)
                names.append(os.path.basename(path)[:-4])
            if verbose and (i + 1) % 5000 == 0:
                print(f"  [{i + 1}/{len(mine)}] kept {len(clips)}")
    if verbose:
        frames = sum(len(c) for c in clips)
        print(f"[bones_seed/{split}] {len(clips)} clips, {frames / 1e6:.2f}M frames "
              f"(~{frames / 30 / 3600:.1f} h at 30fps)")
    return clips, roots, names
