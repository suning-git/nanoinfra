"""
Dataset layer: load a processed feature dataset (building it from raw on first use),
fit a per-dim normalizer, and slice clips into fixed windows for the VQ-VAE.

A dataset FORMAT VERSION is `datasets/<source>/<spec>/<split>.npz`
(`paths.processed_file`). If it is missing, it is built on the fly from the raw
loaders (lafan / amass) — but the large ones take long enough that they are normally
built ahead of time, by `exemplars/nano_motion/data/prepare.py`.
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


def _extract(source: str, split: str, verbose: bool, **kw):
    """Build feature clips from a RAW source (all specs share the 139-dim layout for now)."""
    if source == "lafan1":
        import lafan  # noqa
        clips, root0s, _ = lafan.load_feature_clips(split=split, verbose=verbose)
        return clips, root0s
    if source.startswith("amass"):   # amass, or tagged variants (amass_lowdiv, …)
        import amass  # noqa
        clips, root0s, _ = amass.load_feature_clips(split=split, verbose=verbose, **kw)
        return clips, root0s
    raise ValueError(f"unknown source '{source}' (lafan1 | amass[*])")


def load_or_build(split: str, source: str = "lafan1", spec: str = paths.DEFAULT_SPEC,
                  verbose: bool = True, **kw):
    """Return (clips, root0s) for a source+split, building the processed .npz if missing."""
    f = paths.processed_file(source, split, spec)
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        clips, root0s = list(d["clips"]), list(d["root0s"])
        if verbose:
            tot = sum(len(c) for c in clips)
            print(f"loaded {f}: {len(clips)} clips, {tot} frames, D={clips[0].shape[1]}")
        return clips, root0s
    clips, root0s = _extract(source, split, verbose, **kw)
    os.makedirs(os.path.dirname(f), exist_ok=True)   # format versions live at datasets/<source>/<spec>/
    np.savez(f, clips=np.array(clips, dtype=object), root0s=np.array(root0s, dtype=object))
    if verbose:
        print(f"built -> {f}")
    return clips, root0s


class Normalizer:
    """Per-dim mean/std (rotations are O(1); displacement/height have real scale)."""

    def __init__(self, mean, std):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-4)

    @classmethod
    def fit(cls, clips):
        x = np.concatenate(clips, axis=0)
        return cls(x.mean(0), x.std(0))

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, sd):
        return cls(np.asarray(sd["mean"]), np.asarray(sd["std"]))


def make_windows(clips, win: int, stride: int, norm: Normalizer):
    """Slice normalized clips into [win, D] windows (win must be a multiple of 4)."""
    out = []
    for c in clips:
        if len(c) < win:
            continue
        cn = norm(c)
        for s in range(0, len(c) - win + 1, stride):
            out.append(cn[s:s + win])
    return np.stack(out)


if __name__ == "__main__":
    clips, _ = load_or_build("train", "amass")
    norm = Normalizer.fit(clips)
    W = make_windows(clips, win=64, stride=32, norm=norm)
    print(f"train windows: {W.shape}  (normalized mean={W.mean():.3f} std={W.std():.3f})")
