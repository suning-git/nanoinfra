"""prepare.py — raw capture files to rot139 feature clips.

    download.py -> [prepare.py] -> ../train_codec.py -> encode.py -> ../train_t2m.py

BVH skeletons and SMPL parameter files describe the same thing in incompatible ways.
rot139 is the common representation everything downstream speaks: per frame, the
joint rotations, the root's displacement and height, and four foot-contact flags —
139 numbers. Converting once, here, is what lets one tokenizer train on either
source and one model consume both.

Why root DISPLACEMENT rather than root position: a model that predicts absolute
position has to memorise where in the capture volume each clip happened. The
displacement is the same walk wherever it was recorded, so `root0s` (each clip's
starting root) is kept alongside the features and only used when rendering back to
world space.

Output is one file per split, `datasets/<source>/rot139/<split>.npz`, holding
`clips` (a ragged array of [T,139]) and `root0s`. It is tokenizer-INDEPENDENT — it
does not change when you retrain a codec — which is why it lives beside the dataset
rather than in a cache directory.

    python -m exemplars.nano_motion.data.prepare                    # lafan1
    python -m exemplars.nano_motion.data.prepare --source amass

This is the slow step: minutes for LAFAN1, an hour or more for AMASS. It is done
once, and skipped if the output already exists.
"""

import argparse
import time

import numpy as np

from exemplars.nano_motion import spec
from modalities.motion.data import dataset as md
from modalities.motion.data import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source", default=spec.SOURCE,
                    choices=["lafan1", "amass", "bones_seed"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of raw clips (a quick check, not a real build)")
    args = ap.parse_args()

    for split in args.splits:
        out = paths.processed_file(args.source, split, "rot139")
        print(f"\n=== {args.source}/{split} -> {out}")
        t0 = time.time()
        kw = {"limit": args.limit} if args.limit else {}
        clips, _ = md.load_or_build(split, source=args.source, **kw)
        frames = sum(len(c) for c in clips)
        print(f"  {len(clips)} clips, {frames} frames "
              f"({frames / spec.FPS / 3600:.1f} hours at {spec.FPS:g}fps), "
              f"D={clips[0].shape[1]}, {time.time() - t0:.0f}s")

        # A silent NaN here becomes a silent NaN in the codec's loss much later, and
        # by then the cause is three stages away. One pass over the features is cheap.
        bad = sum(int(not np.isfinite(c).all()) for c in clips)
        assert not bad, f"{bad} clips contain non-finite values — check the raw files"
        assert clips[0].shape[1] == spec.D_FEAT, \
            f"expected {spec.D_FEAT} features per frame, got {clips[0].shape[1]}"

    print("\nnext: python -m exemplars.nano_motion.train_codec")


if __name__ == "__main__":
    main()
