"""train_codec.py — train the motion tokenizer: rot139 features -> 512 discrete codes.

    data/prepare.py -> [train_codec.py] -> data/encode.py -> train_t2m.py

This is the step the video exemplar next door deliberately skips. There, the codec is
borrowed and frozen, because training a video tokenizer is a project of its own. Here
it is cheap enough to do properly, and doing it changes what you can see: the AR model
downstream can only ever be as good as what the codec can reconstruct, so this run
sets the ceiling on everything after it.

A conv autoencoder over 64-frame windows, with a quantizer in the middle and a
KINEMATIC loss on top of feature reconstruction. The kinematic term runs the decoded
rotations through forward kinematics and penalises joint POSITION, velocity and foot
contact — because feature error and joint error are not the same thing. A small
rotation error at the hip moves the foot a long way, and a codec tuned on feature MSE
alone will happily trade a visibly sliding foot for a lower number. That term was the
single biggest lever on reconstruction quality in the work this comes from.

The recipe is the tokenizer's own `recipe.yaml`; anything here overrides it, which is
what makes a smoke run a flag rather than a fork:

    python -m exemplars.nano_motion.train_codec --steps 200      # ~2 min, is it wired
    python -m exemplars.nano_motion.train_codec                  # the real thing

Reported at the end is the FAIR metric: MPJPE in a global canonical frame. An earlier
round of this work crowned a different representation using a heading-blind metric,
which flattered anything that discarded heading. Reconstruction numbers from a
different metric are not comparable to these.

READING THE SMOKE RUN. At a few hundred steps the global number is meaningless and
looks alarming — tens of metres — because it INTEGRATES root displacement across the
window, so a model that has not yet learned to stand still walks off into space. The
number to watch early is `root-rel`, which is scored per frame against the root and
therefore says something at any step count. The global metric only becomes readable
once displacement is learned, which is most of the way through a real run.
"""

import argparse
import importlib
import json
import time
from pathlib import Path

from exemplars.nano_motion import spec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tokenizer", default=spec.TOKENIZER,
                    help="which shelf tokenizer's recipe to reproduce")
    ap.add_argument("--data", default=spec.SOURCE, help="rot139 source to train on")
    ap.add_argument("--steps", type=int, default=None, help="override the recipe's steps")
    ap.add_argument("--out", default=None, help="where to write the artifact")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    mod = importlib.import_module(f"modalities.motion.tokenizers.{args.tokenizer}")
    recipe = mod.recipe()
    overrides = {"data": args.data}
    if args.steps:
        overrides["steps"] = args.steps

    out = Path(args.out) if args.out else \
        spec.MODELS / f"{args.tokenizer}_{args.data}_{args.steps or recipe['steps']}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"training {args.tokenizer} on {args.data}")
    print(f"  recipe: {json.dumps({**recipe, **overrides}, default=str)}")
    print(f"  -> {out}\n", flush=True)

    t0 = time.time()
    from modalities.motion.tokenizers._convae.train import train_codec
    metrics = train_codec({**recipe, **overrides}, str(out), device=args.device)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    print(f"  fair global-canonical recon: {json.dumps(metrics, default=str)}")
    print(f"\nnext: python -m exemplars.nano_motion.data.encode --codec {out}")


if __name__ == "__main__":
    main()
