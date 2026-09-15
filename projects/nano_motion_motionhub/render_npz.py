"""Render one generated rot139 NPZ to a GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    from exemplars.nano_motion import render

    with np.load(args.input, allow_pickle=False) as data:
        features = data["features"]
    render.features_to_gif(features, str(args.output), title=args.title)


if __name__ == "__main__":
    main()
