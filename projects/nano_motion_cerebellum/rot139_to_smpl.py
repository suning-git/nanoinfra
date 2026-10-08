"""Decode a nano_motion rot139 NPZ into an isolated SMPL-H handoff file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    from modalities.motion.data.converters import smpl_to_rot139

    with np.load(args.input, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 139 or len(features) < 2:
        raise ValueError(f"expected rot139 [T,139], got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("rot139 contains non-finite values")
    local_rotmat, trans = smpl_to_rot139.features_to_smpl(
        features.astype(np.float64), np.zeros(3, dtype=np.float64)
    )
    orthogonality = np.matmul(
        np.swapaxes(local_rotmat, -1, -2), local_rotmat
    ) - np.eye(3)
    max_orthogonality_error = float(np.abs(orthogonality).max())
    if max_orthogonality_error > 1e-4 or not np.isfinite(trans).all():
        raise ValueError("decoded SMPL-H transforms are invalid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        local_rotmat=local_rotmat.astype(np.float32),
        trans=trans.astype(np.float32),
        fps=np.float32(args.fps),
    )
    report = {
        "schema": "nano-motion-rot139-to-smpl-v1",
        "result": "passed",
        "frames": int(len(features)),
        "fps": float(args.fps),
        "max_orthogonality_error": max_orthogonality_error,
        "translation_range_m": (
            trans.max(axis=0) - trans.min(axis=0)
        ).astype(float).tolist(),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
