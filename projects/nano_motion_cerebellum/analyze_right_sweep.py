"""Select a right-turn seed using frozen directional trajectory gates only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.nano_motion_motionhub.analyze_candidates import score, trajectory


def directional_metrics(path: Path) -> dict:
    base = trajectory(path)
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
    d6 = features[:, :6]
    first, second = d6[:, :3], d6[:, 3:6]
    axis1 = first / (np.linalg.norm(first, axis=1, keepdims=True) + 1e-8)
    second = second - (axis1 * second).sum(axis=1, keepdims=True) * axis1
    axis2 = second / (np.linalg.norm(second, axis=1, keepdims=True) + 1e-8)
    axis3 = np.cross(axis1, axis2)
    rotation = np.stack((axis1, axis2, axis3), axis=-1)
    yaw = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    window = max(3, min(15, len(yaw) // 10))
    start = float(np.median(yaw[:window]))
    relative_deg = np.degrees(yaw - start)
    base.update({
        "yaw_positive_excursion_deg": float(max(0.0, relative_deg.max())),
        "yaw_negative_excursion_deg": float(min(0.0, relative_deg.min())),
        "yaw_directional_ratio": float(
            abs(base["yaw_change_deg"]) / max(base["yaw_range_deg"], 1e-9)
        ),
    })
    return base


def passes(metrics: dict) -> bool:
    return (
        metrics["net_m"] >= 0.3
        and metrics["efficiency"] >= 0.3
        and -120.0 <= metrics["yaw_change_deg"] <= -25.0
        and metrics["yaw_range_deg"] <= 160.0
        and metrics["yaw_positive_excursion_deg"] <= 10.0
        and metrics["yaw_directional_ratio"] >= 0.65
    )


def directional_score(metrics: dict) -> float:
    return (
        score("right", metrics)
        + metrics["yaw_directional_ratio"]
        - metrics["yaw_positive_excursion_deg"] / 90.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for path in sorted(args.generated.glob("seed_*.npz")):
        metrics = directional_metrics(path)
        candidates.append({
            "seed": int(path.stem.split("_")[-1]),
            "stem": path.stem,
            "prompt": args.prompt,
            "metrics": metrics,
            "directional_gate": passes(metrics),
            "score": directional_score(metrics),
        })
    if not candidates:
        raise ValueError("right seed sweep produced no candidates")
    passing = [row for row in candidates if row["directional_gate"]]
    selected = max(passing, key=lambda row: row["score"]) if passing else None
    report = {
        "schema": "nano-motion-cerebellum-right-seed-sweep-v1",
        "result": "passed" if selected else "failed",
        "selection_rule": {
            "net_m_min": 0.3,
            "efficiency_min": 0.3,
            "yaw_change_deg": [-120.0, -25.0],
            "yaw_range_deg_max": 160.0,
            "opposite_yaw_excursion_deg_max": 10.0,
            "directional_ratio_min": 0.65,
            "uses_tracker_outcomes": False,
        },
        "prompt": args.prompt,
        "seeds": len(candidates),
        "passing_seeds": len(passing),
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
