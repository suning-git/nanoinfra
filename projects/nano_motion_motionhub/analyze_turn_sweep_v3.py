"""Evaluate left/right turns with one mirrored body-and-path semantic gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.nano_motion_cerebellum.analyze_right_sweep import directional_metrics
from projects.nano_motion_motionhub.analyze_candidates import score


def signed_metrics(direction: str, metrics: dict) -> tuple[float, float, float]:
    sign = 1.0 if direction == "left" else -1.0
    signed_yaw = sign * metrics["yaw_change_deg"]
    signed_path = sign * metrics["turn_deg"]
    opposite_excursion = (
        -metrics["yaw_negative_excursion_deg"]
        if direction == "left"
        else metrics["yaw_positive_excursion_deg"]
    )
    return signed_yaw, signed_path, opposite_excursion


def passes(direction: str, metrics: dict) -> bool:
    signed_yaw, signed_path, opposite = signed_metrics(direction, metrics)
    return (
        metrics["net_m"] >= 0.3
        and metrics["efficiency"] >= 0.3
        and 25.0 <= signed_yaw <= 120.0
        and metrics["yaw_range_deg"] <= 160.0
        and opposite <= 10.0
        and metrics["yaw_directional_ratio"] >= 0.65
        and 10.0 <= signed_path <= 120.0
    )


def directional_score(direction: str, metrics: dict) -> float:
    _, _, opposite = signed_metrics(direction, metrics)
    return score(direction, metrics) + metrics["yaw_directional_ratio"] - opposite / 90.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--direction", choices=("left", "right"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for path in sorted(args.generated.glob("seed_*.npz")):
        metrics = directional_metrics(path)
        candidates.append(
            {
                "seed": int(path.stem.split("_")[-1]),
                "stem": path.stem,
                "prompt": args.prompt,
                "metrics": metrics,
                "directional_gate": passes(args.direction, metrics),
                "score": directional_score(args.direction, metrics),
            }
        )
    if not candidates:
        raise ValueError(f"{args.direction} seed sweep produced no candidates")
    passing = [row for row in candidates if row["directional_gate"]]
    selected = max(passing, key=lambda row: row["score"]) if passing else None
    report = {
        "schema": "nano-motion-motionhub-turn-seed-sweep-v3",
        "result": "passed" if selected else "failed",
        "direction": args.direction,
        "selection_rule": {
            "kind": "mirrored_body_and_path_gate",
            "net_m_min": 0.3,
            "efficiency_min": 0.3,
            "signed_yaw_change_deg": [25.0, 120.0],
            "yaw_range_deg_max": 160.0,
            "opposite_yaw_excursion_deg_max": 10.0,
            "directional_ratio_min": 0.65,
            "signed_path_turn_deg": [10.0, 120.0],
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
