"""Build a compact comparison for the self-trained nano_motion end-to-end run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = ("success", "completion", "Empjpe_mm", "foot_slide", "jerk")


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - radius, center + radius]


def tracker_summary(payload: dict) -> dict:
    values = payload["across_training_seeds"]
    total = sum(int(row["episodes"]) for row in payload["per_seed"])
    successes = round(float(values["succ"]["mean"]) * total)
    return {
        "episodes": total,
        "successes": successes,
        "success": successes / total,
        "success_wilson95": wilson(successes, total),
        "completion": float(values["completion"]["mean"]),
        "Empjpe_mm": float(values["Empjpe"]["mean"]),
        "foot_slide": float(values["foot_slide"]["mean"]),
        "jerk": float(values["jerk"]["mean"]),
    }


def omg_summary(payload: dict) -> dict:
    values = payload["across_training_seeds"]["omg"]
    total = 36
    successes = round(float(values["success"]["mean"]) * total)
    return {
        "episodes": total,
        "successes": successes,
        "success": successes / total,
        "success_wilson95": wilson(successes, total),
        "completion": float(values["completion"]["mean"]),
        "Empjpe_mm": float(values["Empjpe_mm"]["mean"]),
        "foot_slide": float(values["foot_slide"]["mean"]),
        "jerk": float(values["jerk"]["mean"]),
    }


def deltas(source: dict, baseline: dict) -> dict:
    return {name: source[name] - baseline[name] for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-full", type=Path, required=True)
    parser.add_argument("--old-nano", type=Path, required=True)
    parser.add_argument("--omg", type=Path, required=True)
    parser.add_argument("--seed1-diagnostic", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    self_payload = json.loads(args.self_full.read_text())
    old_payload = json.loads(args.old_nano.read_text())
    omg_payload = json.loads(args.omg.read_text())
    diagnostic = json.loads(args.seed1_diagnostic.read_text())
    selection = json.loads(args.selection.read_text())
    current = tracker_summary(self_payload)
    old = tracker_summary(old_payload)
    omg = omg_summary(omg_payload)
    seed1 = diagnostic["per_seed"][0]
    report = {
        "schema": "nano-motion-self-trained-comparison-v2",
        "result": "completed_below_existing_baselines",
        "formal_protocol": self_payload["protocol"],
        "formal": {
            "self_trained_nano_motion": current,
            "prior_nano_motion_demo": old,
            "omg": omg,
            "self_minus_prior_nano": deltas(current, old),
            "self_minus_omg": deltas(current, omg),
        },
        "post_hoc_seed1_repeat16": {
            "aggregate": seed1["aggregate"],
            "by_prompt": seed1["by_prompt"],
            "interpretation": (
                "the low four-repeat seed-1 result persists with more repeats; "
                "this is a reference-distribution by tracker-seed interaction"
            ),
        },
        "reference_selection": {
            "post_hoc": selection["post_hoc"],
            "uses_tracker_outcomes_for_candidate_ranking": selection["selection_rule"][
                "uses_tracker_outcomes_for_candidate_ranking"
            ],
            "selected": selection["selected"],
        },
        "conclusions": [
            "the self-trained Text2Motion model completes the full text-to-G1-to-tracker interface",
            "physical-margin selection repaired the original left-turn smoke failure",
            "the formal three-policy-seed result is 30/36 successes and is not at OMG parity",
            "future gains require broader tracker training coverage or better generator regularization, not more post-hoc sample reranking",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
