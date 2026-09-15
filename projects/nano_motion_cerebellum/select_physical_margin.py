"""Select a semantic-passing reference by frozen physical-gate margin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LIMITS = {
    "max_joint_speed_rad_s": 15.0,
    "foot_slide": 12.0,
    "hover": 50.0,
}


def candidate_seed(candidate: dict) -> int:
    if "seed" in candidate:
        return int(candidate["seed"])
    return int(str(candidate["stem"]).split("_")[-1])


def stress(report: dict) -> tuple[float, float]:
    ratios = [float(report[name]) / limit for name, limit in LIMITS.items()]
    return max(ratios), sum(ratios) / len(ratios)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-report", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--max-semantic-candidates", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_semantic_candidates <= 0:
        raise ValueError("max semantic candidates must be positive")

    sweep = json.loads(args.sweep_report.read_text())
    semantic = sorted(
        (row for row in sweep["candidates"] if row.get("directional_gate") is True),
        key=lambda row: (-float(row["score"]), candidate_seed(row)),
    )[: args.max_semantic_candidates]
    reports = {}
    for path in args.attempts.glob("seed_*_retarget.json"):
        seed = int(path.name.split("_")[1])
        reports[seed] = json.loads(path.read_text())
    expected = {candidate_seed(row) for row in semantic}
    if set(reports) != expected:
        raise ValueError("retarget reports do not match the frozen semantic shortlist")

    rows = []
    for candidate in semantic:
        seed = candidate_seed(candidate)
        physical = reports[seed]
        row = {
            "seed": seed,
            "stem": candidate["stem"],
            "semantic_score": float(candidate["score"]),
            **{
                key: physical[key]
                for key in (
                    "result",
                    "quality_gate_reason",
                    "g1_frames",
                    "root_displacement_m",
                    "max_joint_speed_rad_s",
                    "foot_slide",
                    "hover",
                )
                if key in physical
            },
        }
        if physical.get("result") == "passed":
            row["physical_stress_max"], row["physical_stress_mean"] = stress(physical)
        rows.append(row)
    passing = [row for row in rows if row["result"] == "passed"]
    selected = min(
        passing,
        key=lambda row: (
            row["physical_stress_max"],
            row["physical_stress_mean"],
            -row["semantic_score"],
            row["seed"],
        ),
    ) if passing else None

    report = {
        "schema": "nano-motion-self-physical-margin-selection-v1",
        "result": "passed" if selected else "failed",
        "category": "left",
        "prompt": sweep["prompt"],
        "post_hoc": True,
        "post_hoc_reason": (
            "the original highest-semantic-score left reference failed the frozen "
            "tracker smoke test"
        ),
        "selection_rule": {
            "semantic_gate": "unchanged mirrored body-and-path directional gate",
            "semantic_shortlist": (
                f"top {args.max_semantic_candidates} passing candidates by frozen score"
            ),
            "physical_acceptance": "unmodified upstream quality gates",
            "physical_limits": LIMITS,
            "ranking": [
                "minimum maximum normalized physical-gate ratio",
                "minimum mean normalized physical-gate ratio",
                "higher frozen semantic score",
                "lower generation seed",
            ],
            "uses_tracker_outcomes_for_candidate_ranking": False,
        },
        "total_generated": int(sweep["seeds"]),
        "semantic_passing": int(sweep["passing_seeds"]),
        "retargeted": len(rows),
        "physical_passing": len(passing),
        "selected": selected,
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if selected is None:
        raise SystemExit("no shortlisted left candidate passed physical gates")


if __name__ == "__main__":
    main()
