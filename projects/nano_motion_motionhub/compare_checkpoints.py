"""Select an early-stop checkpoint using a predeclared balanced semantic rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_sweep(path: Path) -> dict:
    payload = json.loads(path.read_text())
    candidates = payload.get("candidates", [])
    return {
        "generated": len(candidates),
        "passing": int(payload.get("passing_seeds", 0)),
        "result": payload.get("result"),
        "schema": payload.get("schema"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-2000", type=Path, required=True)
    parser.add_argument("--stage-5000", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stage_reports = [
        json.loads(args.stage_2000.read_text()),
        json.loads(args.stage_5000.read_text()),
    ]
    validation = {
        int(row["step"]): float(row["motion_ce"])
        for stage in stage_reports
        for row in stage["training"]["validation"]
    }
    checkpoints = []
    for step in args.steps:
        sweeps = {
            "forward": load_sweep(args.reports / f"forward_{step}.json"),
            "left": load_sweep(args.reports / f"left_strict_{step}.json"),
            "right": load_sweep(args.reports / f"right_strict_{step}.json"),
        }
        complete = all(row["generated"] == args.expected_seeds for row in sweeps.values())
        rates = {
            name: row["passing"] / row["generated"] if row["generated"] else 0.0
            for name, row in sweeps.items()
        }
        overall = sum(row["passing"] for row in sweeps.values()) / sum(
            row["generated"] for row in sweeps.values()
        )
        balanced_turn = min(rates["left"], rates["right"])
        eligible = complete and rates["forward"] >= 0.5
        checkpoints.append(
            {
                "step": step,
                "validation_ce": validation[step],
                "complete": complete,
                "eligible": eligible,
                "rates": rates,
                "balanced_turn_rate": balanced_turn,
                "overall_pass_rate": overall,
                "sweeps": sweeps,
            }
        )

    # Declared before running the sweep: keep basic forward competence, then
    # prioritize the weaker turn direction, total semantic pass rate, and val CE.
    eligible = [row for row in checkpoints if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["balanced_turn_rate"],
            row["overall_pass_rate"],
            -row["validation_ce"],
        ),
    ) if eligible else None
    report = {
        "schema": "nano-motion-motionhub-checkpoint-selection-v1",
        "result": "passed" if selected else "failed",
        "selection_rule": {
            "minimum_forward_pass_rate": 0.5,
            "rank_order": [
                "balanced_turn_rate=max(min(left_rate,right_rate))",
                "overall_pass_rate=max",
                "validation_ce=min",
            ],
            "uses_tracker_outcomes": False,
        },
        "expected_seeds_per_prompt": args.expected_seeds,
        "checkpoints": checkpoints,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if selected is None:
        raise SystemExit("no checkpoint met the declared forward-completeness gate")


if __name__ == "__main__":
    main()
