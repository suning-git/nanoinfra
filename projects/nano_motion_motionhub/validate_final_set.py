"""Validate and document the final forward/left/right rot139 set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_candidates import quality, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-seed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = {
        "forward": "A person walks forward at a steady pace and stops.",
        "left": "a man walks forward turns left.",
        "right": "A person walks forward and turns right.",
    }
    paths = {"forward": args.forward, "left": args.left, "right": args.right}
    selected = []
    for category in ("forward", "left", "right"):
        metrics = trajectory(paths[category])
        selected.append(
            {
                "category": category,
                "prompt": prompts[category],
                "file": paths[category].name,
                "metrics": metrics,
                "quality_gate": quality(category, metrics),
            }
        )
    sweep = json.loads(args.left_seed_report.read_text())
    report = {
        "schema": "nano-motion-motionhub-validated-set-v1",
        "result": "passed" if all(item["quality_gate"] for item in selected) else "failed",
        "selected": selected,
        "left_seed_search": {
            "tested": 32 + sweep["seeds"],
            "passing_in_extended_range": sweep["passing_seeds"],
            "selected_seed": int(sweep["selected"]["stem"].split("_")[-1]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
