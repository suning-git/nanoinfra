"""Validate and summarize tracking episodes for the nano_motion reference set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


METRICS = ("succ", "completion", "Empjpe", "Eg_mpjpe", "foot_slide", "jerk")
EXPECTED_CLIPS = ("nano_motion_forward", "nano_motion_left", "nano_motion_right")


def aggregate(rows):
    return {metric: fmean(float(row[metric]) for row in rows) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", required=True,
                        metavar=("TRAINING_SEED", "EPISODES_JSON"))
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--tracker-commit", required=True)
    parser.add_argument("--expected-clips", nargs="+", default=list(EXPECTED_CLIPS))
    parser.add_argument("--required-success-clips", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_clips = tuple(args.expected_clips)
    required_success_clips = tuple(args.required_success_clips)
    if len(set(expected_clips)) != len(expected_clips):
        raise ValueError("expected clips must be unique")
    if not set(required_success_clips).issubset(expected_clips):
        raise ValueError("required success clips must be expected clips")

    per_seed = []
    reference_keys = None
    for seed_text, path_text in args.run:
        seed = int(seed_text)
        payload = json.loads(Path(path_text).read_text())
        if payload.get("schema") != "motion-tracking-episodes-v1":
            raise ValueError("unexpected episode schema")
        rows = list(payload.get("episodes", []))
        if len(rows) != len(expected_clips) * args.repeats:
            raise ValueError(f"seed {seed}: episode count mismatch")
        if {int(row["seed"]) for row in rows} != {seed}:
            raise ValueError(f"seed {seed}: policy seed metadata mismatch")
        keys = {(int(row["repeat"]), str(row["clip"])) for row in rows}
        if len(keys) != len(rows):
            raise ValueError(f"seed {seed}: duplicate episode keys")
        if {key[0] for key in keys} != set(range(args.repeats)):
            raise ValueError(f"seed {seed}: repeat mismatch")
        if {key[1] for key in keys} != set(expected_clips):
            raise ValueError(f"seed {seed}: clip mismatch")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("episode keys differ across policy seeds")
        by_prompt = defaultdict(list)
        for row in rows:
            by_prompt[str(row["clip"])].append(row)
        summary = aggregate(rows)
        by_prompt_summary = {
            clip: aggregate(by_prompt[clip]) for clip in expected_clips
        }
        passes_required = all(
            by_prompt_summary[clip]["succ"] == 1.0
            for clip in required_success_clips
        )
        per_seed.append({
            "training_seed": seed,
            "episodes": len(rows),
            "aggregate": summary,
            "by_prompt": by_prompt_summary,
            "passes_required_clips": passes_required,
            "passes_demo_floor": (
                summary["succ"] >= 2 / 3
                and summary["completion"] >= 0.85
                and passes_required
            ),
        })
    per_seed.sort(key=lambda row: row["training_seed"])
    across = {}
    if len(per_seed) == 3:
        for metric in METRICS:
            values = [row["aggregate"][metric] for row in per_seed]
            across[metric] = {"mean": fmean(values), "sample_std": stdev(values)}
    report = {
        "schema": "nano-motion-cerebellum-tracking-review-v1",
        "result": "passed" if all(row["passes_demo_floor"] for row in per_seed) else "failed",
        "protocol": {
            "training_seeds": [row["training_seed"] for row in per_seed],
            "observation_noise": True,
            "repeats": args.repeats,
            "reference_clips": list(expected_clips),
            "required_success_clips": list(required_success_clips),
            "success_floor": 2 / 3,
            "completion_floor": 0.85,
            "tracker_commit": args.tracker_commit,
        },
        "per_seed": per_seed,
        "across_training_seeds": across,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
