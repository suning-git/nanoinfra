"""Aggregate fixed-protocol motion-tracking evaluations across training seeds."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


METRICS = ("succ", "completion", "Empjpe", "Eg_mpjpe", "foot_slide", "jerk")
T_95_DF2 = 4.302652729911275


def load_suite(path: Path, training_seed: int, expected_clips: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "motion-tracking-episodes-v1":
        raise ValueError(f"{path}: unexpected schema")
    rows = list(payload.get("episodes", []))
    if {int(row["seed"]) for row in rows} != {training_seed}:
        raise ValueError(f"{path}: training seed metadata mismatch")
    keys = [(int(row["repeat"]), str(row["clip"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate episode key")
    if sorted({repeat for repeat, _ in keys}) != [0, 1, 2, 3]:
        raise ValueError(f"{path}: expected four repeats")
    counts: dict[int, int] = defaultdict(int)
    for repeat, _ in keys:
        counts[repeat] += 1
    if any(counts[repeat] != expected_clips for repeat in range(4)):
        raise ValueError(f"{path}: expected {expected_clips} clips per repeat")
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: fmean(float(row[metric]) for row in rows) for metric in METRICS}


def keys(rows: list[dict[str, Any]]) -> set[tuple[int, str]]:
    return {(int(row["repeat"]), str(row["clip"])) for row in rows}


def across_seeds(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if len(per_seed) != 3:
        raise ValueError("exactly three training seeds are required")
    report: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [row[metric] for row in per_seed]
        mean = fmean(values)
        sample_std = stdev(values)
        margin = T_95_DF2 * sample_std / math.sqrt(len(values))
        report[metric] = {
            "mean": mean,
            "sample_std": sample_std,
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
            "min": min(values),
            "max": max(values),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("TRAINING_SEED", "NATIVE_JSON", "OMG_JSON"),
        required=True,
    )
    parser.add_argument("--tracker-commit", required=True)
    parser.add_argument("--model-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    parsed = []
    for seed_text, native_text, omg_text in args.run:
        seed = int(seed_text)
        parsed.append({
            "seed": seed,
            "native": load_suite(Path(native_text), seed, 60),
            "omg": load_suite(Path(omg_text), seed, 3),
        })
    parsed.sort(key=lambda row: row["seed"])
    seeds = [row["seed"] for row in parsed]
    if seeds != [0, 1, 2]:
        raise ValueError(f"expected clean training seeds [0, 1, 2], got {seeds}")
    for suite in ("native", "omg"):
        reference_keys = keys(parsed[0][suite])
        if any(keys(row[suite]) != reference_keys for row in parsed[1:]):
            raise ValueError(f"{suite}: episode keys differ across training seeds")

    per_seed = []
    for row in parsed:
        per_seed.append({
            "training_seed": row["seed"],
            "native": aggregate(row["native"]),
            "omg": aggregate(row["omg"]),
        })
    native_across = across_seeds([row["native"] for row in per_seed])
    omg_across = across_seeds([row["omg"] for row in per_seed])
    all_native_credible = all(
        row["native"]["succ"] >= 0.80 and row["native"]["completion"] >= 0.90
        for row in per_seed
    )
    all_demo_ready = all(
        row["omg"]["succ"] >= 2 / 3 and row["omg"]["completion"] >= 0.85
        for row in per_seed
    )
    report = {
        "schema": "text2motion-clean-multiseed-review-v1",
        "protocol": {
            "training_seeds": seeds,
            "training_clips": 1500,
            "available_training_clips": 1694,
            "iterations": 4500,
            "environments": 3840,
            "workers": 80,
            "recipe": "g1/deployable",
            "clean_start": True,
            "observation_noise": True,
            "evaluation_clip_seed": 12345,
            "evaluation_repeats": 4,
            "native_episodes_per_seed": 240,
            "omg_episodes_per_seed": 12,
            "tracker_commit": args.tracker_commit,
            "model_commit": args.model_commit,
        },
        "per_seed": per_seed,
        "native_across_training_seeds": native_across,
        "omg_across_training_seeds": omg_across,
        "decision": {
            "all_native_seeds_credible": all_native_credible,
            "all_seeds_text2motion_demo_ready": all_demo_ready,
            "next_route": (
                "freeze_three_seed_clean_baseline_and_write_final_report"
                if all_native_credible and all_demo_ready
                else "review_seed_variance_before_claiming_reproduction"
            ),
        },
        "inference_note": (
            "The t intervals treat the three independently trained policies as the "
            "replication unit; with df=2 they are intentionally conservative."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
