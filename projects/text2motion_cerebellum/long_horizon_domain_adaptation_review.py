"""Review a preregistered long-horizon cerebellum domain-adaptation smoke."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


METRICS = ("succ", "completion", "Empjpe", "Eg_mpjpe", "foot_slide", "jerk")


def load(
    path: Path,
    *,
    expected_clips: int,
    expected_seed: int,
    repeats: int = 4,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "motion-tracking-episodes-v1":
        raise ValueError(f"{path}: unexpected schema")
    rows = list(payload.get("episodes", []))
    if len(rows) != expected_clips * repeats:
        raise ValueError(f"{path}: episode count mismatch")
    if {int(row["seed"]) for row in rows} != {expected_seed}:
        raise ValueError(f"{path}: training-seed mismatch")
    keys = [(int(row["repeat"]), str(row["clip"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate episode keys")
    if sorted({repeat for repeat, _ in keys}) != list(range(repeats)):
        raise ValueError(f"{path}: repeat mismatch")
    counts: dict[int, int] = defaultdict(int)
    for repeat, _ in keys:
        counts[repeat] += 1
    if any(counts[index] != expected_clips for index in range(repeats)):
        raise ValueError(f"{path}: clip count mismatch")
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: fmean(float(row[metric]) for row in rows) for metric in METRICS}


def paired(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, Any]:
    before_by_key = {(int(row["repeat"]), str(row["clip"])): row for row in before}
    after_by_key = {(int(row["repeat"]), str(row["clip"])): row for row in after}
    if before_by_key.keys() != after_by_key.keys():
        raise ValueError("before/after episode keys differ")
    before_metrics = aggregate(before)
    after_metrics = aggregate(after)
    return {
        "pairs": len(before_by_key),
        "before": before_metrics,
        "after": after_metrics,
        "after_minus_before": {
            metric: after_metrics[metric] - before_metrics[metric] for metric in METRICS
        },
    }


def review(
    native_before: list[dict[str, Any]],
    native_after: list[dict[str, Any]],
    holdout_before: list[dict[str, Any]],
    holdout_after: list[dict[str, Any]],
    *,
    training_references: int = 12,
    native_replay_references: int = 0,
) -> dict[str, Any]:
    native = paired(native_before, native_after)
    holdout = paired(holdout_before, holdout_after)
    heldout_gain = (
        holdout["after_minus_before"]["succ"] >= 0.10
        or holdout["after_minus_before"]["completion"] >= 0.05
    )
    heldout_completion = holdout["after"]["completion"] >= 0.90
    native_preserved = (
        native["after_minus_before"]["succ"] >= -0.05
        and native["after_minus_before"]["completion"] >= -0.03
    )
    return {
        "schema": "text2motion-long-horizon-domain-adaptation-review-v1",
        "protocol": {
            "training_generation_seeds": [0, 1],
            "heldout_generation_seeds": [2],
            "training_references": training_references,
            "native_replay_references": native_replay_references,
            "long_horizon_training_references": 12,
            "heldout_references": 6,
            "added_iterations": 300,
            "observation_noise": True,
            "evaluation_repeats": 4,
            "training_policy_seed": 0,
        },
        "native": native,
        "heldout_long_horizon": holdout,
        "criteria": {
            "heldout_success_gain_at_least_0_10_or_completion_gain_at_least_0_05": heldout_gain,
            "heldout_completion_at_least_0_90": heldout_completion,
            "native_success_drop_at_most_0_05_and_completion_drop_at_most_0_03": native_preserved,
        },
        "expand_to_three_policy_finetune": (
            heldout_gain and heldout_completion and native_preserved
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-before", type=Path, required=True)
    parser.add_argument("--native-after", type=Path, required=True)
    parser.add_argument("--holdout-before", type=Path, required=True)
    parser.add_argument("--holdout-after", type=Path, required=True)
    parser.add_argument("--training-references", type=int, default=12)
    parser.add_argument("--native-replay-references", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = review(
        load(args.native_before, expected_clips=60, expected_seed=0),
        load(args.native_after, expected_clips=60, expected_seed=0),
        load(args.holdout_before, expected_clips=6, expected_seed=0),
        load(args.holdout_after, expected_clips=6, expected_seed=0),
        training_references=args.training_references,
        native_replay_references=args.native_replay_references,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
