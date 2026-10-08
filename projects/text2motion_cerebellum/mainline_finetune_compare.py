"""Compare a continued upstream tracker with its exact pre-training baseline."""

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
    expected_clips: int,
    expected_training_seed: int,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "motion-tracking-episodes-v1":
        raise ValueError(f"{path}: unexpected schema")
    rows = list(payload.get("episodes", []))
    keys = [(int(r["seed"]), int(r["repeat"]), str(r["clip"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate episode key")
    if sorted({key[0] for key in keys}) != [expected_training_seed]:
        raise ValueError(
            f"{path}: expected only training seed {expected_training_seed}"
        )
    if sorted({key[1] for key in keys}) != [0, 1, 2, 3]:
        raise ValueError(f"{path}: expected four observation-noise repeats")
    counts: dict[int, int] = defaultdict(int)
    for _, repeat, _ in keys:
        counts[repeat] += 1
    if any(counts[repeat] != expected_clips for repeat in range(4)):
        raise ValueError(f"{path}: expected {expected_clips} clips per repeat")
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: fmean(float(row[metric]) for row in rows) for metric in METRICS}


def paired(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    before_by_key = {
        (int(r["repeat"]), str(r["clip"])): r
        for r in before
    }
    after_by_key = {
        (int(r["repeat"]), str(r["clip"])): r
        for r in after
    }
    if before_by_key.keys() != after_by_key.keys():
        raise ValueError("before/after episode keys do not match")
    before_agg = aggregate(before)
    after_agg = aggregate(after)
    return {
        "pairs": len(before_by_key),
        "before": before_agg,
        "after": after_agg,
        "after_minus_before": {
            metric: after_agg[metric] - before_agg[metric]
            for metric in METRICS
        },
    }


def by_clip(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["clip"])].append(row)
    return {clip: aggregate(values) for clip, values in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-native", type=Path, required=True)
    parser.add_argument("--after-native", type=Path, required=True)
    parser.add_argument("--before-omg", type=Path, required=True)
    parser.add_argument("--after-omg", type=Path, required=True)
    parser.add_argument("--start-iteration", type=int, required=True)
    parser.add_argument("--added-iterations", type=int, required=True)
    parser.add_argument("--training-clips", type=int, required=True)
    parser.add_argument("--before-training-seed", type=int, default=2)
    parser.add_argument("--after-training-seed", type=int, default=2)
    parser.add_argument("--tracker-commit", required=True)
    parser.add_argument("--model-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    native_before = load(args.before_native, 60, args.before_training_seed)
    native_after = load(args.after_native, 60, args.after_training_seed)
    omg_before = load(args.before_omg, 3, args.before_training_seed)
    omg_after = load(args.after_omg, 3, args.after_training_seed)
    native = paired(native_before, native_after)
    omg = paired(omg_before, omg_after)

    credible_native = (
        native["after"]["succ"] >= 0.80
        and native["after"]["completion"] >= 0.90
    )
    demo_ready = (
        omg["after"]["succ"] >= 2 / 3
        and omg["after"]["completion"] >= 0.85
    )
    success_gain = native["after_minus_before"]["succ"]
    final_iteration = args.start_iteration + args.added_iterations
    if credible_native and demo_ready:
        route = "render_text2motion_demo"
    elif credible_native:
        route = "adapt_original_tracker_to_text2motion_refs"
    elif success_gain >= 0.05 and final_iteration < 7500:
        route = "continue_original_tracker_training"
    else:
        route = "review_training_recipe_before_more_compute"

    report = {
        "schema": "text2motion-cerebellum-mainline-finetune-v1",
        "run": {
            "tracker_commit": args.tracker_commit,
            "model_commit": args.model_commit,
            "seed": args.after_training_seed,
            "training_seed": args.after_training_seed,
            "baseline_training_seed": args.before_training_seed,
            "evaluation_clip_seed": 12345,
            "paired_across_training_seeds": (
                args.before_training_seed != args.after_training_seed
            ),
            "start_iteration": args.start_iteration,
            "added_iterations": args.added_iterations,
            "final_iteration": final_iteration,
            "training_clips": args.training_clips,
            "observation_noise": True,
            "evaluation_repeats": 4,
        },
        "native": native,
        "omg": omg,
        "omg_after_by_prompt": by_clip(omg_after),
        "decision": {
            "credible_native_baseline": credible_native,
            "text2motion_demo_ready": demo_ready,
            "next_route": route,
            "native_success_floor": 0.80,
            "native_completion_floor": 0.90,
            "demo_success_floor": 2 / 3,
            "demo_completion_floor": 0.85,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
