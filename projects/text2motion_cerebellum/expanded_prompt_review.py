"""Validate and summarize the preregistered expanded Text2Motion prompt suite."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

try:
    from .multiseed_review import METRICS, T_95_DF2
except ImportError:  # Direct script execution puts this directory on sys.path.
    from multiseed_review import METRICS, T_95_DF2


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != "text2motion-expanded-prompts-preregistered-v1":
        raise ValueError("unexpected prompt protocol schema")
    prompts = payload.get("prompts", [])
    tags = [str(item["tag"]) for item in prompts]
    if len(tags) != 12 or len(tags) != len(set(tags)):
        raise ValueError("expected 12 unique preregistered prompt tags")
    if [item["source"] for item in prompts].count("new_generation") != 9:
        raise ValueError("expected exactly nine new generated prompts")
    return payload


def load_generation(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != "text2motion-expanded-prompt-generation-v1":
        raise ValueError("unexpected prompt generation schema")
    expected = [item["tag"] for item in protocol["prompts"]]
    actual = [item["tag"] for item in payload.get("prompts", [])]
    if actual != expected:
        raise ValueError("generation records do not match preregistered order")
    return payload


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: fmean(float(row[metric]) for row in rows) for metric in METRICS}


def across_values(values: list[float]) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError("exactly three training seeds are required")
    mean = fmean(values)
    sample_std = stdev(values)
    margin = T_95_DF2 * sample_std / math.sqrt(len(values))
    return {
        "mean": mean,
        "sample_std": sample_std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "min": min(values),
        "max": max(values),
    }


def load_run(
    path: Path, accepted_tags: list[str], repeats: int, training_seed: int
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != "motion-tracking-episodes-v1":
        raise ValueError(f"{path}: unexpected episode schema")
    rows = payload.get("episodes", [])
    if {int(row["seed"]) for row in rows} != {training_seed}:
        raise ValueError(f"{path}: training seed metadata mismatch")
    expected_count = len(accepted_tags) * repeats
    if len(rows) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} rows, got {len(rows)}")
    keys = [(int(row["repeat"]), str(row["clip"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate episode keys")
    if sorted({key[0] for key in keys}) != list(range(repeats)):
        raise ValueError(f"{path}: repeat mismatch")
    if sorted({key[1] for key in keys}) != sorted(accepted_tags):
        raise ValueError(f"{path}: accepted prompt mismatch")
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["clip"])].append(row)
    return {
        "episodes": len(rows),
        "aggregate": aggregate(rows),
        "by_prompt": {tag: aggregate(by_prompt[tag]) for tag in accepted_tags},
    }


def review(
    protocol: dict[str, Any],
    generation: dict[str, Any],
    runs: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    prompt_records = generation["prompts"]
    accepted_tags = [item["tag"] for item in prompt_records if item["quality_gate"] == "passed"]
    new_records = [item for item in prompt_records if item["source"] == "new_generation"]
    new_passed = [item for item in new_records if item["quality_gate"] == "passed"]
    total_prompts = len(prompt_records)
    repeats = int(protocol["evaluation_repeats"])
    thresholds = protocol["thresholds"]

    per_seed = []
    for training_seed, run in runs:
        tracking_success = run["aggregate"]["succ"]
        end_to_end_success = (
            tracking_success * len(accepted_tags) * repeats / (total_prompts * repeats)
        )
        per_seed.append(
            {
                "training_seed": training_seed,
                **run,
                "tracking_success_on_quality_passing_prompts": tracking_success,
                "end_to_end_success_over_all_preregistered_prompts": end_to_end_success,
            }
        )

    metric_values = {
        metric: [item["aggregate"][metric] for item in per_seed] for metric in METRICS
    }
    end_to_end_values = [
        item["end_to_end_success_over_all_preregistered_prompts"] for item in per_seed
    ]
    new_quality_rate = len(new_passed) / len(new_records)
    all_seed_tracking = all(
        item["aggregate"]["succ"] >= thresholds["minimum_per_seed_tracking_success"]
        and item["aggregate"]["completion"] >= thresholds["minimum_per_seed_completion"]
        for item in per_seed
    )
    all_seed_end_to_end = all(
        item["end_to_end_success_over_all_preregistered_prompts"]
        >= thresholds["minimum_per_seed_end_to_end_success"]
        for item in per_seed
    )
    quality_pass = new_quality_rate >= thresholds["minimum_new_quality_pass_rate"]

    return {
        "schema": "text2motion-expanded-prompt-review-v1",
        "protocol": {
            "total_preregistered_prompts": total_prompts,
            "new_generation_prompts": len(new_records),
            "generation_attempts_per_new_prompt": protocol["generation_attempts_per_new_prompt"],
            "evaluation_repeats": repeats,
            "training_seeds": [seed for seed, _ in runs],
            "selection_rule": protocol["selection_rule"],
            "thresholds": thresholds,
        },
        "generation": {
            "quality_passing_tags": accepted_tags,
            "quality_rejected_tags": [
                item["tag"] for item in prompt_records if item["quality_gate"] != "passed"
            ],
            "new_quality_passed": len(new_passed),
            "new_quality_total": len(new_records),
            "new_quality_pass_rate": new_quality_rate,
            "all_prompt_records": prompt_records,
        },
        "per_seed": per_seed,
        "tracking_across_training_seeds": {
            metric: across_values(values) for metric, values in metric_values.items()
        },
        "end_to_end_success_across_training_seeds": across_values(end_to_end_values),
        "decision": {
            "new_prompt_quality_gate_credible": quality_pass,
            "all_seeds_track_quality_passing_prompts": all_seed_tracking,
            "all_seeds_end_to_end_credible": all_seed_end_to_end,
            "expanded_demo_credible": quality_pass and all_seed_tracking and all_seed_end_to_end,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        required=True,
        metavar=("TRAINING_SEED", "EPISODES_JSON"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    generation = load_generation(args.generation, protocol)
    accepted = [
        item["tag"] for item in generation["prompts"] if item["quality_gate"] == "passed"
    ]
    repeats = int(protocol["evaluation_repeats"])
    runs = []
    for seed_text, path in args.run:
        training_seed = int(seed_text)
        runs.append(
            (training_seed, load_run(Path(path), accepted, repeats, training_seed))
        )
    if [seed for seed, _ in runs] != [0, 1, 2]:
        raise ValueError("runs must be supplied in training-seed order 0, 1, 2")
    result = review(protocol, generation, runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
