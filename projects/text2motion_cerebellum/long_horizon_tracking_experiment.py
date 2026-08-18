"""Track every quality-passing sanitized long OMG motion on three cerebella."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    from .expanded_prompt_review import across_values
    from .multiseed_review import METRICS
    from .single_chunk_tracking_experiment import load_episodes
except ImportError:
    from expanded_prompt_review import across_values
    from multiseed_review import METRICS
    from single_chunk_tracking_experiment import load_episodes


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty episode set")
    return {metric: fmean(float(row[metric]) for row in rows) for metric in METRICS}


def review(
    inventory: list[dict[str, Any]],
    runs: list[tuple[int, list[dict[str, Any]]]],
    *,
    repeats: int = 4,
) -> dict[str, Any]:
    if len(inventory) != 27:
        raise ValueError("expected 27 generation cells")
    accepted = [row for row in inventory if row["quality_gate"] == "passed"]
    if len(accepted) != 18:
        raise ValueError("expected 18 preregistered quality-passing cells")
    accepted_lookup = {str(row["clip"]): row for row in accepted}
    if len(accepted_lookup) != len(accepted):
        raise ValueError("accepted clip names must be unique")

    per_training_seed = []
    for training_seed, rows in runs:
        tracking = aggregate(rows)
        end_to_end = sum(float(row["succ"]) for row in rows) / (
            len(inventory) * repeats
        )
        by_generation_seed: dict[str, Any] = {}
        for generation_seed in (0, 1, 2):
            selected = [
                row
                for row in rows
                if int(accepted_lookup[str(row["clip"])]["generation_seed"])
                == generation_seed
            ]
            passed_count = sum(
                int(row["generation_seed"]) == generation_seed for row in accepted
            )
            by_generation_seed[str(generation_seed)] = {
                "quality_passed": passed_count,
                "tracking": aggregate(selected),
                "end_to_end_success": sum(float(row["succ"]) for row in selected)
                / (9 * repeats),
            }
        per_training_seed.append(
            {
                "training_seed": training_seed,
                "episodes": len(rows),
                "tracking_on_quality_passing_references": tracking,
                "end_to_end_success_over_27_generation_cells": end_to_end,
                "by_generation_seed": by_generation_seed,
            }
        )

    tracking_across = {
        metric: across_values(
            [row["tracking_on_quality_passing_references"][metric] for row in per_training_seed]
        )
        for metric in METRICS
    }
    end_to_end_across = across_values(
        [row["end_to_end_success_over_27_generation_cells"] for row in per_training_seed]
    )
    quality_rate = len(accepted) / len(inventory)
    tracking_floor = 0.75
    completion_floor = 0.90
    end_to_end_floor = 0.60
    all_trackers = all(
        row["tracking_on_quality_passing_references"]["succ"] >= tracking_floor
        and row["tracking_on_quality_passing_references"]["completion"] >= completion_floor
        for row in per_training_seed
    )
    all_end_to_end = all(
        row["end_to_end_success_over_27_generation_cells"] >= end_to_end_floor
        for row in per_training_seed
    )
    return {
        "schema": "text2motion-long-horizon-tracking-review-v1",
        "classification": "post_hoc_long_horizon_sanitized_interface_followup",
        "protocol": {
            "generation_seeds": [0, 1, 2],
            "generation_cells": 27,
            "quality_passing_cells": len(accepted),
            "evaluation_repeats": repeats,
            "training_seeds": [seed for seed, _ in runs],
            "selection_rule": "track all 18 quality-passing sanitized cells; rejected cells count as end-to-end failures",
        },
        "generation": {
            "quality_pass_rate": quality_rate,
            "accepted_clips": [row["clip"] for row in accepted],
            "rejected_cells": [
                {
                    "generation_seed": row["generation_seed"],
                    "tag": row["tag"],
                    "reason": row["gate_reason"],
                }
                for row in inventory
                if row["quality_gate"] != "passed"
            ],
        },
        "per_training_seed": per_training_seed,
        "tracking_across_training_seeds": tracking_across,
        "end_to_end_success_across_training_seeds": end_to_end_across,
        "decision": {
            "quality_floor": 2 / 3,
            "tracking_success_floor": tracking_floor,
            "tracking_completion_floor": completion_floor,
            "end_to_end_success_floor": end_to_end_floor,
            "quality_passed": quality_rate >= 2 / 3,
            "all_trackers_passed": all_trackers,
            "all_end_to_end_passed": all_end_to_end,
            "long_horizon_sanitized_demo_credible": (
                quality_rate >= 2 / 3 and all_trackers and all_end_to_end
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sanitizer-result", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--policy", action="append", nargs=2, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    payload = json.loads(args.sanitizer_result.read_text(encoding="utf-8"))
    if payload.get("schema") != "text2motion-long-horizon-sanitizer-experiment-v1":
        raise ValueError("unexpected sanitizer result schema")
    if payload.get("result") != "eligible_for_tracking":
        raise ValueError("sanitizer result was not eligible for tracking")
    records = list(payload["records"])
    if len(records) != 27:
        raise ValueError("expected 27 sanitizer records")

    accepted_root = args.out / "accepted_refs"
    logs_root = args.out / "logs"
    accepted_root.mkdir(parents=True)
    logs_root.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    inventory = []
    accepted_index = 0
    with (logs_root / "run.log").open("w", encoding="utf-8") as log:
        for record in records:
            generation_seed = int(record["generation_seed"])
            tag = str(record["tag"])
            clip = f"long_gseed{generation_seed}__{tag}"
            item = {
                "generation_seed": generation_seed,
                "tag": tag,
                "text": record["text"],
                "clip": clip,
                "quality_gate": record["quality_gate"],
                "gate_reason": record["gate_reason"],
                "accepted_shard": None,
            }
            if record["quality_gate"] == "passed":
                source = (
                    args.generated_root
                    / f"seed_{generation_seed}"
                    / tag
                    / "reference_motion.npz"
                )
                shard_name = f"shard_{accepted_index:03d}.npz"
                shard = accepted_root / shard_name
                adapter_environment = dict(environment)
                adapter_environment["PYTHONPATH"] = f"{args.root}:{args.tracker_repo}"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "projects.text2motion_cerebellum.omg_adapter",
                        "convert",
                        str(source),
                        str(shard),
                        "--tracker-repo",
                        str(args.tracker_repo),
                        "--caption",
                        record["text"],
                        "--clip",
                        clip,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=adapter_environment,
                    check=False,
                )
                if completed.returncode != 0 or not shard.is_file():
                    raise RuntimeError(f"accepted reference conversion failed: {clip}")
                item["accepted_shard"] = shard_name
                accepted_index += 1
            inventory.append(item)

        if accepted_index != 18:
            raise RuntimeError(f"expected 18 accepted references, got {accepted_index}")
        write(
            args.out / "inventory.json",
            {
                "schema": "text2motion-long-horizon-reference-inventory-v1",
                "source_artifact": str(args.sanitizer_result),
                "selection_rule": "all 27 sanitizer cells; all 18 quality-passing cells are tracked",
                "records": inventory,
            },
        )

        tracker_environment = dict(environment)
        tracker_environment["PYTHONPATH"] = (
            f"{args.root}:{args.tracker_repo}:{args.root / 'projects/motion_cerebellum_remote'}"
        )
        policies = [(int(seed), Path(path)) for seed, path in args.policy]
        if [seed for seed, _ in policies] != [0, 1, 2]:
            raise ValueError("policies must be supplied in training-seed order")
        accepted_clips = [row["clip"] for row in inventory if row["quality_gate"] == "passed"]
        runs = []
        for training_seed, policy in policies:
            episodes = args.out / f"seed{training_seed}_episodes.json"
            completed = subprocess.run(
                [
                    str(args.train_python),
                    str(args.wrapper),
                    "evaluate",
                    "--tracker-repo",
                    str(args.tracker_repo),
                    "--preview-offsets",
                    "auto",
                    "--",
                    "--robot",
                    "g1",
                    "--policy",
                    str(policy),
                    "--label",
                    f"long_horizon_sanitized_tracker{training_seed}",
                    "--ref-dir",
                    str(accepted_root),
                    "--split",
                    "all",
                    "--amass",
                    str(accepted_index),
                    "--obs-noise",
                    "--repeats",
                    "4",
                    "--episodes-json",
                    str(episodes),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=tracker_environment,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"tracker evaluation failed for training seed {training_seed}")
            runs.append(
                (
                    training_seed,
                    load_episodes(
                        episodes,
                        training_seed=training_seed,
                        accepted_clips=accepted_clips,
                        repeats=4,
                    ),
                )
            )

    write(args.out / "summary.json", review(inventory, runs))


if __name__ == "__main__":
    main()
