"""Evaluate every quality-passing short OMG generation on three frozen trackers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    from .expanded_prompt_review import across_values
    from .multiseed_review import METRICS
except ImportError:
    from expanded_prompt_review import across_values
    from multiseed_review import METRICS


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty episode set")
    return {metric: fmean(float(row[metric]) for row in rows) for metric in METRICS}


def load_episodes(
    path: Path,
    *,
    training_seed: int,
    accepted_clips: list[str],
    repeats: int,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "motion-tracking-episodes-v1":
        raise ValueError(f"{path}: unexpected episode schema")
    rows = payload.get("episodes", [])
    if len(rows) != len(accepted_clips) * repeats:
        raise ValueError(f"{path}: episode count mismatch")
    if {int(row["seed"]) for row in rows} != {training_seed}:
        raise ValueError(f"{path}: training seed mismatch")
    keys = [(int(row["repeat"]), str(row["clip"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate episode keys")
    if sorted({repeat for repeat, _ in keys}) != list(range(repeats)):
        raise ValueError(f"{path}: repeat mismatch")
    if sorted({clip for _, clip in keys}) != sorted(accepted_clips):
        raise ValueError(f"{path}: clip inventory mismatch")
    return rows


def review(
    inventory: list[dict[str, Any]],
    runs: list[tuple[int, list[dict[str, Any]]]],
    *,
    repeats: int = 4,
) -> dict[str, Any]:
    if len(inventory) != 27:
        raise ValueError("expected 27 generation-seed by prompt cells")
    accepted = [row for row in inventory if row["quality_gate"] == "passed"]
    accepted_clips = [row["clip"] for row in accepted]
    if len(accepted_clips) != len(set(accepted_clips)):
        raise ValueError("accepted clip names must be unique")
    accepted_lookup = {row["clip"]: row for row in accepted}

    per_training_seed = []
    for training_seed, rows in runs:
        metrics = aggregate(rows)
        end_to_end = sum(float(row["succ"]) for row in rows) / (len(inventory) * repeats)
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
                "tracking": aggregate(selected) if selected else None,
                "end_to_end_success": (
                    sum(float(row["succ"]) for row in selected) / (9 * repeats)
                ),
            }
        per_training_seed.append(
            {
                "training_seed": training_seed,
                "episodes": len(rows),
                "tracking_on_quality_passing_references": metrics,
                "end_to_end_success_over_27_generation_cells": end_to_end,
                "by_generation_seed": by_generation_seed,
            }
        )

    tracking_values = {
        metric: [
            row["tracking_on_quality_passing_references"][metric]
            for row in per_training_seed
        ]
        for metric in METRICS
    }
    end_to_end_values = [
        row["end_to_end_success_over_27_generation_cells"]
        for row in per_training_seed
    ]
    quality_rate = len(accepted) / len(inventory)
    quality_floor = 2 / 3
    tracking_floor = 0.75
    completion_floor = 0.90
    end_to_end_floor = 0.60
    tracking_pass = all(
        row["tracking_on_quality_passing_references"]["succ"] >= tracking_floor
        and row["tracking_on_quality_passing_references"]["completion"]
        >= completion_floor
        for row in per_training_seed
    )
    end_to_end_pass = all(value >= end_to_end_floor for value in end_to_end_values)
    return {
        "schema": "text2motion-single-chunk-tracking-review-v1",
        "classification": "post_hoc_short_horizon_followup",
        "protocol": {
            "generation_seeds": [0, 1, 2],
            "new_prompt_tags": sorted({row["tag"] for row in inventory}),
            "generation_cells": len(inventory),
            "quality_passing_cells": len(accepted),
            "evaluation_repeats": repeats,
            "training_seeds": [seed for seed, _ in runs],
            "selection_rule": "all quality-passing single-60 cells are tracked; rejected cells count as end-to-end failures",
        },
        "generation": {
            "quality_pass_rate": quality_rate,
            "accepted_clips": accepted_clips,
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
        "tracking_across_training_seeds": {
            metric: across_values(values) for metric, values in tracking_values.items()
        },
        "end_to_end_success_across_training_seeds": across_values(end_to_end_values),
        "decision": {
            "quality_floor": quality_floor,
            "tracking_success_floor": tracking_floor,
            "tracking_completion_floor": completion_floor,
            "end_to_end_success_floor": end_to_end_floor,
            "quality_passed": quality_rate >= quality_floor,
            "all_trackers_passed": tracking_pass,
            "all_end_to_end_passed": end_to_end_pass,
            "short_horizon_demo_credible": (
                quality_rate >= quality_floor and tracking_pass and end_to_end_pass
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generation-result", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--policy", action="append", nargs=2, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    payload = json.loads(args.generation_result.read_text(encoding="utf-8"))
    if payload.get("schema") != "text2motion-generation-continuation-experiment-v1":
        raise ValueError("unexpected generation result schema")
    records = [row for row in payload["records"] if row["variant"] == "single_60"]
    if len(records) != 27:
        raise ValueError("expected 27 single-chunk records")

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
            clip = f"gseed{generation_seed}__{tag}"
            item = {
                "generation_seed": generation_seed,
                "tag": tag,
                "text": record["text"],
                "clip": clip,
                "quality_gate": record["quality_gate"],
                "gate_reason": record["gate_reason"],
                "source_qpos_sha256": record["stitched_qpos_sha256"],
                "accepted_shard": None,
                "accepted_shard_sha256": None,
            }
            if record["quality_gate"] == "passed":
                source = (
                    args.generated_root
                    / "single_60"
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
                    raise RuntimeError(f"accepted reference failed replay conversion: {clip}")
                item["accepted_shard"] = shard_name
                item["accepted_shard_sha256"] = sha256(shard)
                accepted_index += 1
            inventory.append(item)

        write(
            args.out / "inventory.json",
            {
                "schema": "text2motion-single-chunk-reference-inventory-v1",
                "source_artifact": str(args.generation_result),
                "source_sha256": sha256(args.generation_result),
                "selection_rule": "all 27 single-60 generation cells; no rerolls; all quality-passing cells are tracked",
                "records": inventory,
            },
        )
        if accepted_index != 21:
            raise RuntimeError(f"expected 21 accepted references, got {accepted_index}")

        tracker_environment = dict(environment)
        tracker_environment["PYTHONPATH"] = (
            f"{args.root}:{args.tracker_repo}:{args.root / 'projects/motion_cerebellum_remote'}"
        )
        policies = [(int(seed), Path(path)) for seed, path in args.policy]
        if [seed for seed, _ in policies] != [0, 1, 2]:
            raise ValueError("policies must be supplied in training-seed order")
        runs = []
        accepted_clips = [row["clip"] for row in inventory if row["quality_gate"] == "passed"]
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
                    f"single_chunk_multiseed_tracker{training_seed}",
                    "--ref-dir",
                    str(accepted_root),
                    "--split",
                    "all",
                    "--min-ref-frames",
                    "2",
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
