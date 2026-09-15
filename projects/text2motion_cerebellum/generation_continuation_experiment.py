"""Evaluate OMG multi-chunk continuation without modifying the upstream checkout."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = {
    "single_60": {"chunks": 1, "overlap": 0, "continuation_steps": 0},
    "baseline_two_chunks": {"chunks": 2, "overlap": 0, "continuation_steps": 0},
    "continuation_overlap1_full": {"chunks": 2, "overlap": 1, "continuation_steps": 50},
    "continuation_overlap10_last10": {"chunks": 2, "overlap": 10, "continuation_steps": 10},
    "continuation_overlap10_full": {"chunks": 2, "overlap": 10, "continuation_steps": 50},
}


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stitch_arrays(first: np.ndarray, second: np.ndarray, overlap: int) -> np.ndarray:
    first_array = np.asarray(first)
    second_array = np.asarray(second)
    if first_array.ndim != 2 or second_array.ndim != 2:
        raise ValueError("plans must be rank-two arrays")
    if first_array.shape[1] != second_array.shape[1]:
        raise ValueError("plan feature dimensions do not match")
    if overlap < 0 or overlap >= len(second_array):
        raise ValueError("overlap must be non-negative and shorter than the second plan")
    return np.concatenate([first_array, second_array[overlap:]], axis=0)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in records}):
        rows = [row for row in records if row["variant"] == variant]
        rejected = [row for row in rows if row.get("quality_gate") == "rejected"]
        seam_steps = [
            row["raw_omg_30hz"]["joint_step_max_rad_frame"]
            for row in rows
            if row.get("raw_omg_30hz", {}).get("joint_step_worst_transition") == [59, 60]
        ]
        summaries[variant] = {
            "attempted": len(rows),
            "generated": sum(row["generation"] == "passed" for row in rows),
            "quality_passed": sum(row.get("quality_gate") == "passed" for row in rows),
            "quality_rejected": len(rejected),
            "reason_counts": dict(sorted(Counter(row["gate_reason"] for row in rejected).items())),
            "seam_is_worst_transition_count": len(seam_steps),
            "seam_worst_step_mean_rad_frame": (
                float(np.mean(seam_steps)) if seam_steps else None
            ),
        }
    return summaries


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-protocol", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--seed-motion", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--probe-script", type=Path, required=True)
    parser.add_argument("--generation-seeds", default="0,1,2")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--tags", default="")
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    requested_variants = parse_csv(args.variants)
    unknown = sorted(set(requested_variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    generation_seeds = [int(value) for value in parse_csv(args.generation_seeds)]
    if len(generation_seeds) != len(set(generation_seeds)):
        raise ValueError("generation seeds must be unique")

    protocol = json.loads(args.prompt_protocol.read_text(encoding="utf-8"))
    prompts = [row for row in protocol["prompts"] if row["source"] == "new_generation"]
    requested_tags = set(parse_csv(args.tags))
    if requested_tags:
        prompts = [row for row in prompts if row["tag"] in requested_tags]
        missing = requested_tags - {row["tag"] for row in prompts}
        if missing:
            raise ValueError(f"unknown or non-new prompt tags: {sorted(missing)}")
    if not prompts:
        raise ValueError("no prompts selected")

    from omg.pipeline import MotionPlan, OnnxDiffusionPlanner, save_motion_plan
    from omg.tracking.holomotion.io import load_reference_motion

    seed_reference = load_reference_motion(args.seed_motion)
    fps = float(seed_reference.fps)
    seed_qpos = np.asarray(seed_reference.qpos_36, dtype=np.float32)
    planner = OnnxDiffusionPlanner(
        args.onnx,
        providers="CUDAExecutionProvider",
        torch_device="cuda",
        seed=0,
        compile_history_encoder=False,
    )
    if planner.sequence_length != 60 or planner.num_prev_states != 10:
        raise RuntimeError(
            f"frozen protocol requires sequence_length=60 and history=10, got "
            f"{planner.sequence_length} and {planner.num_prev_states}"
        )
    sampling_steps = int(planner.sample_timestep_map.shape[0])
    if sampling_steps != 50:
        raise RuntimeError(f"frozen protocol requires 50 sampling steps, got {sampling_steps}")

    probe_environment = os.environ.copy()
    probe_environment.pop("MUJOCO_GL", None)
    records: list[dict[str, Any]] = []
    args.out.mkdir(parents=True)
    for generation_seed in generation_seeds:
        for prompt in prompts:
            tag = prompt["tag"]
            text = prompt["text"]
            planner.rng = np.random.default_rng(generation_seed)
            first = planner.plan(
                seed_qpos_36=seed_qpos,
                text=text,
                fps=fps,
                num_frames=planner.sequence_length,
            )
            state_after_first = copy.deepcopy(planner.rng.bit_generator.state)
            current_seed_qpos = np.concatenate(
                [seed_qpos, first.qpos_36], axis=0
            )[-planner.num_prev_states :]

            for variant_name in requested_variants:
                config = VARIANTS[variant_name]
                planner.rng.bit_generator.state = copy.deepcopy(state_after_first)
                row: dict[str, Any] = {
                    "variant": variant_name,
                    "generation_seed": generation_seed,
                    "tag": tag,
                    "text": text,
                }
                try:
                    if config["chunks"] == 1:
                        stitched_qpos = first.qpos_36
                        stitched_features = first.motion_features
                    else:
                        overlap = int(config["overlap"])
                        continuation_steps = int(config["continuation_steps"])
                        second = planner.plan(
                            seed_qpos_36=current_seed_qpos,
                            text=text,
                            fps=fps,
                            num_frames=planner.sequence_length,
                            previous_plan=first if continuation_steps else None,
                            previous_plan_cursor_frames=(
                                planner.sequence_length - overlap if continuation_steps else 0
                            ),
                            continuation_steps=continuation_steps,
                        )
                        stitched_qpos = stitch_arrays(first.qpos_36, second.qpos_36, overlap)
                        stitched_features = stitch_arrays(
                            first.motion_features, second.motion_features, overlap
                        )

                    output_dir = (
                        args.out
                        / "generated"
                        / variant_name
                        / f"seed_{generation_seed}"
                        / tag
                    )
                    save_motion_plan(
                        MotionPlan(
                            qpos_36=np.asarray(stitched_qpos, dtype=np.float32),
                            motion_features=np.asarray(stitched_features, dtype=np.float32),
                            fps=fps,
                            metadata={
                                "experiment": "text2motion_generation_continuation_v1",
                                "variant": variant_name,
                                "generation_seed": generation_seed,
                                **config,
                            },
                        ),
                        output_dir,
                    )
                    reference = output_dir / "reference_motion.npz"
                    row.update(
                        {
                            "generation": "passed",
                            "frames": int(len(stitched_qpos)),
                        }
                    )
                    probe_out = args.out / "probes" / variant_name / f"seed_{generation_seed}" / f"{tag}.json"
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(args.probe_script),
                            "--source",
                            str(reference),
                            "--tracker-repo",
                            str(args.tracker_repo),
                            "--tag",
                            tag,
                            "--out",
                            str(probe_out),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=probe_environment,
                        check=False,
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(f"quality probe failed for {variant_name}/{generation_seed}/{tag}")
                    probe = json.loads(probe_out.read_text(encoding="utf-8"))["record"]
                    row.update(
                        {
                            "quality_gate": probe["upstream_gate"]["result"],
                            "gate_reason": probe["upstream_gate"]["reason"],
                            "raw_omg_30hz": probe["raw_omg_30hz"],
                            "bridge_50hz": probe["bridge_50hz"],
                        }
                    )
                except Exception as error:
                    row.update(
                        {
                            "generation": "failed",
                            "quality_gate": "not_run",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                records.append(row)
                write(
                    args.out / "result.json",
                    {
                        "schema": "text2motion-generation-continuation-experiment-v1",
                        "result": "running",
                        "selection_rule": "all requested prompt, generation-seed, and variant cells; no rerolls",
                        "records": records,
                    },
                )

    write(
        args.out / "result.json",
        {
            "schema": "text2motion-generation-continuation-experiment-v1",
            "result": (
                "passed"
                if all(row["generation"] == "passed" for row in records)
                else "failed_cells_present"
            ),
            "selection_rule": "all requested prompt, generation-seed, and variant cells; no rerolls",
            "generation_seeds": generation_seeds,
            "prompt_tags": [row["tag"] for row in prompts],
            "variants": {name: VARIANTS[name] for name in requested_variants},
            "summaries": summarize(records),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
