"""Repair frozen two-chunk OMG seams with one prompt-agnostic C1 residual rule."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .omg_adapter import load_omg_motion, validate_qpos_36
except ImportError:
    from omg_adapter import load_omg_motion, validate_qpos_36


LINEAR_CHANNELS = np.asarray([0, 1, 2, *range(7, 36)], dtype=np.int64)


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def normalize_quaternion(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("zero-norm quaternion")
    return array / norm


def quaternion_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def quaternion_slerp_identity(target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Interpolate from identity to target, choosing the shortest rotation."""

    quaternion = normalize_quaternion(target)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    weights = np.asarray(weight, dtype=np.float64)
    cosine = float(np.clip(quaternion[0], -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        result = np.zeros(weights.shape + (4,), dtype=np.float64)
        result[..., 0] = 1.0
        return result
    sine = np.sin(angle)
    identity_scale = np.sin((1.0 - weights) * angle) / sine
    target_scale = np.sin(weights * angle) / sine
    identity = np.zeros(weights.shape + (4,), dtype=np.float64)
    identity[..., 0] = 1.0
    return normalize_quaternion(
        identity_scale[..., None] * identity
        + target_scale[..., None] * quaternion
    )


def quaternion_yaw(value: np.ndarray) -> float:
    w, x, y, z = normalize_quaternion(value)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quaternion(angle: float) -> np.ndarray:
    return np.asarray(
        [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
        dtype=np.float64,
    )


def quintic_position_envelope(t: np.ndarray) -> np.ndarray:
    values = np.asarray(t, dtype=np.float64)
    return 1.0 - 10.0 * values**3 + 15.0 * values**4 - 6.0 * values**5


def quintic_start_derivative_envelope(t: np.ndarray) -> np.ndarray:
    values = np.asarray(t, dtype=np.float64)
    return values - 6.0 * values**3 + 8.0 * values**4 - 3.0 * values**5


def c1_residual_stitch(
    qpos: np.ndarray,
    *,
    boundary_frame: int,
    decay_frames: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Align chunk two to chunk one, then smoothly return to the original plan."""

    source = validate_qpos_36(qpos).astype(np.float64)
    if boundary_frame < 2 or boundary_frame + 2 > len(source):
        raise ValueError("boundary must leave two frames on each side")
    if decay_frames < 2 or decay_frames >= len(source) - boundary_frame:
        raise ValueError("decay horizon must fit inside the second chunk")

    result = source.copy()
    first = source[:boundary_frame]
    second = source[boundary_frame:]
    position_residual = first[-1, LINEAR_CHANNELS] - second[0, LINEAR_CHANNELS]
    velocity_residual = (
        first[-1, LINEAR_CHANNELS]
        - first[-2, LINEAR_CHANNELS]
        - (second[1, LINEAR_CHANNELS] - second[0, LINEAR_CHANNELS])
    )
    indices = np.arange(decay_frames + 1, dtype=np.float64)
    phase = indices / float(decay_frames)
    position_weight = quintic_position_envelope(phase)
    derivative_weight = quintic_start_derivative_envelope(phase)
    correction = (
        position_weight[:, None] * position_residual[None, :]
        + derivative_weight[:, None]
        * (float(decay_frames) * velocity_residual)[None, :]
    )
    result[
        boundary_frame : boundary_frame + decay_frames + 1, LINEAR_CHANNELS
    ] += correction

    first_quaternion = normalize_quaternion(first[-1, 3:7])
    second_quaternion = normalize_quaternion(second[0, 3:7])
    orientation_residual = quaternion_multiply(
        first_quaternion, quaternion_conjugate(second_quaternion)
    )
    orientation_correction = quaternion_slerp_identity(
        orientation_residual, position_weight
    )
    original_orientation = normalize_quaternion(
        second[: decay_frames + 1, 3:7]
    )
    result[boundary_frame : boundary_frame + decay_frames + 1, 3:7] = (
        quaternion_multiply(orientation_correction, original_orientation)
    )
    result = validate_qpos_36(result).astype(np.float32)

    changed = result.astype(np.float64) - source
    linear_change = changed[:, LINEAR_CHANNELS]
    return result, {
        "linear_correction_rms": float(np.sqrt(np.mean(linear_change**2))),
        "linear_correction_max_abs": float(np.abs(linear_change).max()),
        "start_joint_pose_residual_max_abs": float(
            np.abs(position_residual[3:]).max()
        ),
        "start_joint_velocity_residual_max_abs": float(
            np.abs(velocity_residual[3:]).max()
        ),
    }


def planar_space_aligned_selective_c1_stitch(
    qpos: np.ndarray,
    *,
    boundary_frame: int,
    decay_frames: int,
    maximum_joint_velocity_residual: float = 0.1,
    maximum_root_z_velocity_residual: float = 0.03,
) -> tuple[np.ndarray, dict[str, float]]:
    """Use planar symmetry for global pose and C1 decay only on unsafe channels."""

    source = validate_qpos_36(qpos).astype(np.float64)
    if boundary_frame < 2 or boundary_frame + 2 > len(source):
        raise ValueError("boundary must leave two frames on each side")
    if decay_frames < 2 or decay_frames >= len(source) - boundary_frame:
        raise ValueError("decay horizon must fit inside the second chunk")

    result = source.copy()
    first = source[:boundary_frame]
    second = source[boundary_frame:]
    yaw_delta = quaternion_yaw(first[-1, 3:7]) - quaternion_yaw(second[0, 3:7])
    cosine, sine = np.cos(yaw_delta), np.sin(yaw_delta)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    relative_xy = second[:, :2] - second[0, :2]
    result[boundary_frame:, :2] = first[-1, :2] + relative_xy @ rotation.T
    yaw_rotation = yaw_quaternion(yaw_delta)
    result[boundary_frame:, 3:7] = quaternion_multiply(
        yaw_rotation, normalize_quaternion(second[:, 3:7])
    )

    selected_columns = np.asarray([2, *range(7, 36)], dtype=np.int64)
    position_residual = first[-1, selected_columns] - second[0, selected_columns]
    velocity_residual = (
        first[-1, selected_columns]
        - first[-2, selected_columns]
        - (second[1, selected_columns] - second[0, selected_columns])
    )
    apply_mask = np.zeros(len(selected_columns), dtype=bool)
    apply_mask[0] = abs(position_residual[0]) > 0.05
    apply_mask[1:] = (np.abs(position_residual[1:]) > 0.5) | (
        np.abs(velocity_residual[1:]) > 0.15
    )
    clipped_velocity = velocity_residual.copy()
    clipped_velocity[0] = np.clip(
        clipped_velocity[0],
        -maximum_root_z_velocity_residual,
        maximum_root_z_velocity_residual,
    )
    clipped_velocity[1:] = np.clip(
        clipped_velocity[1:],
        -maximum_joint_velocity_residual,
        maximum_joint_velocity_residual,
    )
    indices = np.arange(decay_frames + 1, dtype=np.float64)
    phase = indices / float(decay_frames)
    position_weight = quintic_position_envelope(phase)
    derivative_weight = quintic_start_derivative_envelope(phase)
    correction = (
        position_weight[:, None] * position_residual[None, :]
        + derivative_weight[:, None]
        * (float(decay_frames) * clipped_velocity)[None, :]
    )
    correction[:, ~apply_mask] = 0.0
    result[
        boundary_frame : boundary_frame + decay_frames + 1, selected_columns
    ] += correction
    result = validate_qpos_36(result).astype(np.float32)

    intrinsic_change = (
        result[:, selected_columns].astype(np.float64) - source[:, selected_columns]
    )
    original_relative = np.diff(second[:, :2], axis=0)
    repaired_relative = np.diff(result[boundary_frame:, :2], axis=0)
    return result, {
        "linear_correction_rms": float(np.sqrt(np.mean(intrinsic_change**2))),
        "linear_correction_max_abs": float(np.abs(intrinsic_change).max()),
        "selected_joint_channels": int(apply_mask[1:].sum()),
        "root_z_selected": bool(apply_mask[0]),
        "yaw_alignment_rad": float(yaw_delta),
        "planar_step_norm_max_error": float(
            np.abs(
                np.linalg.norm(original_relative, axis=1)
                - np.linalg.norm(repaired_relative, axis=1)
            ).max()
        ),
    }


def candidate_summary(
    records: list[dict[str, Any]],
    *,
    variant: str,
    development_seeds: set[int],
    holdout_seeds: set[int],
    baseline_passed: set[tuple[int, str]],
) -> dict[str, Any]:
    rows = [row for row in records if row["variant"] == variant]
    passed = [row for row in rows if row.get("quality_gate") == "passed"]
    passed_keys = {(int(row["generation_seed"]), str(row["tag"])) for row in passed}
    development = [row for row in rows if int(row["generation_seed"]) in development_seeds]
    holdout = [row for row in rows if int(row["generation_seed"]) in holdout_seeds]
    return {
        "attempted": len(rows),
        "quality_passed": len(passed),
        "quality_rejected": len(rows) - len(passed),
        "development_quality_passed": sum(
            row.get("quality_gate") == "passed" for row in development
        ),
        "holdout_quality_passed": sum(
            row.get("quality_gate") == "passed" for row in holdout
        ),
        "per_generation_seed_quality_passed": {
            str(seed): sum(
                row.get("quality_gate") == "passed"
                for row in rows
                if int(row["generation_seed"]) == seed
            )
            for seed in sorted(development_seeds | holdout_seeds)
        },
        "reason_counts": dict(
            sorted(
                Counter(
                    row.get("gate_reason", "not_run")
                    for row in rows
                    if row.get("quality_gate") != "passed"
                ).items()
            )
        ),
        "preserved_baseline_passing": len(baseline_passed & passed_keys),
        "baseline_passing_total": len(baseline_passed),
        "correction_rms_mean": float(
            np.mean([row["correction"]["linear_correction_rms"] for row in rows])
        ),
    }


def select_candidate(
    summaries: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    acceptance = protocol["acceptance"]
    eligible: dict[str, bool] = {}
    for name, summary in summaries.items():
        eligible[name] = bool(
            summary["development_quality_passed"]
            >= int(acceptance["minimum_development_quality_passed"])
            and summary["holdout_quality_passed"]
            >= int(acceptance["minimum_holdout_quality_passed"])
            and summary["quality_passed"]
            >= int(acceptance["minimum_overall_quality_passed"])
            and (
                not acceptance["preserve_every_baseline_passing_cell"]
                or summary["preserved_baseline_passing"]
                == summary["baseline_passing_total"]
            )
        )
    ranked = sorted(
        summaries,
        key=lambda name: (
            -int(summaries[name]["development_quality_passed"]),
            float(summaries[name]["correction_rms_mean"]),
            int(protocol["candidates"][name]["decay_frames"]),
            name,
        ),
    )
    selected = next((name for name in ranked if eligible[name]), None)
    return selected, {"eligible": eligible, "development_ranking": ranked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--probe-script", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") not in {
        "text2motion-long-horizon-seam-protocol-v1",
        "text2motion-long-horizon-seam-protocol-v2",
    }:
        raise ValueError("unexpected seam protocol schema")
    source_payload = json.loads(args.source_result.read_text(encoding="utf-8"))
    if source_payload.get("schema") != "text2motion-generation-continuation-experiment-v1":
        raise ValueError("unexpected source-result schema")
    source_variant = str(protocol["source_variant"])
    source_records = [
        row for row in source_payload["records"] if row["variant"] == source_variant
    ]
    expected_seeds = [int(value) for value in protocol["source_generation_seeds"]]
    prompt_tags = sorted({str(row["tag"]) for row in source_records})
    if len(source_records) != len(expected_seeds) * len(prompt_tags):
        raise ValueError("source grid is incomplete")
    if sorted({int(row["generation_seed"]) for row in source_records}) != expected_seeds:
        raise ValueError("source generation seeds do not match protocol")
    baseline_passed = {
        (int(row["generation_seed"]), str(row["tag"]))
        for row in source_records
        if row.get("quality_gate") == "passed"
    }

    args.out.mkdir(parents=True)
    probe_environment = os.environ.copy()
    probe_environment.pop("MUJOCO_GL", None)
    records: list[dict[str, Any]] = []
    for variant, candidate in protocol["candidates"].items():
        for source_row in source_records:
            generation_seed = int(source_row["generation_seed"])
            tag = str(source_row["tag"])
            source = (
                args.generated_root
                / source_variant
                / f"seed_{generation_seed}"
                / tag
                / "reference_motion.npz"
            )
            row: dict[str, Any] = {
                "variant": variant,
                "generation_seed": generation_seed,
                "tag": tag,
                "text": source_row["text"],
                "source_quality_gate": source_row.get("quality_gate"),
                "source_gate_reason": source_row.get("gate_reason"),
            }
            try:
                motion = load_omg_motion(source)
                if not np.isclose(motion.fps, float(protocol["source_fps"])):
                    raise ValueError(f"unexpected source fps: {motion.fps}")
                method = protocol["correction"].get("method", "c1_residual_decay")
                if method == "c1_residual_decay":
                    repaired, correction = c1_residual_stitch(
                        motion.qpos,
                        boundary_frame=int(protocol["boundary_frame"]),
                        decay_frames=int(candidate["decay_frames"]),
                    )
                elif method == "planar_space_aligned_selective_c1":
                    repaired, correction = planar_space_aligned_selective_c1_stitch(
                        motion.qpos,
                        boundary_frame=int(protocol["boundary_frame"]),
                        decay_frames=int(candidate["decay_frames"]),
                        maximum_joint_velocity_residual=float(
                            protocol["correction"][
                                "maximum_joint_velocity_residual_rad_frame"
                            ]
                        ),
                        maximum_root_z_velocity_residual=float(
                            protocol["correction"][
                                "maximum_root_z_velocity_residual_m_frame"
                            ]
                        ),
                    )
                else:
                    raise ValueError(f"unknown correction method: {method}")
                output_dir = args.out / "generated" / variant / f"seed_{generation_seed}" / tag
                output_dir.mkdir(parents=True)
                reference = output_dir / "reference_motion.npz"
                np.savez_compressed(
                    reference,
                    qpos_36=repaired,
                    fps=np.float32(motion.fps),
                    metadata_json=np.asarray(
                        json.dumps(
                            {
                                "schema": protocol["schema"],
                                "variant": variant,
                                "generation_seed": generation_seed,
                            },
                            sort_keys=True,
                        )
                    ),
                )
                probe_out = args.out / "probes" / variant / f"seed_{generation_seed}" / f"{tag}.json"
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
                    raise RuntimeError(f"quality probe failed: {variant}/{generation_seed}/{tag}")
                probe = json.loads(probe_out.read_text(encoding="utf-8"))["record"]
                row.update(
                    {
                        "result": "passed",
                        "correction": correction,
                        "quality_gate": probe["upstream_gate"]["result"],
                        "gate_reason": probe["upstream_gate"]["reason"],
                        "raw_omg_30hz": probe["raw_omg_30hz"],
                        "bridge_50hz": probe["bridge_50hz"],
                    }
                )
            except Exception as error:
                row.update(
                    {
                        "result": "failed",
                        "quality_gate": "not_run",
                        "gate_reason": "experiment_error",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
            records.append(row)
            write(
                args.out / "result.json",
                {
                    "schema": "text2motion-long-horizon-seam-experiment-v1",
                    "result": "running",
                    "records": records,
                },
            )

    development_seeds = {int(value) for value in protocol["development_generation_seeds"]}
    holdout_seeds = {int(value) for value in protocol["holdout_generation_seeds"]}
    summaries = {
        variant: candidate_summary(
            records,
            variant=variant,
            development_seeds=development_seeds,
            holdout_seeds=holdout_seeds,
            baseline_passed=baseline_passed,
        )
        for variant in protocol["candidates"]
    }
    selected, decision = select_candidate(summaries, protocol)
    write(
        args.out / "result.json",
        {
            "schema": "text2motion-long-horizon-seam-experiment-v1",
            "result": (
                "candidate_selected"
                if selected is not None
                else "no_candidate_met_preregistered_floors"
            ),
            "prompt_tags": prompt_tags,
            "source_baseline_quality_passed": len(baseline_passed),
            "summaries": summaries,
            "decision": {**decision, "selected_variant": selected},
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
