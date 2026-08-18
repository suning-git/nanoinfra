"""Attribute frozen OMG reference failures to generation or bridge stages."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .omg_adapter import G1_JOINT_NAMES, load_omg_motion, qpos_to_qvel, resample_qpos
except ImportError:
    from omg_adapter import G1_JOINT_NAMES, load_omg_motion, qpos_to_qvel, resample_qpos


ROOT_SPEED_LIMIT = 2.0
JOINT_SPEED_LIMIT = 15.0
JOINT_STEP_LIMIT_AT_50HZ = 0.5
FOOT_SLIDE_LIMIT_AT_50HZ = 12.0
HOVER_LIMIT = 50.0


def stage_kinematics(qpos: np.ndarray, fps: float) -> dict[str, Any]:
    qvel = qpos_to_qvel(qpos, fps)
    joint_steps = np.abs(np.diff(qpos[:, 7:], axis=0))
    step_flat = int(joint_steps.argmax())
    step_frame, step_joint = np.unravel_index(step_flat, joint_steps.shape)
    joint_speeds = np.abs(qvel[:, 6:])
    speed_flat = int(joint_speeds.argmax())
    speed_frame, speed_joint = np.unravel_index(speed_flat, joint_speeds.shape)
    root_speeds = np.linalg.norm(qvel[:, :3], axis=1)
    root_frame = int(root_speeds.argmax())
    return {
        "frames": int(len(qpos)),
        "fps": float(fps),
        "duration_seconds": float((len(qpos) - 1) / fps),
        "root_speed_max_m_s": float(root_speeds[root_frame]),
        "root_speed_worst_frame": root_frame,
        "joint_speed_max_rad_s": float(joint_speeds[speed_frame, speed_joint]),
        "joint_speed_worst_frame": speed_frame,
        "joint_speed_worst_joint": G1_JOINT_NAMES[speed_joint],
        "joint_step_max_rad_frame": float(joint_steps[step_frame, step_joint]),
        "joint_step_worst_transition": [int(step_frame), int(step_frame + 1)],
        "joint_step_worst_joint": G1_JOINT_NAMES[step_joint],
        "joint_step_equivalent_at_50hz": float(
            joint_steps[step_frame, step_joint] * fps / 50.0
        ),
    }


def source_rate_evidence(metrics: dict[str, Any], reason: str) -> bool:
    if reason == "speed":
        return metrics["root_speed_max_m_s"] > ROOT_SPEED_LIMIT
    if reason == "joint_vel":
        return metrics["joint_speed_max_rad_s"] > JOINT_SPEED_LIMIT
    if reason == "continuity":
        return metrics["joint_step_equivalent_at_50hz"] > JOINT_STEP_LIMIT_AT_50HZ
    return False


def diagnose_one(source: Path, tracker_repo: Path, tag: str) -> dict[str, Any]:
    motion = load_omg_motion(source)
    validated = np.asarray(motion.qpos, dtype=np.float64)
    resampled = resample_qpos(motion.qpos, motion.fps, 50.0)
    raw_metrics = stage_kinematics(validated, motion.fps)
    resampled_metrics = stage_kinematics(resampled, 50.0)

    root_text = str(tracker_repo.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import mujoco
    from motion_tracking.quality import foot_contact_setup, ref_contact_stats
    from motion_tracking.retarget import build_ref
    from motion_tracking.robots import G1

    model = mujoco.MjModel.from_xml_path(str(G1.xml))
    slide_setup = foot_contact_setup(model, G1)
    data = mujoco.MjData(model)
    raw_contact = ref_contact_stats(model, data, validated, *slide_setup)
    data = mujoco.MjData(model)
    resampled_contact = ref_contact_stats(model, data, resampled, *slide_setup)
    ref, reason = build_ref(resampled, model, tag, slide_setup, robot=G1)
    reason = "ok" if ref is not None else str(reason)

    raw_slide_equivalent = float(raw_contact["slide"] * motion.fps / 50.0)
    if source_rate_evidence(raw_metrics, reason):
        attribution = "omg_source_motion"
    elif reason == "foot_slide" and raw_slide_equivalent > FOOT_SLIDE_LIMIT_AT_50HZ:
        attribution = "omg_source_motion"
    elif reason == "hover" and float(raw_contact["hover"]) > HOVER_LIMIT:
        attribution = "omg_source_motion"
    elif reason == "ok":
        attribution = "no_failure"
    else:
        attribution = "bridge_or_gate_unresolved"

    return {
        "tag": tag,
        "source_contract": {
            "qpos_key": motion.qpos_key,
            "shape": list(validated.shape),
            "declared_fps": motion.fps,
        },
        "validation": {
            "validated_quaternion_norm_max_error": float(
                np.max(np.abs(np.linalg.norm(validated[:, 3:7], axis=1) - 1.0))
            ),
            "code_audit": "validation does not alter root translation or 29 joint channels",
        },
        "raw_omg_30hz": {
            **raw_metrics,
            "foot_slide_native_frame_units": float(raw_contact["slide"]),
            "foot_slide_equivalent_at_50hz": raw_slide_equivalent,
            "hover": float(raw_contact["hover"]),
        },
        "bridge_50hz": {
            **resampled_metrics,
            "foot_slide": float(resampled_contact["slide"]),
            "hover": float(resampled_contact["hover"]),
        },
        "upstream_gate": {
            "result": "passed" if ref is not None else "rejected",
            "reason": reason,
        },
        "attribution": attribution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-status", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompt_status = json.loads(args.prompt_status.read_text(encoding="utf-8"))
    new_rows = [row for row in prompt_status["prompts"] if row["source"] == "new_generation"]
    if len(new_rows) != 9:
        raise ValueError("expected nine frozen new-generation prompts")
    records = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema": "text2motion-generator-stage-diagnostic-v1",
                "result": "running",
                "next_tag": new_rows[0]["tag"],
                "completed_records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for row_index, row in enumerate(new_rows):
        try:
            records.append(
                diagnose_one(
                    args.generated_root / row["tag"] / "reference_motion.npz",
                    args.tracker_repo,
                    row["tag"],
                )
            )
            args.out.write_text(
                json.dumps(
                    {
                        "schema": "text2motion-generator-stage-diagnostic-v1",
                        "result": "running",
                        "next_tag": (
                            new_rows[row_index + 1]["tag"]
                            if row_index + 1 < len(new_rows)
                            else None
                        ),
                        "completed_records": records,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as error:
            args.out.write_text(
                json.dumps(
                    {
                        "schema": "text2motion-generator-stage-diagnostic-v1",
                        "result": "failed",
                        "failed_tag": row["tag"],
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "completed_records": records,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
    expected = {row["tag"]: row["quality_gate"] for row in new_rows}
    if any(record["upstream_gate"]["result"] != expected[record["tag"]] for record in records):
        raise RuntimeError("frozen quality-gate replay mismatch")
    payload = {
        "schema": "text2motion-generator-stage-diagnostic-v1",
        "selection_rule": "same nine frozen OMG outputs; no regeneration or repair",
        "bridge_contract": "OMG already emits G1 qpos_36; bridge validates and resamples 30 to 50 Hz",
        "records": records,
        "counts": {
            "passed": sum(row["upstream_gate"]["result"] == "passed" for row in records),
            "rejected": sum(row["upstream_gate"]["result"] == "rejected" for row in records),
            "attribution": dict(sorted(Counter(row["attribution"] for row in records).items())),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
