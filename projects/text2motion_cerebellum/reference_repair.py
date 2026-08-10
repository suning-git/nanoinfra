"""Repair the frozen OMG references without regeneration or prompt-specific tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .omg_adapter import (
        load_omg_motion,
        qpos_to_qvel,
        resample_qpos,
        validate_qpos_36,
        write_ref_shard,
    )
except ImportError:
    from omg_adapter import (
        load_omg_motion,
        qpos_to_qvel,
        resample_qpos,
        validate_qpos_36,
        write_ref_shard,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def kinematic_maxima(qpos: np.ndarray, fps: float = 50.0) -> dict[str, float]:
    checked = validate_qpos_36(qpos)
    qvel = qpos_to_qvel(checked, fps)
    return {
        "root_speed_max": float(np.linalg.norm(qvel[:, :3], axis=1).max()),
        "joint_speed_max": float(np.abs(qvel[:, 6:]).max()),
        "joint_step_max": float(np.abs(np.diff(checked[:, 7:], axis=0)).max()),
    }


def required_time_scale(qpos: np.ndarray, config: dict[str, Any]) -> float:
    maxima = kinematic_maxima(qpos)
    return max(
        1.0,
        maxima["root_speed_max"] / float(config["target_root_speed_m_s"]),
        maxima["joint_speed_max"] / float(config["target_joint_speed_rad_s"]),
        maxima["joint_step_max"] / float(config["target_joint_step_rad_frame"]),
    )


def stretch_qpos(qpos: np.ndarray, scale: float, fps: float = 50.0) -> np.ndarray:
    if scale < 1.0:
        raise ValueError("time scale cannot be below one")
    checked = validate_qpos_36(qpos)
    if np.isclose(scale, 1.0):
        return checked.copy()
    return resample_qpos(checked, fps / float(scale), fps)


def smooth_linear_channels(
    qpos: np.ndarray,
    half_width: int,
    *,
    include_root_translation: bool = True,
) -> np.ndarray:
    if half_width < 0:
        raise ValueError("smoothing half width cannot be negative")
    checked = validate_qpos_36(qpos).astype(np.float64)
    if half_width == 0:
        return checked.astype(np.float32)
    rising = np.arange(1, half_width + 2, dtype=np.float64)
    kernel = np.concatenate((rising, rising[-2::-1]))
    kernel /= kernel.sum()
    columns = (*((0, 1, 2) if include_root_translation else ()), *range(7, 36))
    padded = np.pad(checked[:, columns], ((half_width, half_width), (0, 0)), mode="edge")
    for output_column, source_column in enumerate(columns):
        checked[:, source_column] = np.convolve(
            padded[:, output_column], kernel, mode="valid"
        )
    return validate_qpos_36(checked)


def inpaint_joint_discontinuities(
    qpos: np.ndarray,
    *,
    threshold: float,
    half_window: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace joint-step outliers with fixed-window smoothstep transitions."""

    if threshold <= 0 or half_window < 1:
        raise ValueError("invalid discontinuity inpaint parameters")
    source = validate_qpos_36(qpos).astype(np.float64)
    repaired = source.copy()
    event_count = 0
    spans = []
    for joint in range(29):
        events = (np.flatnonzero(np.abs(np.diff(source[:, 7 + joint])) > threshold) + 1).tolist()
        event_count += len(events)
        intervals = []
        for event in events:
            start = max(1, event - half_window)
            end = min(len(source) - 2, event + half_window)
            if intervals and start <= intervals[-1][1] + 1:
                intervals[-1][1] = max(intervals[-1][1], end)
            else:
                intervals.append([start, end])
        for start, end in intervals:
            left, right = start - 1, end + 1
            fraction = np.linspace(0.0, 1.0, right - left + 1)
            smoothstep = fraction * fraction * (3.0 - 2.0 * fraction)
            repaired[left : right + 1, 7 + joint] = (
                source[left, 7 + joint]
                + smoothstep * (source[right, 7 + joint] - source[left, 7 + joint])
            )
            spans.append({"joint_index": joint, "left": left, "right": right})
    return validate_qpos_36(repaired), {"events": event_count, "spans": spans}


def apply_contact_root_lock(
    qpos: np.ndarray,
    contact_xyz: np.ndarray,
    *,
    floor: float,
    contact_tolerance: float,
    gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate root XY so contact geoms remain fixed during planted transitions."""

    checked = validate_qpos_36(qpos).astype(np.float64)
    contacts = np.asarray(contact_xyz, dtype=np.float64)
    if contacts.ndim != 3 or contacts.shape[0] != len(checked) or contacts.shape[2] != 3:
        raise ValueError("contact positions must have shape [T, contact_geoms, 3]")
    if not 0.0 <= gain <= 1.0:
        raise ValueError("foot-lock gain must be in [0, 1]")
    correction = np.zeros((len(checked), 2), dtype=np.float64)
    for frame in range(len(checked) - 1):
        touching = (contacts[frame, :, 2] - floor) < contact_tolerance
        delta = np.zeros(2, dtype=np.float64)
        if touching.any():
            displacement = contacts[frame + 1, touching, :2] - contacts[frame, touching, :2]
            delta = -gain * displacement.mean(axis=0)
        correction[frame + 1] = correction[frame] + delta
    checked[:, :2] += correction
    return validate_qpos_36(checked), correction.astype(np.float32)


def contact_geometry(model, qpos: np.ndarray, slide_setup) -> tuple[np.ndarray, float]:
    import mujoco

    fids, foot_geom_z, cgeoms = slide_setup
    data = mujoco.MjData(model)
    sole = np.zeros((len(qpos), len(fids)), dtype=np.float64)
    contact = np.zeros((len(qpos), len(cgeoms), 3), dtype=np.float64)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        sole[frame] = data.xpos[fids][:, 2] + foot_geom_z
        contact[frame] = data.geom_xpos[cgeoms]
    return contact, float(np.percentile(sole.min(axis=1), 2))


def gate_reference(qpos, model, slide_setup, clip: str, caption: str, robot):
    import mujoco
    from motion_tracking.quality import ref_contact_stats
    from motion_tracking.retarget import build_ref

    checked = validate_qpos_36(qpos)
    data = mujoco.MjData(model)
    contact_stats = ref_contact_stats(model, data, checked, *slide_setup)
    ref, reason = build_ref(checked, model, clip, slide_setup, robot=robot)
    if ref is not None:
        ref = dict(ref)
        ref["caption"] = caption
    return ref, str(reason), {
        "kinematics": kinematic_maxima(checked),
        "foot_slide": float(contact_stats["slide"]),
        "hover": float(contact_stats["hover"]),
        "frames": int(len(checked)),
        "duration_seconds": float((len(checked) - 1) / 50.0),
    }


def repair_rejected(
    qpos: np.ndarray,
    *,
    reason: str,
    model,
    slide_setup,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate = validate_qpos_36(qpos)
    operations: list[dict[str, Any]] = []
    discontinuity_config = protocol.get("discontinuity_inpaint")
    if reason == "continuity" and discontinuity_config:
        candidate, detail = inpaint_joint_discontinuities(
            candidate,
            threshold=float(discontinuity_config["detection_threshold_rad"]),
            half_window=int(discontinuity_config["half_window_frames"]),
        )
        operations.append({"operation": "discontinuity_inpaint", **detail})
    foot_config = protocol["foot_lock"]
    if reason == foot_config["enabled_for_gate_reason"]:
        contacts, floor = contact_geometry(model, candidate, slide_setup)
        candidate, correction = apply_contact_root_lock(
            candidate,
            contacts,
            floor=floor,
            contact_tolerance=float(foot_config["contact_tolerance_m"]),
            gain=float(foot_config["gain"]),
        )
        operations.append({
            "operation": "contact_root_lock",
            "root_xy_correction_max_m": float(np.linalg.norm(correction, axis=1).max()),
        })

    stretch_config = protocol["time_stretch"]
    maximum_scale = float(stretch_config["maximum_scale"])
    accumulated_scale = 1.0
    for pass_index in range(int(stretch_config["passes"])):
        requested = required_time_scale(candidate, stretch_config)
        available = maximum_scale / accumulated_scale
        applied = min(max(1.0, requested), available)
        if applied > 1.0 + 1e-6:
            candidate = stretch_qpos(candidate, applied)
            accumulated_scale *= applied
            operations.append({
                "operation": "time_stretch",
                "pass": pass_index + 1,
                "requested_scale": float(requested),
                "applied_scale": float(applied),
            })
        if pass_index == 0:
            half_width = int(protocol["smoothing"]["triangular_half_width_frames"])
            include_root = bool(
                protocol["smoothing"].get("include_root_translation", True)
            )
            candidate = smooth_linear_channels(
                candidate,
                half_width,
                include_root_translation=include_root,
            )
            operations.append({
                "operation": "triangular_smoothing",
                "half_width_frames": half_width,
                "include_root_translation": include_root,
            })
    return candidate, {
        "applied": True,
        "initial_gate_reason": reason,
        "total_time_scale": float(accumulated_scale),
        "operations": operations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prompt-protocol", type=Path, required=True)
    parser.add_argument("--baseline-status", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--existing-refs", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_json(args.protocol)
    protocol_schema = protocol.get("schema")
    if protocol_schema not in {
        "text2motion-reference-repair-protocol-v1",
        "text2motion-reference-repair-protocol-v2",
    }:
        raise ValueError("unexpected repair protocol schema")
    prompt_protocol = load_json(args.prompt_protocol)
    baseline = load_json(args.baseline_status)
    if baseline.get("schema") != "text2motion-expanded-prompt-generation-v1":
        raise ValueError("unexpected baseline prompt status schema")
    prompts = prompt_protocol.get("prompts", [])
    if len(prompts) != 12 or sum(row["source"] == "new_generation" for row in prompts) != 9:
        raise ValueError("expected the frozen 12-prompt protocol with nine new motions")
    baseline_by_tag = {row["tag"]: row for row in baseline["prompts"]}
    if set(baseline_by_tag) != {row["tag"] for row in prompts}:
        raise ValueError("baseline status does not match the prompt protocol")

    tracker_root = args.tracker_repo.resolve()
    sys.path.insert(0, str(tracker_root))
    import mujoco
    from motion_tracking.quality import foot_contact_setup
    from motion_tracking.robots import G1

    model = mujoco.MjModel.from_xml_path(str(G1.xml))
    slide_setup = foot_contact_setup(model, G1)
    accepted_root = args.out / "accepted_refs"
    repaired_root = args.out / "repaired"
    if (args.out / "prompt_status.json").exists() or accepted_root.exists() or repaired_root.exists():
        raise FileExistsError("reference-repair output already exists")
    accepted_root.mkdir(parents=True)
    repaired_root.mkdir(parents=True)

    existing = {
        "walk_forward": args.existing_refs / "shard_000.npz",
        "walk_turn_left": args.existing_refs / "shard_001.npz",
        "walk_turn_right": args.existing_refs / "shard_002.npz",
    }
    records = []
    accepted_index = 0
    for item in prompts:
        tag = str(item["tag"])
        caption = str(item["text"])
        baseline_row = baseline_by_tag[tag]
        if item["source"] == "frozen_existing":
            source = existing[tag]
            expected_sha = baseline_row.get("reference_sha256")
            if expected_sha and sha256(source) != expected_sha:
                raise RuntimeError(f"{tag}: frozen existing reference hash mismatch")
            shard_name = f"shard_{accepted_index:03d}.npz"
            destination = accepted_root / shard_name
            shutil.copy2(source, destination)
            records.append({
                **item,
                "generation": "inherited",
                "quality_gate": "passed",
                "accepted_shard": shard_name,
                "reference_sha256": sha256(destination),
                "repair": {"applied": False, "reason": "frozen_existing"},
            })
            accepted_index += 1
            continue

        source = args.generated_root / tag / "reference_motion.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        motion = load_omg_motion(source)
        original = resample_qpos(motion.qpos, motion.fps, 50.0)
        before_ref, before_reason, before = gate_reference(
            original, model, slide_setup, tag, caption, G1
        )
        baseline_passed = before_ref is not None
        if baseline_passed != (baseline_row["quality_gate"] == "passed"):
            raise RuntimeError(f"{tag}: baseline quality-gate replay mismatch")

        if baseline_passed:
            repaired = original.copy()
            repair = {"applied": False, "reason": "already_passed"}
        else:
            repaired, repair = repair_rejected(
                original,
                reason=before_reason,
                model=model,
                slide_setup=slide_setup,
                protocol=protocol,
            )
        repaired_dir = repaired_root / tag
        repaired_dir.mkdir()
        repaired_path = repaired_dir / "reference_motion.npz"
        np.savez_compressed(repaired_path, qpos_36=repaired, fps=np.float32(50.0))
        after_ref, after_reason, after = gate_reference(
            repaired, model, slide_setup, tag, caption, G1
        )
        shard_name = None
        reference_sha = None
        if after_ref is not None:
            shard_name = f"shard_{accepted_index:03d}.npz"
            destination = accepted_root / shard_name
            write_ref_shard(destination, (after_ref,))
            reference_sha = sha256(destination)
            accepted_index += 1
        records.append({
            **item,
            "generation": "passed",
            "quality_gate": "passed" if after_ref is not None else "rejected",
            "accepted_shard": shard_name,
            "reference_sha256": reference_sha,
            "source_sha256": sha256(source),
            "repaired_motion_sha256": sha256(repaired_path),
            "before": {"gate_reason": before_reason, **before},
            "after": {"gate_reason": after_reason, **after},
            "repair": repair,
        })

    payload = {
        "schema": "text2motion-expanded-prompt-generation-v1",
        "repair_schema": protocol_schema.replace("-protocol-", "-run-"),
        "repair_protocol_sha256": sha256(args.protocol),
        "prompt_protocol_sha256": sha256(args.prompt_protocol),
        "selection_rule": protocol["selection_rule"],
        "accepted_count": accepted_index,
        "prompts": records,
    }
    (args.out / "prompt_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
