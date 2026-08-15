"""Sanitize non-speed failures after frozen space-invariant seam correction."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .omg_adapter import load_omg_motion, resample_qpos, validate_qpos_36
    from .reference_repair import (
        apply_contact_root_lock,
        contact_geometry,
        gate_reference,
        inpaint_joint_discontinuities,
    )
except ImportError:
    from omg_adapter import load_omg_motion, resample_qpos, validate_qpos_36
    from reference_repair import (
        apply_contact_root_lock,
        contact_geometry,
        gate_reference,
        inpaint_joint_discontinuities,
    )


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def ground_align_to_sole(
    qpos: np.ndarray,
    model,
    slide_setup,
) -> tuple[np.ndarray, float]:
    checked = validate_qpos_36(qpos).astype(np.float64)
    _, sole_floor = contact_geometry(model, checked, slide_setup)
    checked[:, 2] -= sole_floor
    return validate_qpos_36(checked), float(sole_floor)


def repair_for_reason(
    qpos: np.ndarray,
    *,
    reason: str,
    model,
    slide_setup,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any] | None]:
    candidate = validate_qpos_36(qpos)
    repairs = protocol["repairs"]
    if reason in {"continuity", "joint_vel"}:
        config = repairs["continuity_or_joint_vel"]
        repaired, detail = inpaint_joint_discontinuities(
            candidate,
            threshold=float(config["detection_threshold_rad_frame"]),
            half_window=int(config["half_window_frames"]),
        )
        return repaired, {"operation": config["operation"], **detail}
    if reason == "foot_slide":
        config = repairs["foot_slide"]
        contacts, floor = contact_geometry(model, candidate, slide_setup)
        repaired, correction = apply_contact_root_lock(
            candidate,
            contacts,
            floor=floor,
            contact_tolerance=float(config["contact_tolerance_m"]),
            gain=float(config["gain"]),
        )
        return repaired, {
            "operation": config["operation"],
            "root_xy_correction_max_m": float(
                np.linalg.norm(correction, axis=1).max()
            ),
        }
    if reason == "hover":
        config = repairs["hover"]
        repaired, shift = ground_align_to_sole(candidate, model, slide_setup)
        return repaired, {
            "operation": config["operation"],
            "root_z_shift_m": float(-shift),
        }
    return candidate, None


def summarize(
    records: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    passed = [row for row in records if row["quality_gate"] == "passed"]
    source_passed = [row for row in records if row["source_quality_gate"] == "passed"]
    preserved = [row for row in source_passed if row["quality_gate"] == "passed"]
    per_seed = {
        str(seed): sum(
            row["quality_gate"] == "passed"
            for row in records
            if int(row["generation_seed"]) == int(seed)
        )
        for seed in protocol["generation_seeds"]
    }
    acceptance = protocol["acceptance"]
    criteria = {
        "minimum_quality_passed": len(passed)
        >= int(acceptance["minimum_quality_passed"]),
        "minimum_per_generation_seed_quality_passed": all(
            value >= int(acceptance["minimum_per_generation_seed_quality_passed"])
            for value in per_seed.values()
        ),
        "preserve_every_source_passing_cell": (
            not acceptance["preserve_every_source_passing_cell"]
            or len(preserved) == len(source_passed)
        ),
    }
    return {
        "quality_passed": len(passed),
        "quality_rejected": len(records) - len(passed),
        "source_quality_passed": len(source_passed),
        "preserved_source_passing": len(preserved),
        "per_generation_seed_quality_passed": per_seed,
        "reason_counts": dict(
            sorted(
                Counter(
                    row["gate_reason"]
                    for row in records
                    if row["quality_gate"] != "passed"
                ).items()
            )
        ),
        "recovered_by_source_reason": dict(
            sorted(
                Counter(
                    row["source_gate_reason"]
                    for row in records
                    if row["source_quality_gate"] != "passed"
                    and row["quality_gate"] == "passed"
                ).items()
            )
        ),
        "criteria": criteria,
        "eligible_for_tracking": all(criteria.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "text2motion-long-horizon-sanitizer-protocol-v1":
        raise ValueError("unexpected sanitizer protocol schema")
    source_payload = json.loads(args.source_result.read_text(encoding="utf-8"))
    if source_payload.get("schema") != "text2motion-long-horizon-seam-experiment-v1":
        raise ValueError("unexpected source experiment schema")
    source_variant = str(protocol["source_variant"])
    source_records = [
        row for row in source_payload["records"] if row["variant"] == source_variant
    ]
    if len(source_records) != 27:
        raise ValueError("expected 27 source cells")

    tracker_root = str(args.tracker_repo.resolve())
    if tracker_root not in sys.path:
        sys.path.insert(0, tracker_root)
    import mujoco
    from motion_tracking.quality import foot_contact_setup
    from motion_tracking.robots import G1

    model = mujoco.MjModel.from_xml_path(str(G1.xml))
    slide_setup = foot_contact_setup(model, G1)
    args.out.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for source_row in source_records:
        generation_seed = int(source_row["generation_seed"])
        tag = str(source_row["tag"])
        text = str(source_row["text"])
        source = (
            args.generated_root
            / source_variant
            / f"seed_{generation_seed}"
            / tag
            / "reference_motion.npz"
        )
        motion = load_omg_motion(source)
        candidate = resample_qpos(
            motion.qpos, motion.fps, float(protocol["target_fps"])
        )
        before_ref, before_reason, before_metrics = gate_reference(
            candidate, model, slide_setup, tag, text, G1
        )
        replayed_source_result = "passed" if before_ref is not None else "rejected"
        if replayed_source_result != source_row["quality_gate"]:
            raise RuntimeError(f"{generation_seed}/{tag}: source gate replay mismatch")

        operations: list[dict[str, Any]] = []
        reason = before_reason
        if before_ref is None:
            for pass_index in range(int(protocol["maximum_repair_passes"])):
                repaired, operation = repair_for_reason(
                    candidate,
                    reason=reason,
                    model=model,
                    slide_setup=slide_setup,
                    protocol=protocol,
                )
                if operation is None:
                    break
                candidate = repaired
                operations.append({"pass": pass_index + 1, **operation})
                trial_ref, reason, _ = gate_reference(
                    candidate, model, slide_setup, tag, text, G1
                )
                if trial_ref is not None:
                    break

        after_ref, after_reason, after_metrics = gate_reference(
            candidate, model, slide_setup, tag, text, G1
        )
        output_dir = args.out / "generated" / f"seed_{generation_seed}" / tag
        output_dir.mkdir(parents=True)
        destination = output_dir / "reference_motion.npz"
        np.savez_compressed(
            destination,
            qpos_36=validate_qpos_36(candidate),
            fps=np.float32(protocol["target_fps"]),
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "schema": protocol["schema"],
                        "generation_seed": generation_seed,
                        "operations": operations,
                    },
                    sort_keys=True,
                )
            ),
        )
        records.append(
            {
                "generation_seed": generation_seed,
                "tag": tag,
                "text": text,
                "source_quality_gate": replayed_source_result,
                "source_gate_reason": before_reason,
                "quality_gate": "passed" if after_ref is not None else "rejected",
                "gate_reason": after_reason,
                "before": before_metrics,
                "after": after_metrics,
                "operations": operations,
            }
        )
        write(
            args.out / "result.json",
            {
                "schema": "text2motion-long-horizon-sanitizer-experiment-v1",
                "result": "running",
                "records": records,
            },
        )

    summary = summarize(records, protocol)
    write(
        args.out / "result.json",
        {
            "schema": "text2motion-long-horizon-sanitizer-experiment-v1",
            "result": (
                "eligible_for_tracking"
                if summary["eligible_for_tracking"]
                else "not_eligible_for_tracking"
            ),
            "summary": summary,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
