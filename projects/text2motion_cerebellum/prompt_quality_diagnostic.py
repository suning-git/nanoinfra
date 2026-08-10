"""Re-run the upstream reference gate and record structured rejection reasons."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .omg_adapter import load_omg_motion, qpos_to_qvel, resample_qpos
except ImportError:
    from omg_adapter import load_omg_motion, qpos_to_qvel, resample_qpos


def diagnose(source: Path, tracker_repo: Path, tag: str) -> dict[str, Any]:
    root_text = str(tracker_repo.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import mujoco
    from motion_tracking.quality import foot_contact_setup
    from motion_tracking.retarget import build_ref
    from motion_tracking.robots import G1

    motion = load_omg_motion(source)
    qpos = resample_qpos(motion.qpos, motion.fps, 50.0)
    qvel = qpos_to_qvel(qpos, 50.0)
    model = mujoco.MjModel.from_xml_path(str(G1.xml))
    ref, reason = build_ref(qpos, model, tag, foot_contact_setup(model, G1), robot=G1)
    result = {
        "tag": tag,
        "source_frames": len(motion.qpos),
        "source_fps": motion.fps,
        "resampled_frames": len(qpos),
        "gate_result": "passed" if ref is not None else "rejected",
        "reason": None if ref is not None else str(reason),
        "kinematics": {
            "root_height_min": float(qpos[:, 2].min()),
            "root_height_max": float(qpos[:, 2].max()),
            "root_speed_max": float(np.linalg.norm(qvel[:, :3], axis=1).max()),
            "joint_step_max": float(np.abs(np.diff(qpos[:, 7:], axis=0)).max()),
            "joint_speed_max": float(np.abs(qvel[:, 6:]).max()),
            "quaternion_norm_max_error": float(
                np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0))
            ),
        },
    }
    if ref is not None:
        result["accepted_metrics"] = {
            "foot_slide": float(ref["foot_slide"]),
            "hover": float(ref["hover"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-status", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads(args.prompt_status.read_text(encoding="utf-8"))
    if status.get("schema") != "text2motion-expanded-prompt-generation-v1":
        raise ValueError("unexpected prompt status schema")
    expected = {
        item["tag"]: item["quality_gate"]
        for item in status["prompts"]
        if item["source"] == "new_generation"
    }
    if len(expected) != 9:
        raise ValueError("expected nine new prompt records")

    records = []
    for tag, expected_result in expected.items():
        source = args.generated_root / tag / "reference_motion.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        record = diagnose(source, args.tracker_repo, tag)
        record["expected_gate_result"] = expected_result
        record["matches_original_gate"] = record["gate_result"] == expected_result
        records.append(record)
    if not all(item["matches_original_gate"] for item in records):
        raise RuntimeError("quality gate replay mismatch")
    reason_counts = Counter(
        item["reason"] for item in records if item["gate_result"] == "rejected"
    )
    payload = {
        "schema": "text2motion-prompt-quality-diagnostic-v1",
        "new_prompt_count": len(records),
        "passed": sum(item["gate_result"] == "passed" for item in records),
        "rejected": sum(item["gate_result"] == "rejected" for item in records),
        "reason_counts": dict(sorted(reason_counts.items())),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
