"""Checkpointed single-motion probe for the generator-stage diagnostic runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from projects.text2motion_cerebellum.generator_stage_diagnostic import stage_kinematics
from projects.text2motion_cerebellum.omg_adapter import load_omg_motion, resample_qpos


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = {"schema": "text2motion-generator-stage-probe-v1", "steps": []}

    def step(name: str, **values) -> None:
        payload["steps"].append({"name": name, **values})
        write(args.out, payload)

    try:
        step("python_started")
        motion = load_omg_motion(args.source)
        step("motion_loaded", frames=len(motion.qpos), fps=motion.fps)
        raw_metrics = stage_kinematics(motion.qpos, motion.fps)
        step("raw_kinematics", joint_step=raw_metrics["joint_step_max_rad_frame"])
        resampled = resample_qpos(motion.qpos, motion.fps, 50.0)
        step("resampled", frames=len(resampled))

        root_text = str(args.tracker_repo.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        import mujoco
        from motion_tracking.quality import foot_contact_setup, ref_contact_stats
        from motion_tracking.retarget import build_ref
        from motion_tracking.robots import G1

        step("tracker_imported")
        model = mujoco.MjModel.from_xml_path(str(G1.xml))
        step("model_loaded", nq=model.nq, nv=model.nv)
        slide_setup = foot_contact_setup(model, G1)
        step("contact_setup")
        raw_contact = ref_contact_stats(
            model, mujoco.MjData(model), motion.qpos, *slide_setup
        )
        step("raw_contact", slide=float(raw_contact["slide"]), hover=float(raw_contact["hover"]))
        target_contact = ref_contact_stats(
            model, mujoco.MjData(model), resampled, *slide_setup
        )
        step(
            "target_contact",
            slide=float(target_contact["slide"]),
            hover=float(target_contact["hover"]),
        )
        ref, reason = build_ref(resampled, model, args.tag, slide_setup, robot=G1)
        gate_reason = "ok" if ref is not None else str(reason)
        step(
            "gate_complete",
            result="passed" if ref is not None else "rejected",
            reason=gate_reason,
        )
        payload["record"] = {
            "tag": args.tag,
            "source_contract": {
                "qpos_key": motion.qpos_key,
                "shape": list(motion.qpos.shape),
                "fps": motion.fps,
            },
            "raw_omg_30hz": {
                **raw_metrics,
                "foot_slide_native_frame_units": float(raw_contact["slide"]),
                "foot_slide_equivalent_at_50hz": float(
                    raw_contact["slide"] * motion.fps / 50.0
                ),
                "hover": float(raw_contact["hover"]),
            },
            "bridge_50hz": {
                "frames": len(resampled),
                "foot_slide": float(target_contact["slide"]),
                "hover": float(target_contact["hover"]),
            },
            "upstream_gate": {
                "result": "passed" if ref is not None else "rejected",
                "reason": gate_reason,
            },
        }
        payload["result"] = "passed"
        write(args.out, payload)
    except Exception as error:
        payload.update(
            {
                "result": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        write(args.out, payload)
        raise


if __name__ == "__main__":
    main()
