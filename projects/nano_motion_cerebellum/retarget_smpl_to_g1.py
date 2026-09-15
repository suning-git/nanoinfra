"""Retarget an isolated SMPL-H handoff to a quality-gated G1 tracker reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


def human_frames(local_rotmat, trans, *, fps, robot):
    """Build GMR frames while fixing the pinned upstream robot-argument omission."""
    from motion_tracking.retarget import FOOT_JOINTS, SMPL_TO_GMR, resample_axis_angle
    from motion_tracking.smplh import get_smplh

    poses = Rot.from_matrix(local_rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(
        len(local_rotmat), 22, 3
    )
    poses, trans = resample_axis_angle(poses, trans, fps, robot=robot)
    body = get_smplh("neutral")
    rotmats = torch.from_numpy(
        Rot.from_rotvec(poses.reshape(-1, 3)).as_matrix()
        .reshape(len(poses), 22, 3, 3).astype(np.float32)
    )
    betas = torch.zeros(16, dtype=torch.float32)
    pos, global_rot = body.fk_joints_full(
        rotmats, torch.from_numpy(trans.astype(np.float32)), betas
    )
    pos = pos.numpy()
    global_rot = global_rot.numpy()
    quat = Rot.from_matrix(global_rot.reshape(-1, 3, 3)).as_quat(
        scalar_first=True
    ).reshape(len(pos), 22, 4)
    frames = [
        {
            name: (pos[t, joint], quat[t, joint])
            for joint, name in SMPL_TO_GMR.items()
        }
        for t in range(len(pos))
    ]
    height = 1.66
    ground = float(pos[:, FOOT_JOINTS, 2].min())
    return frames, height, ground


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ref-output", type=Path, required=True)
    parser.add_argument("--qpos-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--warmup-solves", type=int, default=0)
    args = parser.parse_args()

    import mujoco
    from motion_tracking.quality import foot_contact_setup
    from motion_tracking.retarget import (
        _gmr_for_height,
        _import_gmr,
        build_ref,
        ground_align,
        register_with_gmr,
    )
    from motion_tracking.robots import get_robot

    with np.load(args.input, allow_pickle=False) as data:
        local_rotmat = np.asarray(data["local_rotmat"], dtype=np.float32)
        trans = np.asarray(data["trans"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if local_rotmat.shape != (len(trans), 22, 3, 3):
        raise ValueError(f"invalid SMPL handoff shape: {local_rotmat.shape}/{trans.shape}")

    robot = get_robot("g1")
    register_with_gmr(robot)
    if args.time_scale < 1.0:
        raise ValueError("time scale must be at least one")
    frames, height, ground = human_frames(
        local_rotmat, trans, fps=fps / args.time_scale, robot=robot
    )
    gmr = _gmr_for_height(
        _import_gmr().GeneralMotionRetargeting, height, robot, verbose=False
    )
    gmr.set_ground_offset(ground)
    if args.warmup_solves < 0:
        raise ValueError("warmup solves must be non-negative")
    for _ in range(args.warmup_solves):
        gmr.retarget(frames[0])
    qpos = np.stack([gmr.retarget(frame) for frame in frames]).astype(np.float32)

    model = mujoco.MjModel.from_xml_path(str(robot.xml))
    slide_setup = foot_contact_setup(model, robot)
    foot_ids, foot_geom_z, _ = slide_setup
    qpos = ground_align(qpos, model, foot_geom_z, foot_ids)
    ref, reason = build_ref(qpos, model, args.clip, slide_setup, robot=robot)
    joint_steps = np.abs(np.diff(qpos[:, 7:], axis=0))
    worst_joint_step = np.unravel_index(int(joint_steps.argmax()), joint_steps.shape)
    np.savez_compressed(
        args.qpos_output, qpos_36=qpos, fps=np.float32(1.0 / robot.ctrl_dt)
    )
    report = {
        "schema": "nano-motion-smpl-to-g1-preflight-v1",
        "result": "passed" if ref is not None else "failed",
        "quality_gate_reason": reason,
        "source_frames": int(len(local_rotmat)),
        "g1_frames": int(len(qpos)),
        "source_fps": fps,
        "target_fps": float(1.0 / robot.ctrl_dt),
        "time_scale": float(args.time_scale),
        "warmup_solves": int(args.warmup_solves),
        "qpos_shape": list(qpos.shape),
        "root_height_median": float(np.median(qpos[:, 2])),
        "root_displacement_m": float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])),
        "max_joint_step_rad": float(joint_steps.max()),
        "max_joint_speed_rad_s": float(joint_steps.max() / robot.ctrl_dt),
        "worst_joint_step_frame": int(worst_joint_step[0]),
        "worst_joint_index": int(worst_joint_step[1]),
    }
    if ref is not None:
        report["foot_slide"] = float(ref["foot_slide"])
        report["hover"] = float(ref["hover"])
        ref = dict(ref)
        ref["caption"] = args.caption
        ref["time_scale"] = float(args.time_scale)
        args.ref_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.ref_output, refs=np.array([ref], dtype=object))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if ref is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
