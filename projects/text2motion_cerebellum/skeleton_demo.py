"""Render side-by-side reference/policy skeleton demos without an OpenGL context."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


EDGES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7),
    (7, 8), (8, 9), (9, 10),
    (7, 11), (11, 12), (12, 13),
)


def slug(value: str) -> str:
    result = re.sub(r"\.npz$", "", str(value))
    return re.sub(r"[^a-z0-9]+", "-", result.lower()).strip("-")[-46:]


def caption(value: str) -> str:
    name = slug(value)
    if "turn-left" in name:
        return "walk forward and turn left"
    if "turn-right" in name:
        return "walk forward and turn right"
    return "walk forward"


def project(points, panel_left: int):
    import numpy as np

    relative = np.asarray(points, dtype=np.float64).copy()
    relative[:, :2] -= relative[0, :2]
    yaw = np.deg2rad(135.0)
    horizontal = np.cos(yaw) * relative[:, 0] + np.sin(yaw) * relative[:, 1]
    depth = -np.sin(yaw) * relative[:, 0] + np.cos(yaw) * relative[:, 1]
    vertical = points[:, 2] + 0.18 * depth
    x = panel_left + 240 + 205 * horizontal
    y = 430 - 245 * vertical
    return np.stack((x, y), axis=1).round().astype(np.int32)


def draw_panel(frame, points, panel_left: int, title: str, color) -> None:
    import cv2

    coords = project(points, panel_left)
    cv2.rectangle(frame, (panel_left, 0), (panel_left + 479, 479), (20, 24, 31), -1)
    cv2.line(frame, (panel_left + 18, 430), (panel_left + 462, 430), (65, 71, 82), 2)
    for offset in (-160, -80, 0, 80, 160):
        cv2.line(frame, (panel_left + 240 + offset, 424),
                 (panel_left + 240 + offset, 436), (65, 71, 82), 1)
    for start, end in EDGES:
        cv2.line(frame, tuple(coords[start]), tuple(coords[end]), color, 6,
                 lineType=cv2.LINE_AA)
    for index, point in enumerate(coords):
        radius = 8 if index in (0, 7) else 6
        cv2.circle(frame, tuple(point), radius, (238, 242, 247), -1,
                   lineType=cv2.LINE_AA)
        cv2.circle(frame, tuple(point), radius, color, 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, title, (panel_left + 20, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)


def draw_frame(ref_points, sim_points, prompt: str, step: int, total: int):
    import cv2
    import numpy as np

    frame = np.zeros((480, 964, 3), dtype=np.uint8)
    draw_panel(frame, ref_points, 0, "TARGET (OMG reference)", (235, 190, 45))
    draw_panel(frame, sim_points, 484, "CONTROLLED G1 (tracker)", (65, 165, 245))
    frame[:, 480:484] = 78
    progress = min(1.0, step / max(1, total))
    cv2.rectangle(frame, (20, 451), (944, 467), (45, 51, 61), -1)
    cv2.rectangle(frame, (20, 451), (20 + int(924 * progress), 467),
                  (80, 190, 120), -1)
    cv2.putText(frame, prompt, (20, 474), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (225, 228, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, f"frame {step}/{total} | observation noise ON",
                (655, 474), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (190, 196, 207), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--refs-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="t2m_skeleton_")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--claim-scope",
        default="partial-training demo; not an official-scale reproduction",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.tracker_repo.resolve()))
    import cv2
    import mujoco
    import numpy as np
    from motion_tracking.data import load_amass_refs
    from motion_tracking.env import TrackVecEnv
    from motion_tracking.ppo import env_kwargs_from_meta, load_policy
    from motion_tracking.robots import G1

    agent, metadata = load_policy(str(args.policy))
    env_kw = env_kwargs_from_meta(metadata, obs_noise=True)
    refs = load_amass_refs(n=3, seed=12345, split="all",
                           ref_dir=args.refs_dir, robot=G1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos = []

    for clip_index, reference in enumerate(refs):
        name = slug(reference["clip"])
        prompt = caption(reference["clip"])
        destination = args.out_dir / f"{args.prefix}{name}.mp4"
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (964, 480)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV mp4v writer did not open")

        env = TrackVecEnv([reference], num_envs=1, seed=clip_index,
                          robot=G1, **env_kw)
        body_ids = [env.model.body(value).id for value in G1.tracked_bodies]
        data = env.datas[0]
        ref_data = mujoco.MjData(env.model)
        total = int(reference["qpos"].shape[0])
        env.clip[0] = 0
        env.t[0] = 0
        data.qpos[:] = reference["qpos"][0]
        data.qvel[:] = reference["qvel"][0]
        env.prev_action[0] = 0
        mujoco.mj_forward(env.model, data)
        env.begin_episode(0)

        steps = frames = 0
        terminated = truncated = False
        while True:
            action = agent.act(env._obs_actor(0)[None], deterministic=True)[0][0]
            _, terminated, truncated = env._step_one(0, action.astype(np.float32))
            steps += 1
            ref_index = min(int(env.t[0]), total - 1)
            if steps % 2 == 0:
                ref_data.qpos[:] = reference["qpos"][ref_index]
                mujoco.mj_kinematics(env.model, ref_data)
                frame = draw_frame(
                    ref_data.xpos[body_ids].copy(), data.xpos[body_ids].copy(),
                    prompt, steps, total - 1,
                )
                writer.write(frame)
                frames += 1
            if terminated or truncated or int(env.t[0]) >= total - 1:
                break
        writer.release()
        if not destination.is_file() or destination.stat().st_size <= 1000:
            raise RuntimeError("rendered MP4 is missing or empty")
        videos.append({
            "clip": str(reference["clip"]),
            "prompt": prompt,
            "file": destination.name,
            "bytes": destination.stat().st_size,
            "frames": frames,
            "steps": steps,
            "reference_steps": total - 1,
            "completion": steps / max(1, total - 1),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })

    quantitative = json.loads(args.summary.read_text(encoding="utf-8"))
    report = {
        "schema": "text2motion-cerebellum-skeleton-demo-v1",
        "architecture": "text -> OMG -> G1 reference -> upstream motion_tracking policy -> simulated G1",
        "policy": args.policy.name,
        "rendering": "kinematic skeleton projection; rollout remains MuJoCo physics",
        "condition": "hardware observation noise enabled",
        "claim_scope": args.claim_scope,
        "quantitative": {
            "native": quantitative["native"]["after"],
            "omg": quantitative["omg"]["after"],
            "by_prompt": quantitative["omg_after_by_prompt"],
        },
        "videos": videos,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(args.report)


if __name__ == "__main__":
    main()
