"""Score generated Text2Motion candidates from their rot139 root trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def trajectory(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
    delta = features[:, 132:134]
    position = np.vstack((np.zeros((1, 2)), np.cumsum(delta, axis=0)))
    frames = len(delta)
    early = position[max(2, frames // 3)] - position[0]
    late = position[-1] - position[min(frames - 1, 2 * frames // 3)]
    cross = early[0] * late[1] - early[1] * late[0]
    turn = math.degrees(math.atan2(cross, float(np.dot(early, late))))
    net = float(np.linalg.norm(position[-1]))
    length = float(np.linalg.norm(delta, axis=1).sum())
    d6 = features[:, :6]
    a1, a2 = d6[:, :3], d6[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    a2 = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    root_rotation = np.stack((b1, b2, b3), axis=-1)
    yaw = np.unwrap(np.arctan2(root_rotation[:, 1, 0], root_rotation[:, 0, 0]))
    window = max(3, min(15, frames // 10))
    yaw_unwrapped = math.degrees(float(np.median(yaw[-window:]) - np.median(yaw[:window])))
    yaw_change = (yaw_unwrapped + 180.0) % 360.0 - 180.0
    return {
        "frames": frames,
        "net_m": net,
        "path_m": length,
        "efficiency": net / max(length, 1e-9),
        "turn_deg": turn,
        "yaw_change_deg": yaw_change,
        "yaw_unwrapped_deg": yaw_unwrapped,
        "yaw_range_deg": math.degrees(float(yaw.max() - yaw.min())),
        "end_xz_m": position[-1].tolist(),
    }


def score(category: str, metrics: dict) -> float:
    base = metrics["efficiency"] + min(metrics["net_m"], 2.0) / 2.0
    turn = metrics["yaw_change_deg"]
    if category == "forward":
        return base - abs(turn) / 45.0
    if category == "left":
        return base + turn / 45.0 if turn > 0 else base - 10.0
    return base - turn / 45.0 if turn < 0 else base - 10.0


def quality(category: str, metrics: dict) -> bool:
    if category == "forward":
        return (
            metrics["net_m"] >= 0.75
            and metrics["efficiency"] >= 0.6
            and abs(metrics["yaw_change_deg"]) <= 25.0
            and metrics["yaw_range_deg"] <= 60.0
        )
    if category == "left":
        return (
            metrics["net_m"] >= 0.3
            and metrics["efficiency"] >= 0.3
            and 25.0 <= metrics["yaw_change_deg"] <= 120.0
            and metrics["yaw_range_deg"] <= 160.0
        )
    return (
        metrics["net_m"] >= 0.3
        and metrics["efficiency"] >= 0.3
        and -120.0 <= metrics["yaw_change_deg"] <= -25.0
        and metrics["yaw_range_deg"] <= 160.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text())["flat"]
    candidates = []
    for index, prompt in enumerate(prompts):
        matches = sorted(args.generated.glob(f"{index:02d}_*.npz"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one generated NPZ for index {index}, found {len(matches)}")
        item = dict(prompt)
        item["stem"] = matches[0].stem
        item["metrics"] = trajectory(matches[0])
        item["score"] = score(item["category"], item["metrics"])
        candidates.append(item)

    selected = []
    for category in ("forward", "left", "right"):
        best = max((item for item in candidates if item["category"] == category), key=lambda x: x["score"])
        best = dict(best)
        best["quality_gate"] = quality(category, best["metrics"])
        selected.append(best)
    result = {
        "schema": "nano-motion-motionhub-candidate-metrics-v1",
        "result": "passed" if all(item["quality_gate"] for item in selected) else "failed",
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
