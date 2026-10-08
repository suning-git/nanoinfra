"""Convert OMG-generated G1 motions into ``motion_tracking`` references.

OMG and ``suning-git/motion_tracking`` use the same Unitree G1 state layout:

    qpos_36 = [root_xyz(3), root_quaternion_wxyz(4), g1_joints(29)]

There is therefore no human-to-robot retargeting in this bridge.  The conversion
only validates the contract, resamples the generated motion to the tracker's
50 Hz control rate, differentiates it using MuJoCo's free-joint convention, and
uses the target MJCF to calculate the two foot body positions.

NumPy is the only import-time dependency.  MuJoCo is imported lazily by
``feet_from_mjcf`` so format checks and unit tests remain lightweight.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

QPOS_KEYS = ("qpos_36", "pred_qpos_36", "qpos")
DEFAULT_SOURCE_FPS = 30.0
DEFAULT_TARGET_FPS = 50.0
DEFAULT_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


@dataclass(frozen=True)
class LoadedMotion:
    """Validated OMG motion and the provenance needed for a reproducible bridge."""

    qpos: np.ndarray
    fps: float
    qpos_key: str
    source: Path


def _positive_fps(value: float, label: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return fps


def _joint_names_from_npz(data: np.lib.npyio.NpzFile) -> tuple[str, ...] | None:
    for key in ("joint_names", "g1_joint_names"):
        if key in data.files:
            values = np.asarray(data[key]).reshape(-1)
            return tuple(str(value) for value in values)
    return None


def validate_qpos_36(
    qpos: np.ndarray,
    *,
    joint_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Return a float32, quaternion-normalized copy of a G1 ``qpos_36`` array.

    Quaternion signs are made continuous over time.  This does not alter the
    represented orientations, but prevents artificial jumps during interpolation
    or finite differencing.
    """

    out = np.asarray(qpos, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 36:
        raise ValueError(f"expected qpos shape [T, 36], got {out.shape}")
    if out.shape[0] < 2:
        raise ValueError("a tracker reference needs at least two frames")
    if not np.isfinite(out).all():
        bad = np.argwhere(~np.isfinite(out))[0]
        raise ValueError(f"qpos contains a non-finite value at frame/column {tuple(bad)}")

    if joint_names is not None and tuple(joint_names) != G1_JOINT_NAMES:
        raise ValueError(
            "joint order does not match OMG/motion_tracking G1 order; refusing "
            "a silent 29-DoF permutation"
        )

    quat = out[:, 3:7]
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms < 1e-8):
        frame = int(np.flatnonzero(norms < 1e-8)[0])
        raise ValueError(f"root quaternion is degenerate at frame {frame}")
    quat /= norms[:, None]
    for index in range(1, len(quat)):
        if float(np.dot(quat[index - 1], quat[index])) < 0.0:
            quat[index] *= -1.0
    out[:, 3:7] = quat
    return out.astype(np.float32)


def load_omg_motion(path: str | Path, *, fps: float | None = None) -> LoadedMotion:
    """Load an official OMG ``.npz`` output.

    The official key is ``qpos_36``.  ``pred_qpos_36`` and ``qpos`` are accepted
    because OMG's tracking loader accepts those aliases too.  If an older export
    has no ``fps`` field, its documented 30 Hz generation rate is used unless the
    caller supplies an explicit override.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        qpos_key = next((key for key in QPOS_KEYS if key in data.files), None)
        if qpos_key is None:
            raise ValueError(
                f"{source} has no motion key; expected one of {', '.join(QPOS_KEYS)}"
            )
        names = _joint_names_from_npz(data)
        loaded_qpos = np.asarray(data[qpos_key])
        if fps is None:
            loaded_fps = (
                float(np.asarray(data["fps"]).reshape(-1)[0])
                if "fps" in data.files
                else DEFAULT_SOURCE_FPS
            )
        else:
            loaded_fps = fps

    return LoadedMotion(
        qpos=validate_qpos_36(loaded_qpos, joint_names=names),
        fps=_positive_fps(loaded_fps, "source fps"),
        qpos_key=qpos_key,
        source=source.resolve(),
    )


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-arc SLERP for two normalized scalar-first quaternions."""

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - fraction) * theta) / sin_theta
    b = np.sin(fraction * theta) / sin_theta
    return a * q0 + b * q1


def resample_qpos(
    qpos: np.ndarray,
    source_fps: float,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> np.ndarray:
    """Resample G1 states with linear translation/joints and root-quaternion SLERP."""

    source_fps = _positive_fps(source_fps, "source fps")
    target_fps = _positive_fps(target_fps, "target fps")
    source = validate_qpos_36(qpos).astype(np.float64)
    if np.isclose(source_fps, target_fps):
        return source.astype(np.float32)

    duration = (len(source) - 1) / source_fps
    # Match OMG's HoloMotion bridge exactly: only emit points on the target-rate
    # clock that lie inside the source interval.  Using linspace here would make
    # the nominal 50 Hz output subtly non-uniform for most clip lengths.
    target_frames = max(2, int(np.floor(duration * target_fps)) + 1)
    source_times = np.arange(len(source), dtype=np.float64) / source_fps
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    target_times = np.clip(target_times, source_times[0], source_times[-1])

    result = np.empty((target_frames, 36), dtype=np.float64)
    linear_columns = (0, 1, 2, *range(7, 36))
    for column in linear_columns:
        result[:, column] = np.interp(target_times, source_times, source[:, column])

    right = np.searchsorted(source_times, target_times, side="right")
    left = np.clip(right - 1, 0, len(source) - 2)
    for out_index, (time, left_index) in enumerate(zip(target_times, left)):
        interval = source_times[left_index + 1] - source_times[left_index]
        fraction = (time - source_times[left_index]) / interval
        result[out_index, 3:7] = _slerp_wxyz(
            source[left_index, 3:7], source[left_index + 1, 3:7], float(fraction)
        )

    return validate_qpos_36(result)


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _quat_to_rotvec_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector_norm = float(np.linalg.norm(quat[1:]))
    if vector_norm < 1e-10:
        return 2.0 * quat[1:]
    angle = 2.0 * np.arctan2(vector_norm, float(quat[0]))
    return quat[1:] * (angle / vector_norm)


def qpos_to_qvel(qpos: np.ndarray, fps: float = DEFAULT_TARGET_FPS) -> np.ndarray:
    """Differentiate qpos into MuJoCo G1 qvel ``[T, 35]``.

    Root linear velocity is in the world frame.  Root angular velocity is the
    local shortest rotation ``inverse(q[t]) * q[t+1]``, matching the original
    repository's ``retarget.build_ref`` implementation.
    """

    fps = _positive_fps(fps, "fps")
    qpos = validate_qpos_36(qpos).astype(np.float64)
    qvel = np.zeros((len(qpos), 35), dtype=np.float64)
    qvel[:-1, 0:3] = np.diff(qpos[:, 0:3], axis=0) * fps
    for index in range(len(qpos) - 1):
        q0 = qpos[index, 3:7]
        q1 = qpos[index + 1, 3:7]
        inverse_q0 = np.array((q0[0], -q0[1], -q0[2], -q0[3]))
        delta = _quat_mul_wxyz(inverse_q0, q1)
        qvel[index, 3:6] = _quat_to_rotvec_wxyz(delta) * fps
    qvel[:-1, 6:] = np.diff(qpos[:, 7:], axis=0) * fps
    qvel[-1] = qvel[-2]
    return qvel.astype(np.float32)


def feet_from_mjcf(
    qpos: np.ndarray,
    mjcf_path: str | Path,
    foot_bodies: Sequence[str] = DEFAULT_FOOT_BODIES,
) -> np.ndarray:
    """Calculate world-frame foot body positions with the tracker's target MJCF."""

    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "foot FK requires the optional 'mujoco' package; install it in the "
            "conversion environment"
        ) from error

    qpos = validate_qpos_36(qpos)
    model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path)))
    if model.nq != 36 or model.nv != 35:
        raise ValueError(
            f"MJCF has nq={model.nq}, nv={model.nv}; expected G1 nq=36, nv=35"
        )
    body_ids = []
    for name in foot_bodies:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MJCF has no foot body named {name!r}")
        body_ids.append(body_id)

    data = mujoco.MjData(model)
    feet = np.empty((len(qpos), len(body_ids), 3), dtype=np.float32)
    for index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        feet[index] = data.xpos[body_ids]
    return feet


def build_tracker_ref(
    qpos: np.ndarray,
    *,
    fps: float,
    feet: np.ndarray,
    clip: str,
    caption: str,
) -> dict[str, object]:
    """Build the exact object-dict shape consumed by ``motion_tracking.data``."""

    qpos = validate_qpos_36(qpos)
    feet = np.asarray(feet, dtype=np.float32)
    if feet.shape != (len(qpos), 2, 3):
        raise ValueError(f"expected feet shape {(len(qpos), 2, 3)}, got {feet.shape}")
    if not np.isfinite(feet).all():
        raise ValueError("feet contains a non-finite value")
    return {
        "robot": "g1",
        "qpos": qpos,
        "qvel": qpos_to_qvel(qpos, fps),
        "feet": feet,
        "clip": str(clip),
        "caption": str(caption),
        "time_scale": 1.0,
        "foot_slide": None,
        "hover": None,
    }


def build_tracker_ref_via_upstream(
    qpos: np.ndarray,
    *,
    tracker_repo: str | Path,
    clip: str,
    caption: str,
) -> dict[str, object]:
    """Build and quality-gate a reference with ``motion_tracking`` itself.

    This is the preferred production bridge.  Besides constructing ``qvel`` and
    foot positions, upstream ``retarget.build_ref`` rejects implausible posture,
    joint discontinuities, excessive speeds, foot sliding, and hovering.  Using
    the tracker's own pinned checkout also ensures FK is evaluated with exactly
    the MJCF used by the policy rather than a merely shape-compatible G1 model.
    """

    root = Path(tracker_repo).resolve()
    if not (root / "motion_tracking" / "retarget.py").is_file():
        raise ValueError(f"not a motion_tracking checkout: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    try:
        import mujoco
        from motion_tracking.quality import foot_contact_setup
        from motion_tracking.retarget import build_ref as upstream_build_ref
        from motion_tracking.robots import G1
    except ImportError as error:
        raise RuntimeError(
            "upstream conversion requires motion_tracking and its MuJoCo runtime"
        ) from error

    model = mujoco.MjModel.from_xml_path(str(G1.xml))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError(
            "motion_tracking G1 dimensions are "
            f"{model.nq}/{model.nv}/{model.nu}; expected 36/35/29"
        )
    actuator_joints = tuple(
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[index, 0]),
        )
        for index in range(model.nu)
    )
    if actuator_joints != G1_JOINT_NAMES:
        raise ValueError("motion_tracking G1 actuator order differs from OMG")

    checked_qpos = validate_qpos_36(qpos)
    slide_setup = foot_contact_setup(model, G1)
    ref, reason = upstream_build_ref(
        checked_qpos,
        model,
        str(clip),
        slide_setup,
        robot=G1,
    )
    if ref is None:
        raise ValueError(f"motion_tracking quality gate rejected reference: {reason}")
    result = dict(ref)
    result["caption"] = str(caption)
    return result


def write_ref_shard(
    path: str | Path,
    refs: Iterable[dict[str, object]],
    *,
    overwrite: bool = False,
) -> Path:
    """Write refs using the original repository's object-array shard format."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing shard: {destination}")
    refs = list(refs)
    if not refs:
        raise ValueError("ref shard cannot be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, refs=np.array(refs, dtype=object))
    return destination


def convert_file(
    source: str | Path,
    destination: str | Path,
    *,
    mjcf_path: str | Path | None = None,
    tracker_repo: str | Path | None = None,
    caption: str,
    clip: str | None = None,
    source_fps: float | None = None,
    target_fps: float = DEFAULT_TARGET_FPS,
    overwrite: bool = False,
) -> dict[str, object]:
    """End-to-end conversion for one OMG output file."""

    if (mjcf_path is None) == (tracker_repo is None):
        raise ValueError("provide exactly one of mjcf_path or tracker_repo")

    motion = load_omg_motion(source, fps=source_fps)
    qpos = resample_qpos(motion.qpos, motion.fps, target_fps)
    clip_name = clip or motion.source.name
    if tracker_repo is not None:
        ref = build_tracker_ref_via_upstream(
            qpos,
            tracker_repo=tracker_repo,
            clip=clip_name,
            caption=caption,
        )
    else:
        feet = feet_from_mjcf(qpos, mjcf_path)
        ref = build_tracker_ref(
            qpos,
            fps=target_fps,
            feet=feet,
            clip=clip_name,
            caption=caption,
        )
    write_ref_shard(destination, (ref,), overwrite=overwrite)
    return ref


def _summary(motion: LoadedMotion) -> dict[str, object]:
    return {
        "source": str(motion.source),
        "qpos_key": motion.qpos_key,
        "frames": int(len(motion.qpos)),
        "fps": motion.fps,
        "duration_seconds": (len(motion.qpos) - 1) / motion.fps,
        "shape": list(motion.qpos.shape),
        "quaternion_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(motion.qpos[:, 3:7], axis=1) - 1.0))
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="validate and summarize OMG output")
    inspect_parser.add_argument("source", type=Path)
    inspect_parser.add_argument("--source-fps", type=float)

    convert_parser = commands.add_parser("convert", help="write one tracker ref shard")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("destination", type=Path)
    target = convert_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--mjcf", type=Path)
    target.add_argument("--tracker-repo", type=Path)
    convert_parser.add_argument("--caption", required=True)
    convert_parser.add_argument("--clip")
    convert_parser.add_argument("--source-fps", type=float)
    convert_parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    convert_parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "inspect":
        print(json.dumps(_summary(load_omg_motion(args.source, fps=args.source_fps)), indent=2))
        return

    ref = convert_file(
        args.source,
        args.destination,
        mjcf_path=args.mjcf,
        tracker_repo=args.tracker_repo,
        caption=args.caption,
        clip=args.clip,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        overwrite=args.force,
    )
    print(
        json.dumps(
            {
                "output": str(args.destination.resolve()),
                "frames": len(ref["qpos"]),
                "fps": args.target_fps,
                "caption": ref["caption"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
