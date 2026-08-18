"""Bridge a pretrained text-to-motion model to the motion-tracking project."""

from .omg_adapter import (
    G1_JOINT_NAMES,
    LoadedMotion,
    build_tracker_ref,
    load_omg_motion,
    qpos_to_qvel,
    resample_qpos,
    validate_qpos_36,
    write_ref_shard,
)

__all__ = [
    "G1_JOINT_NAMES",
    "LoadedMotion",
    "build_tracker_ref",
    "load_omg_motion",
    "qpos_to_qvel",
    "resample_qpos",
    "validate_qpos_36",
    "write_ref_shard",
]
