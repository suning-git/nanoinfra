import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.text2motion_cerebellum.omg_adapter import (
    G1_JOINT_NAMES,
    build_tracker_ref,
    load_omg_motion,
    qpos_to_qvel,
    resample_qpos,
    validate_qpos_36,
    write_ref_shard,
    convert_file,
)


def synthetic_qpos(fps: float = 30.0, seconds: float = 1.0) -> np.ndarray:
    frames = int(round(fps * seconds)) + 1
    time = np.linspace(0.0, seconds, frames)
    qpos = np.zeros((frames, 36), np.float32)
    qpos[:, 0] = time
    qpos[:, 2] = 0.8
    yaw = 0.5 * time
    qpos[:, 3] = np.cos(yaw / 2.0)
    qpos[:, 6] = np.sin(yaw / 2.0)
    qpos[:, 7] = 0.25 * time
    return qpos


class OmgAdapterTest(unittest.TestCase):
    def test_resample_and_qvel_preserve_constant_motion(self):
        resampled = resample_qpos(synthetic_qpos(), 30.0, 50.0)
        self.assertEqual(resampled.shape, (51, 36))
        np.testing.assert_allclose(resampled[:, 0], np.linspace(0, 1, 51), atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(resampled[:, 3:7], axis=1), 1.0, atol=1e-6
        )

        qvel = qpos_to_qvel(resampled, 50.0)
        self.assertEqual(qvel.shape, (51, 35))
        np.testing.assert_allclose(qvel[:, 0], 1.0, atol=2e-5)
        np.testing.assert_allclose(qvel[:, 5], 0.5, atol=2e-5)
        np.testing.assert_allclose(qvel[:, 6], 0.25, atol=2e-5)

    def test_quaternion_sign_flip_is_removed(self):
        qpos = synthetic_qpos()
        qpos[15:, 3:7] *= -1.0
        validated = validate_qpos_36(qpos)
        dots = np.sum(validated[:-1, 3:7] * validated[1:, 3:7], axis=1)
        self.assertTrue(np.all(dots > 0.0))

    def test_resample_uses_the_official_target_rate_clock(self):
        # Eight 30 Hz frames end at t=7/30. At 50 Hz, valid timestamps are
        # 0.00 through 0.22, i.e. 12 frames; the 0.24 point is outside the clip.
        qpos = synthetic_qpos(fps=30.0, seconds=7.0 / 30.0)
        resampled = resample_qpos(qpos, 30.0, 50.0)
        self.assertEqual(resampled.shape, (12, 36))
        self.assertAlmostEqual(float(resampled[-1, 0]), 0.22, places=6)

    def test_wrong_joint_order_is_rejected(self):
        names = list(G1_JOINT_NAMES)
        names[0], names[1] = names[1], names[0]
        with self.assertRaisesRegex(ValueError, "joint order"):
            validate_qpos_36(synthetic_qpos(), joint_names=names)

    def test_npz_load_and_tracker_shard_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "omg_output.npz"
            np.savez(
                source,
                qpos_36=synthetic_qpos(),
                fps=np.array(30.0),
                joint_names=np.array(G1_JOINT_NAMES),
            )
            motion = load_omg_motion(source)
            self.assertEqual(motion.qpos_key, "qpos_36")
            self.assertEqual(motion.fps, 30.0)

            qpos = resample_qpos(motion.qpos, motion.fps)
            feet = np.zeros((len(qpos), 2, 3), np.float32)
            ref = build_tracker_ref(
                qpos,
                fps=50.0,
                feet=feet,
                clip="synthetic",
                caption="walk forward",
            )
            destination = write_ref_shard(root / "shard_000.npz", (ref,))
            with np.load(destination, allow_pickle=True) as shard:
                loaded = shard["refs"].tolist()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["robot"], "g1")
            self.assertEqual(loaded[0]["caption"], "walk forward")
            self.assertEqual(loaded[0]["qpos"].shape, (51, 36))
            self.assertEqual(loaded[0]["qvel"].shape, (51, 35))
            self.assertEqual(loaded[0]["feet"].shape, (51, 2, 3))

    def test_existing_shard_is_not_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "shard_000.npz"
            ref = {"robot": "g1"}
            write_ref_shard(destination, (ref,))
            with self.assertRaises(FileExistsError):
                write_ref_shard(destination, (ref,))

    def test_convert_requires_exactly_one_target_model_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "motion.npz"
            np.savez(source, qpos_36=synthetic_qpos(), fps=np.float32(30.0))
            common = {
                "source": source,
                "destination": root / "out.npz",
                "caption": "walk forward",
            }
            with self.assertRaisesRegex(ValueError, "exactly one"):
                convert_file(**common)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                convert_file(
                    **common,
                    mjcf_path=root / "scene.xml",
                    tracker_repo=root / "tracker",
                )


if __name__ == "__main__":
    unittest.main()
