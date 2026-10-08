import unittest

import numpy as np

from projects.text2motion_cerebellum.reference_repair import (
    apply_contact_root_lock,
    inpaint_joint_discontinuities,
    kinematic_maxima,
    required_time_scale,
    smooth_linear_channels,
    stretch_qpos,
)


def synthetic(frames: int = 21) -> np.ndarray:
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    return qpos


class ReferenceRepairTest(unittest.TestCase):
    def test_time_stretch_brings_linear_limits_below_targets(self) -> None:
        qpos = synthetic()
        qpos[:, 0] = np.linspace(0.0, 2.0, len(qpos))
        qpos[10:, 7] = 1.2
        config = {
            "target_root_speed_m_s": 1.8,
            "target_joint_speed_rad_s": 13.5,
            "target_joint_step_rad_frame": 0.45,
        }
        scale = required_time_scale(qpos, config)
        repaired = stretch_qpos(qpos, scale)
        repaired = smooth_linear_channels(repaired, half_width=4)
        maxima = kinematic_maxima(repaired)
        self.assertLessEqual(maxima["root_speed_max"], 1.8 + 1e-4)
        self.assertLessEqual(maxima["joint_speed_max"], 13.5 + 1e-4)
        self.assertLessEqual(maxima["joint_step_max"], 0.45 + 1e-4)
        self.assertGreater(len(repaired), len(qpos))

    def test_contact_root_lock_cancels_planted_geom_motion(self) -> None:
        qpos = synthetic(frames=8)
        contact = np.zeros((8, 2, 3), dtype=np.float32)
        contact[:, :, 0] = -0.01 * np.arange(8)[:, None]
        contact[:, :, 2] = 0.01
        repaired, correction = apply_contact_root_lock(
            qpos,
            contact,
            floor=0.0,
            contact_tolerance=0.02,
            gain=1.0,
        )
        corrected_contact = contact[:, :, :2] + correction[:, None, :]
        np.testing.assert_allclose(np.diff(corrected_contact, axis=0), 0.0, atol=1e-7)
        np.testing.assert_allclose(repaired[:, 0], 0.01 * np.arange(8), atol=1e-7)

    def test_zero_width_smoothing_is_identity(self) -> None:
        qpos = synthetic()
        np.testing.assert_array_equal(smooth_linear_channels(qpos, 0), qpos)

    def test_discontinuity_inpaint_repairs_step_without_lengthening_clip(self) -> None:
        qpos = synthetic(frames=101)
        qpos[50:, 7] = 1.5
        repaired, detail = inpaint_joint_discontinuities(
            qpos,
            threshold=0.5,
            half_window=12,
        )
        self.assertEqual(len(repaired), len(qpos))
        self.assertEqual(detail["events"], 1)
        self.assertLess(np.abs(np.diff(repaired[:, 7])).max(), 0.2)
        self.assertAlmostEqual(float(repaired[0, 7]), 0.0)
        self.assertAlmostEqual(float(repaired[-1, 7]), 1.5)


if __name__ == "__main__":
    unittest.main()
