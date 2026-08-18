import unittest

import numpy as np

from projects.text2motion_cerebellum.generator_stage_diagnostic import (
    source_rate_evidence,
    stage_kinematics,
)


def synthetic_qpos(frames: int = 120) -> np.ndarray:
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    return qpos


class GeneratorStageDiagnosticTest(unittest.TestCase):
    def test_source_discontinuity_is_normalized_to_target_rate(self):
        qpos = synthetic_qpos()
        qpos[60:, 7] = 1.0
        metrics = stage_kinematics(qpos, 30.0)
        self.assertAlmostEqual(metrics["joint_step_max_rad_frame"], 1.0)
        self.assertAlmostEqual(metrics["joint_step_equivalent_at_50hz"], 0.6)
        self.assertTrue(source_rate_evidence(metrics, "continuity"))
        self.assertEqual(metrics["joint_step_worst_transition"], [59, 60])

    def test_smooth_source_has_no_rate_evidence(self):
        qpos = synthetic_qpos()
        qpos[:, 0] = np.linspace(0.0, 1.0, len(qpos))
        qpos[:, 7] = np.linspace(0.0, 0.5, len(qpos))
        metrics = stage_kinematics(qpos, 30.0)
        self.assertFalse(source_rate_evidence(metrics, "speed"))
        self.assertFalse(source_rate_evidence(metrics, "joint_vel"))
        self.assertFalse(source_rate_evidence(metrics, "continuity"))


if __name__ == "__main__":
    unittest.main()
