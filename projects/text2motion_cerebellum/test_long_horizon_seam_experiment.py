import unittest

import numpy as np

from projects.text2motion_cerebellum.long_horizon_seam_experiment import (
    c1_residual_stitch,
    normalize_quaternion,
    planar_space_aligned_selective_c1_stitch,
    select_candidate,
)


def motion() -> np.ndarray:
    qpos = np.zeros((120, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:60, 0] = np.arange(60) * 0.01
    qpos[60:, 0] = 1.2 + np.arange(60) * 0.03
    qpos[:60, 7] = np.arange(60) * 0.02
    qpos[60:, 7] = 2.0 + np.arange(60) * -0.01
    angle = np.deg2rad(20.0)
    qpos[60:, 3] = np.cos(angle / 2.0)
    qpos[60:, 6] = np.sin(angle / 2.0)
    return qpos


class C1ResidualStitchTest(unittest.TestCase):
    def test_preserves_first_chunk_and_returns_to_source(self) -> None:
        source = motion()
        repaired, metrics = c1_residual_stitch(
            source, boundary_frame=60, decay_frames=30
        )
        np.testing.assert_allclose(repaired[:60], source[:60])
        np.testing.assert_allclose(repaired[90:], source[90:], atol=1e-6)
        np.testing.assert_allclose(repaired[60, :3], source[59, :3], atol=1e-6)
        np.testing.assert_allclose(repaired[60, 7:], source[59, 7:], atol=1e-6)
        self.assertGreater(metrics["linear_correction_rms"], 0.0)

    def test_matches_root_orientation_at_the_seam(self) -> None:
        source = motion()
        repaired, _ = c1_residual_stitch(
            source, boundary_frame=60, decay_frames=30
        )
        expected = normalize_quaternion(source[59, 3:7])
        actual = normalize_quaternion(repaired[60, 3:7])
        self.assertAlmostEqual(abs(float(np.dot(expected, actual))), 1.0, places=6)
        norms = np.linalg.norm(repaired[:, 3:7], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_reduces_seam_position_and_velocity_residuals(self) -> None:
        source = motion()
        repaired, _ = c1_residual_stitch(
            source, boundary_frame=60, decay_frames=30
        )
        source_pose_step = abs(float(source[60, 7] - source[59, 7]))
        repaired_pose_step = abs(float(repaired[60, 7] - repaired[59, 7]))
        self.assertGreater(source_pose_step, 0.5)
        self.assertLess(repaired_pose_step, 1e-6)
        incoming = repaired[59, 7] - repaired[58, 7]
        outgoing = repaired[61, 7] - repaired[60, 7]
        self.assertLess(abs(float(incoming - outgoing)), 0.01)

    def test_rejects_horizon_that_does_not_fit(self) -> None:
        with self.assertRaises(ValueError):
            c1_residual_stitch(motion(), boundary_frame=60, decay_frames=60)


class CandidateSelectionTest(unittest.TestCase):
    def test_ranking_uses_development_and_requires_holdout(self) -> None:
        protocol = {
            "candidates": {
                "short": {"decay_frames": 15},
                "long": {"decay_frames": 30},
            },
            "acceptance": {
                "minimum_development_quality_passed": 12,
                "minimum_holdout_quality_passed": 6,
                "minimum_overall_quality_passed": 18,
                "preserve_every_baseline_passing_cell": True,
            },
        }
        summaries = {
            "short": {
                "development_quality_passed": 13,
                "holdout_quality_passed": 5,
                "quality_passed": 18,
                "preserved_baseline_passing": 9,
                "baseline_passing_total": 9,
                "correction_rms_mean": 0.1,
            },
            "long": {
                "development_quality_passed": 12,
                "holdout_quality_passed": 6,
                "quality_passed": 18,
                "preserved_baseline_passing": 9,
                "baseline_passing_total": 9,
                "correction_rms_mean": 0.2,
            },
        }
        selected, decision = select_candidate(summaries, protocol)
        self.assertEqual(decision["development_ranking"], ["short", "long"])
        self.assertFalse(decision["eligible"]["short"])
        self.assertEqual(selected, "long")


class PlanarSpaceAlignedStitchTest(unittest.TestCase):
    def test_preserves_first_chunk_and_planar_step_lengths(self) -> None:
        source = motion()
        repaired, metrics = planar_space_aligned_selective_c1_stitch(
            source, boundary_frame=60, decay_frames=15
        )
        np.testing.assert_allclose(repaired[:60], source[:60])
        np.testing.assert_allclose(repaired[60, :2], source[59, :2], atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(repaired[60:, :2], axis=0), axis=1),
            np.linalg.norm(np.diff(source[60:, :2], axis=0), axis=1),
            atol=1e-6,
        )
        self.assertLess(metrics["planar_step_norm_max_error"], 1e-6)

    def test_selectively_repairs_large_joint_seam(self) -> None:
        source = motion()
        source[:, 8] = 0.1
        repaired, metrics = planar_space_aligned_selective_c1_stitch(
            source, boundary_frame=60, decay_frames=15
        )
        self.assertLess(abs(float(repaired[60, 7] - repaired[59, 7])), 1e-6)
        np.testing.assert_allclose(repaired[:, 8], source[:, 8], atol=1e-7)
        self.assertEqual(metrics["selected_joint_channels"], 1)

    def test_aligns_yaw_without_changing_second_chunk_relative_yaw(self) -> None:
        source = motion()
        repaired, _ = planar_space_aligned_selective_c1_stitch(
            source, boundary_frame=60, decay_frames=15
        )
        expected = normalize_quaternion(source[59, 3:7])
        actual = normalize_quaternion(repaired[60, 3:7])
        self.assertAlmostEqual(abs(float(np.dot(expected, actual))), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
