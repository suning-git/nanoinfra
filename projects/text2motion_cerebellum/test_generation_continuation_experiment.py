import unittest

import numpy as np

from projects.text2motion_cerebellum.generation_continuation_experiment import (
    stitch_arrays,
    summarize,
)


class GenerationContinuationExperimentTest(unittest.TestCase):
    def test_stitch_arrays_removes_the_constrained_overlap(self):
        first = np.arange(12, dtype=np.float32).reshape(6, 2)
        second = np.arange(100, 112, dtype=np.float32).reshape(6, 2)
        stitched = stitch_arrays(first, second, overlap=2)
        np.testing.assert_array_equal(stitched[:6], first)
        np.testing.assert_array_equal(stitched[6:], second[2:])
        self.assertEqual(stitched.shape, (10, 2))

    def test_stitch_arrays_rejects_invalid_overlap(self):
        value = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            stitch_arrays(value, value, overlap=2)

    def test_summary_counts_gate_and_seam_outcomes(self):
        records = [
            {
                "variant": "candidate",
                "generation": "passed",
                "quality_gate": "passed",
                "gate_reason": "ok",
                "raw_omg_30hz": {
                    "joint_step_worst_transition": [59, 60],
                    "joint_step_max_rad_frame": 0.2,
                },
            },
            {
                "variant": "candidate",
                "generation": "passed",
                "quality_gate": "rejected",
                "gate_reason": "speed",
                "raw_omg_30hz": {
                    "joint_step_worst_transition": [12, 13],
                    "joint_step_max_rad_frame": 0.1,
                },
            },
        ]
        summary = summarize(records)["candidate"]
        self.assertEqual(summary["quality_passed"], 1)
        self.assertEqual(summary["reason_counts"], {"speed": 1})
        self.assertEqual(summary["seam_is_worst_transition_count"], 1)


if __name__ == "__main__":
    unittest.main()
