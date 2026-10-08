import unittest

from projects.text2motion_cerebellum.long_horizon_sanitizer_experiment import summarize


class SanitizerSummaryTest(unittest.TestCase):
    def test_requires_overall_per_seed_and_preservation_floors(self) -> None:
        records = []
        for seed in range(3):
            for index in range(9):
                source_passed = index < 5
                passed = index < 6
                records.append(
                    {
                        "generation_seed": seed,
                        "source_quality_gate": "passed" if source_passed else "rejected",
                        "source_gate_reason": "ok" if source_passed else "joint_vel",
                        "quality_gate": "passed" if passed else "rejected",
                        "gate_reason": "ok" if passed else "speed",
                    }
                )
        protocol = {
            "generation_seeds": [0, 1, 2],
            "acceptance": {
                "minimum_quality_passed": 18,
                "minimum_per_generation_seed_quality_passed": 6,
                "preserve_every_source_passing_cell": True,
            },
        }
        result = summarize(records, protocol)
        self.assertEqual(result["quality_passed"], 18)
        self.assertEqual(result["recovered_by_source_reason"], {"joint_vel": 3})
        self.assertTrue(result["eligible_for_tracking"])

    def test_rejects_loss_of_a_source_passing_cell(self) -> None:
        records = [
            {
                "generation_seed": seed,
                "source_quality_gate": "passed",
                "source_gate_reason": "ok",
                "quality_gate": "passed",
                "gate_reason": "ok",
            }
            for seed in range(3)
            for _ in range(9)
        ]
        records[0]["quality_gate"] = "rejected"
        records[0]["gate_reason"] = "hover"
        protocol = {
            "generation_seeds": [0, 1, 2],
            "acceptance": {
                "minimum_quality_passed": 18,
                "minimum_per_generation_seed_quality_passed": 6,
                "preserve_every_source_passing_cell": True,
            },
        }
        self.assertFalse(summarize(records, protocol)["eligible_for_tracking"])


if __name__ == "__main__":
    unittest.main()
