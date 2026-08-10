import unittest

from projects.text2motion_cerebellum.long_horizon_tracking_experiment import review


METRICS = {
    "succ": 0.95,
    "completion": 0.96,
    "Empjpe": 0.1,
    "Eg_mpjpe": 0.1,
    "foot_slide": 1.0,
    "jerk": 1.0,
}


class LongHorizonTrackingReviewTest(unittest.TestCase):
    def test_credible_result_requires_all_three_trackers(self) -> None:
        inventory = []
        for generation_seed in range(3):
            for index in range(9):
                passed = index < 6
                inventory.append(
                    {
                        "generation_seed": generation_seed,
                        "tag": f"tag{index}",
                        "clip": f"g{generation_seed}_{index}",
                        "quality_gate": "passed" if passed else "rejected",
                        "gate_reason": "ok" if passed else "speed",
                    }
                )
        runs = []
        accepted = [row for row in inventory if row["quality_gate"] == "passed"]
        for training_seed in range(3):
            rows = []
            for repeat in range(4):
                for item in accepted:
                    rows.append(
                        {
                            "seed": training_seed,
                            "repeat": repeat,
                            "clip": item["clip"],
                            **METRICS,
                        }
                    )
            runs.append((training_seed, rows))
        result = review(inventory, runs)
        self.assertEqual(result["protocol"]["quality_passing_cells"], 18)
        self.assertAlmostEqual(
            result["end_to_end_success_across_training_seeds"]["mean"],
            18 * 0.95 / 27,
        )
        self.assertTrue(result["decision"]["long_horizon_sanitized_demo_credible"])

    def test_rejects_wrong_accepted_count(self) -> None:
        inventory = [
            {
                "generation_seed": 0,
                "tag": str(index),
                "clip": str(index),
                "quality_gate": "passed",
                "gate_reason": "ok",
            }
            for index in range(27)
        ]
        with self.assertRaises(ValueError):
            review(inventory, [])


if __name__ == "__main__":
    unittest.main()
