import unittest

from projects.text2motion_cerebellum.multiseed_review import METRICS
from projects.text2motion_cerebellum.single_chunk_tracking_experiment import review


def episode(seed, repeat, clip, success=True):
    row = {metric: 1.0 for metric in METRICS}
    row.update(
        {
            "seed": seed,
            "repeat": repeat,
            "clip": clip,
            "succ": success,
            "completion": 1.0 if success else 0.5,
        }
    )
    return row


class SingleChunkTrackingExperimentTest(unittest.TestCase):
    def test_rejected_generation_cells_count_as_end_to_end_failures(self):
        inventory = []
        for generation_seed in range(3):
            for prompt_index in range(9):
                passed = prompt_index < 7
                inventory.append(
                    {
                        "generation_seed": generation_seed,
                        "tag": f"p{prompt_index}",
                        "clip": f"gseed{generation_seed}__p{prompt_index}",
                        "quality_gate": "passed" if passed else "rejected",
                        "gate_reason": "ok" if passed else "speed",
                    }
                )
        accepted = [row["clip"] for row in inventory if row["quality_gate"] == "passed"]
        runs = []
        for training_seed in range(3):
            rows = [
                episode(training_seed, repeat, clip)
                for repeat in range(4)
                for clip in accepted
            ]
            runs.append((training_seed, rows))
        result = review(inventory, runs)
        self.assertEqual(result["protocol"]["quality_passing_cells"], 21)
        self.assertAlmostEqual(
            result["per_training_seed"][0]["end_to_end_success_over_27_generation_cells"],
            21 / 27,
        )
        self.assertTrue(result["decision"]["short_horizon_demo_credible"])


if __name__ == "__main__":
    unittest.main()
