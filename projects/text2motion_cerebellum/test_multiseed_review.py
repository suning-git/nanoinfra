import json
import tempfile
import unittest
from pathlib import Path

from projects.text2motion_cerebellum.multiseed_review import across_seeds, keys, load_suite


def write_suite(path: Path, training_seed: int, clips: int) -> None:
    rows = []
    for repeat in range(4):
        for index in range(clips):
            rows.append({
                "seed": training_seed,
                "repeat": repeat,
                "clip": f"clip_{index:03d}",
                "succ": float(training_seed) / 2,
                "completion": 1.0,
                "Empjpe": 10.0 + training_seed,
                "Eg_mpjpe": 20.0,
                "foot_slide": 1.0,
                "jerk": 2.0,
            })
    path.write_text(json.dumps({
        "schema": "motion-tracking-episodes-v1",
        "episodes": rows,
    }))


class MultiseedReviewTest(unittest.TestCase):
    def test_loads_matching_fixed_protocol_suites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suites = []
            for seed in range(3):
                path = root / f"seed_{seed}.json"
                write_suite(path, seed, clips=60)
                suites.append(load_suite(path, seed, expected_clips=60))
            self.assertTrue(keys(suites[0]) == keys(suites[1]) == keys(suites[2]))

    def test_reports_training_seed_variation(self) -> None:
        rows = []
        for seed in range(3):
            rows.append({
                "succ": float(seed) / 2,
                "completion": 1.0,
                "Empjpe": 10.0 + seed,
                "Eg_mpjpe": 20.0,
                "foot_slide": 1.0,
                "jerk": 2.0,
            })
        report = across_seeds(rows)
        self.assertEqual(report["succ"]["mean"], 0.5)
        self.assertEqual(report["succ"]["sample_std"], 0.5)
        self.assertEqual(report["Empjpe"]["mean"], 11.0)


if __name__ == "__main__":
    unittest.main()
