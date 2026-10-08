import json
import tempfile
import unittest
from pathlib import Path

from projects.text2motion_cerebellum.mainline_finetune_compare import load, paired


def write_episodes(path: Path, training_seed: int, clips: int = 60) -> None:
    rows = []
    for repeat in range(4):
        for index in range(clips):
            rows.append({
                "seed": training_seed,
                "repeat": repeat,
                "clip": f"clip_{index:03d}",
                "succ": True,
                "completion": 1.0,
                "Empjpe": 10.0,
                "Eg_mpjpe": 20.0,
                "foot_slide": 1.0,
                "jerk": 2.0,
            })
    path.write_text(json.dumps({
        "schema": "motion-tracking-episodes-v1",
        "episodes": rows,
    }))


class MainlineFinetuneCompareTest(unittest.TestCase):
    def test_pairs_identical_episodes_across_training_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_path = root / "before.json"
            after_path = root / "after.json"
            write_episodes(before_path, training_seed=2)
            write_episodes(after_path, training_seed=0)

            before = load(before_path, expected_clips=60, expected_training_seed=2)
            after = load(after_path, expected_clips=60, expected_training_seed=0)
            report = paired(before, after)

            self.assertEqual(report["pairs"], 240)
            self.assertEqual(report["after_minus_before"]["succ"], 0.0)

    def test_rejects_incorrect_training_seed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.json"
            write_episodes(path, training_seed=0)

            with self.assertRaisesRegex(ValueError, "expected only training seed 2"):
                load(path, expected_clips=60, expected_training_seed=2)


if __name__ == "__main__":
    unittest.main()
