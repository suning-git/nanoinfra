import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def tracker_payload(success: float, completion: float, mpjpe: float) -> dict:
    return {
        "protocol": {"training_seeds": [0, 1, 2]},
        "across_training_seeds": {
            "succ": {"mean": success},
            "completion": {"mean": completion},
            "Empjpe": {"mean": mpjpe},
            "foot_slide": {"mean": 6.0},
            "jerk": {"mean": 5.0},
        },
        "per_seed": [{"episodes": 12} for _ in range(3)],
    }


class SelfTrainedComparisonTests(unittest.TestCase):
    def test_builds_normalized_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            current = tmp_path / "current.json"
            old = tmp_path / "old.json"
            omg = tmp_path / "omg.json"
            diagnostic = tmp_path / "diagnostic.json"
            selection = tmp_path / "selection.json"
            output = tmp_path / "comparison.json"
            current.write_text(json.dumps(tracker_payload(30 / 36, 0.97, 49.0)))
            old.write_text(json.dumps(tracker_payload(34 / 36, 0.99, 46.0)))
            omg.write_text(json.dumps({
                "across_training_seeds": {"omg": {
                    "success": {"mean": 34 / 36},
                    "completion": {"mean": 1.0},
                    "Empjpe_mm": {"mean": 29.0},
                    "foot_slide": {"mean": 5.0},
                    "jerk": {"mean": 4.0},
                }}
            }))
            diagnostic.write_text(json.dumps({"per_seed": [{
                "aggregate": {"succ": 0.6},
                "by_prompt": {"left": {"succ": 0.4}},
            }]}))
            selection.write_text(json.dumps({
                "post_hoc": True,
                "selection_rule": {
                    "uses_tracker_outcomes_for_candidate_ranking": False
                },
                "selected": {"seed": 16},
            }))
            script = Path(__file__).with_name("compare_self_trained_v2.py")
            subprocess.run([
                sys.executable, str(script),
                "--self-full", str(current),
                "--old-nano", str(old),
                "--omg", str(omg),
                "--seed1-diagnostic", str(diagnostic),
                "--selection", str(selection),
                "--output", str(output),
            ], check=True)
            report = json.loads(output.read_text())
            self.assertEqual(
                report["formal"]["self_trained_nano_motion"]["successes"], 30
            )
            self.assertEqual(report["formal"]["omg"]["successes"], 34)
            self.assertFalse(report["reference_selection"][
                "uses_tracker_outcomes_for_candidate_ranking"
            ])


if __name__ == "__main__":
    unittest.main()
