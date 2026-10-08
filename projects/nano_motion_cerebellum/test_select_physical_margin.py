import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PhysicalMarginSelectionTests(unittest.TestCase):
    def test_selects_lowest_worst_gate_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            sweep = {
                "prompt": "turn left",
                "seeds": 3,
                "passing_seeds": 2,
                "candidates": [
                    {"seed": 1, "stem": "seed_0001", "directional_gate": True,
                     "score": 5.0},
                    {"seed": 2, "stem": "seed_0002", "directional_gate": True,
                     "score": 4.0},
                    {"seed": 3, "stem": "seed_0003", "directional_gate": False,
                     "score": 6.0},
                ],
            }
            sweep_path = tmp_path / "sweep.json"
            sweep_path.write_text(json.dumps(sweep))
            attempts = tmp_path / "attempts"
            attempts.mkdir()
            for seed, slide, hover, speed in (
                (1, 10.0, 10.0, 10.0), (2, 4.0, 15.0, 6.0)
            ):
                (attempts / f"seed_{seed:04d}_retarget.json").write_text(json.dumps({
                    "result": "passed",
                    "quality_gate_reason": "ok",
                    "foot_slide": slide,
                    "hover": hover,
                    "max_joint_speed_rad_s": speed,
                }))
            output = tmp_path / "selection.json"
            script = Path(__file__).with_name("select_physical_margin.py")
            subprocess.run([
                sys.executable,
                str(script),
                "--sweep-report", str(sweep_path),
                "--attempts", str(attempts),
                "--output", str(output),
            ], check=True)
            report = json.loads(output.read_text())
            self.assertEqual(report["result"], "passed")
            self.assertEqual(report["selected"]["seed"], 2)
            self.assertFalse(report["selection_rule"][
                "uses_tracker_outcomes_for_candidate_ranking"
            ])


if __name__ == "__main__":
    unittest.main()
