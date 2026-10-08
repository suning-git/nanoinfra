import unittest

from projects.text2motion_cerebellum.long_horizon_domain_adaptation_review import review


def episodes(seed: int, clips: int, *, succ: float, completion: float):
    rows = []
    for repeat in range(4):
        for clip in range(clips):
            rows.append(
                {
                    "seed": seed,
                    "repeat": repeat,
                    "clip": f"clip{clip}",
                    "succ": succ,
                    "completion": completion,
                    "Empjpe": 1.0,
                    "Eg_mpjpe": 1.0,
                    "foot_slide": 1.0,
                    "jerk": 1.0,
                }
            )
    return rows


class DomainAdaptationReviewTest(unittest.TestCase):
    def test_expands_only_on_heldout_gain_and_native_preservation(self) -> None:
        result = review(
            episodes(0, 60, succ=0.85, completion=0.95),
            episodes(0, 60, succ=0.82, completion=0.94),
            episodes(0, 6, succ=0.58, completion=0.84),
            episodes(0, 6, succ=0.70, completion=0.91),
            training_references=132,
            native_replay_references=120,
        )
        self.assertTrue(result["expand_to_three_policy_finetune"])
        self.assertEqual(result["protocol"]["native_replay_references"], 120)

    def test_rejects_native_regression(self) -> None:
        result = review(
            episodes(0, 60, succ=0.85, completion=0.95),
            episodes(0, 60, succ=0.70, completion=0.80),
            episodes(0, 6, succ=0.58, completion=0.84),
            episodes(0, 6, succ=0.75, completion=0.95),
        )
        self.assertFalse(result["expand_to_three_policy_finetune"])


if __name__ == "__main__":
    unittest.main()
