import unittest

from projects.text2motion_cerebellum.expanded_prompt_review import review


METRICS = ("succ", "completion", "Empjpe", "Eg_mpjpe", "foot_slide", "jerk")


def aggregate(success: float = 0.8, completion: float = 0.95) -> dict[str, float]:
    return {
        "succ": success,
        "completion": completion,
        "Empjpe": 30.0,
        "Eg_mpjpe": 100.0,
        "foot_slide": 5.0,
        "jerk": 5.0,
    }


def test_review_counts_rejected_prompts_as_end_to_end_failures() -> None:
    prompts = []
    for index in range(12):
        prompts.append(
            {
                "tag": f"p{index}",
                "source": "frozen_existing" if index < 3 else "new_generation",
                "quality_gate": "passed" if index < 9 else "rejected",
            }
        )
    protocol = {
        "generation_attempts_per_new_prompt": 1,
        "evaluation_repeats": 4,
        "selection_rule": "no rerolls; every preregistered prompt is reported",
        "thresholds": {
            "minimum_new_quality_pass_rate": 2 / 3,
            "minimum_per_seed_tracking_success": 0.75,
            "minimum_per_seed_completion": 0.9,
            "minimum_per_seed_end_to_end_success": 0.6,
        },
    }
    generation = {"prompts": prompts}
    runs = [
        (seed, {"episodes": 36, "aggregate": aggregate(), "by_prompt": {}})
        for seed in range(3)
    ]
    result = review(protocol, generation, runs)
    assert result["generation"]["new_quality_pass_rate"] == 2 / 3
    assert result["per_seed"][0]["tracking_success_on_quality_passing_prompts"] == 0.8
    assert result["per_seed"][0]["end_to_end_success_over_all_preregistered_prompts"] == 0.6
    assert result["decision"]["expanded_demo_credible"] is True


def load_tests(loader, tests, pattern):  # noqa: ARG001
    return unittest.TestSuite(
        [unittest.FunctionTestCase(test_review_counts_rejected_prompts_as_end_to_end_failures)]
    )
