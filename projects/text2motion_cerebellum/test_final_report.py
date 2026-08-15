"""Clean-checkout checks for the submission reports and compact results."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects/text2motion_cerebellum"
REPORT = PROJECT / "FINAL_REPORT_CN.md"
RESULTS = PROJECT / "RESULTS.md"
MAIN = PROJECT / "results/main_results.json"
EXPANDED = PROJECT / "results/expanded_prompt_results.json"
QUALITY = PROJECT / "results/prompt_quality_diagnostics.json"
REPAIR = PROJECT / "results/reference_repair_results.json"
GENERATOR_DIAGNOSIS = PROJECT / "results/generator_diagnosis_results.json"
SHORT_HORIZON = PROJECT / "results/short_horizon_results.json"
LONG_HORIZON = PROJECT / "results/long_horizon_results.json"
SUBMISSION_FILES = PROJECT / "submission_files.txt"
NANO_MOTION_CONFIG = ROOT / "exemplars/nano_motion/configs/train_t2m.yaml"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_tracked_main_result_matches_frozen_protocol_and_reports() -> None:
    payload = load(MAIN)
    protocol = payload["protocol"]
    report = REPORT.read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")

    assert payload["schema"] == "text2motion-cerebellum-submission-results-v1"
    assert protocol["training_seeds"] == [0, 1, 2]
    assert protocol["clean_start"] is True
    assert protocol["training_clips"] == 1500
    assert protocol["iterations"] == 4500
    assert protocol["environments"] == 3840
    assert protocol["workers"] == 80
    assert protocol["evaluation_repeats"] == 4
    assert protocol["native_episodes_per_seed"] == 240
    assert protocol["omg_episodes_per_seed"] == 12

    native = payload["across_training_seeds"]["native"]
    omg = payload["across_training_seeds"]["omg"]
    assert math.isclose(native["success"]["mean"], 0.9166666666666666)
    assert math.isclose(native["success"]["sample_std"], 0.00833333333333336)
    assert math.isclose(native["Empjpe_mm"]["mean"], 29.313530217961617)
    assert math.isclose(omg["success"]["mean"], 0.9444444444444443)
    assert math.isclose(omg["success"]["sample_std"], 0.048112522432468836)
    assert math.isclose(omg["Empjpe_mm"]["mean"], 28.52018143466093)
    assert payload["decision"] == {
        "all_native_seeds_pass_frozen_floor": True,
        "all_seeds_pass_three_prompt_demo_floor": True,
    }

    for text in (report, results):
        assert "91.67" in text and "0.83" in text
        assert "94.44" in text and "4.81" in text
        assert "29.31" in text and "28.52" in text
    assert "34/36 episodes" in report
    assert protocol["tracker_commit"] in report
    assert protocol["model_commit"] in report


def test_tracked_expanded_prompt_result_matches_report() -> None:
    payload = load(EXPANDED)
    report = REPORT.read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")

    assert payload["protocol"]["total_preregistered_prompts"] == 12
    assert payload["protocol"]["new_generation_prompts"] == 9
    assert payload["protocol"]["generation_attempts_per_new_prompt"] == 1
    assert payload["generation"]["new_quality_passed"] == 3
    assert payload["generation"]["new_quality_pass_rate"] == 1 / 3
    assert math.isclose(
        payload["across_training_seeds"]["tracking_success"]["mean"],
        0.8194444444444445,
    )
    assert math.isclose(
        payload["across_training_seeds"]["end_to_end_success"]["mean"],
        0.40972222222222227,
    )
    assert payload["decision"] == {
        "new_prompt_quality_gate_credible": False,
        "all_seeds_track_quality_passing_prompts": True,
        "all_seeds_end_to_end_credible": False,
        "expanded_demo_credible": False,
    }
    for text in (report, results):
        assert "81.94" in text
        assert "40.97" in text
        assert "3 / 9" in text or "3/9" in text


def test_tracked_quality_diagnostics_match_report() -> None:
    payload = load(QUALITY)
    report = REPORT.read_text(encoding="utf-8")
    assert payload["passed"] == 3
    assert payload["rejected"] == 6
    assert payload["reason_counts"] == {
        "continuity": 2,
        "foot_slide": 1,
        "joint_vel": 1,
        "speed": 2,
    }
    records = {item["tag"]: item for item in payload["rejections"]}
    assert round(records["sidestep_left"]["root_speed_max_m_s"], 2) == 2.06
    assert round(records["jog_forward"]["root_speed_max_m_s"], 2) == 6.13
    assert round(records["squat"]["joint_speed_max_rad_s"], 2) == 15.69
    assert round(records["raise_both_arms"]["joint_step_max_rad_frame"], 2) == 1.70
    assert "6.13 m/s > 2.0 m/s" in report
    assert "1.70 rad/frame > 0.5" in report


def test_post_hoc_reference_repair_matches_report() -> None:
    payload = load(REPAIR)
    report = REPORT.read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")

    assert payload["phase"] == "post_hoc_followup_after_preregistered_stress_test"
    assert payload["protocol_invariants"]["generation_rerolls"] == 0
    assert payload["repair_v2"]["new_quality_passed"] == 6
    assert math.isclose(
        payload["repair_v2"]["end_to_end_success_over_all_preregistered_prompts"],
        0.5694444444444444,
    )
    assert payload["repair_v2"]["continuity_prompt_tracking_success"]["kick_right_leg"] == [0.0, 0.0, 0.0]
    assert payload["repair_v2"]["acceptance_passed"] is False
    for text in (report, results):
        assert "56.94" in text
        assert "6/9" in text
        assert "事后" in text or "post-hoc" in text

    for source in payload["provenance"].values():
        path = ROOT / source["source_artifact"]
        assert len(source["source_sha256"]) == 64
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["source_sha256"]


def test_generator_stage_diagnosis_matches_reports() -> None:
    payload = load(GENERATOR_DIAGNOSIS)
    report = REPORT.read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")

    assert payload["phase"] == "post_hoc_generator_stage_attribution"
    assert payload["bridge_contract"]["smpl_to_g1_retargeting"] is False
    frozen = payload["frozen_output_attribution"]
    assert frozen["quality_passed"] == 3
    assert frozen["quality_rejected"] == 6
    assert frozen["attribution_counts"] == {
        "no_failure": 3,
        "omg_source_motion": 6,
    }
    assert all(row["raw_source_already_exceeds_gate"] for row in frozen["rejections"])

    variants = payload["documented_chunk_experiment"]["variants"]
    single = variants["documented_single_chunk_60"]
    double = variants["documented_two_chunks_120"]
    assert (single["quality_passed"], double["quality_passed"]) == (6, 3)
    assert (single["midpoint_worst_step_count"], double["midpoint_worst_step_count"]) == (0, 8)
    assert payload["decision"]["tracker_retraining_indicated_by_this_diagnostic"] is False
    assert payload["decision"]["whole_text_to_motion_retraining_is_the_first_action"] is False

    for text in (report, results):
        assert "6/9" in text and "3/9" in text
        assert "59→60" in text
        assert "SMPL" in text and "G1" in text

    for section in ("frozen_output_attribution", "documented_chunk_experiment"):
        source = payload[section]["provenance"]
        path = ROOT / source["source_artifact"]
        assert len(source["source_sha256"]) == 64
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["source_sha256"]


def test_short_horizon_followup_matches_reports() -> None:
    payload = load(SHORT_HORIZON)
    report = REPORT.read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")

    assert payload["classification"] == "post_hoc_short_horizon_followup"
    comparison = payload["single_vs_double_multiseed"]
    assert comparison["single_60"]["quality_passed"] == 21
    assert comparison["baseline_two_chunks"]["quality_passed"] == 9
    assert comparison["paired_cells"] == {
        "both_pass": 9,
        "single_only_pass": 12,
        "double_only_pass": 0,
        "both_fail": 6,
    }
    tracking = payload["closed_loop_tracking"]
    assert [row["tracking_success"] for row in tracking["per_training_seed"]] == [
        0.8214285714285714,
        0.8095238095238095,
        0.7857142857142857,
    ]
    assert math.isclose(
        tracking["across_training_seeds"]["end_to_end_success"]["mean"],
        0.6265432098765432,
    )
    assert tracking["decision"]["short_horizon_demo_credible_under_overall_protocol"] is True
    assert tracking["decision"]["robust_to_generation_seed"] is False
    assert tracking["decision"]["long_horizon_demo_credible"] is False

    for text in (report, results):
        assert "21/27" in text and "9/27" in text
        assert "80.56" in text and "62.65" in text
        assert "短" in text or "short-horizon" in text

    sources = [
        payload["continuation_preflight"]["provenance"],
        payload["single_vs_double_multiseed"]["provenance"],
        tracking["provenance"]["inventory"],
        tracking["provenance"]["summary"],
        *tracking["provenance"]["episode_files"],
    ]
    for source in sources:
        path = ROOT / source["source_artifact"]
        assert len(source["source_sha256"]) == 64
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["source_sha256"]


def test_long_horizon_followup_matches_reports() -> None:
    payload = load(LONG_HORIZON)
    report = REPORT.read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")

    assert payload["schema"] == "text2motion-long-horizon-results-v1"
    quality = payload["quality_pipeline"]
    assert quality["raw_two_chunk"]["quality_passed"] == 9
    assert quality["unconstrained_c1_residual"]["quality_passed"] == 14
    assert quality["planar_space_aligned_selective_c1"]["quality_passed"] == 15
    assert quality["reason_specific_adapter_sanitization"]["quality_passed"] == 18
    assert quality["reason_specific_adapter_sanitization"][
        "per_generation_seed_quality_passed"
    ] == [6, 6, 6]

    tracking = payload["closed_loop_tracking"]
    assert [row["tracking_success"] for row in tracking["per_training_seed"]] == [
        0.6527777777777778,
        0.7222222222222222,
        0.7361111111111112,
    ]
    assert math.isclose(
        tracking["across_training_seeds"]["tracking_success"]["mean"],
        0.7037037037037037,
    )
    assert math.isclose(
        tracking["across_training_seeds"]["end_to_end_success"]["mean"],
        0.46913580246913583,
    )
    assert tracking["decision"]["long_horizon_sanitized_demo_credible"] is False
    assert payload["domain_adaptation_smokes"][
        "long_only_12_refs_plus_300_iterations"
    ]["expand_to_three_policies"] is False
    assert payload["domain_adaptation_smokes"][
        "long_12_plus_native_replay_120_plus_300_iterations"
    ]["expand_to_three_policies"] is False

    for text in (report, results):
        assert "18/27" in text and "70.37" in text and "46.91" in text
        assert "91.67" in text and "73.33" in text and "76.25" in text

    provenance = payload["provenance"]
    sources = [
        provenance["raw_two_chunk_baseline"],
        provenance["unconstrained_c1"],
        provenance["space_aligned_c1"],
        provenance["sanitizer"],
        provenance["tracking"]["inventory"],
        provenance["tracking"]["summary"],
        *provenance["tracking"]["episode_files"],
        provenance["long_only_adaptation"],
        provenance["replay_adaptation"],
    ]
    for source in sources:
        path = ROOT / source["source_artifact"]
        assert len(source["source_sha256"]) == 64
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["source_sha256"]


def test_compact_results_retain_source_provenance() -> None:
    for path in (MAIN, EXPANDED, QUALITY):
        provenance = load(path)["provenance"]
        digest = provenance["source_sha256"]
        assert len(digest) == 64
        int(digest, 16)
        source = ROOT / provenance["source_artifact"]
        if source.is_file():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_public_submission_docs_do_not_contain_machine_credentials() -> None:
    docs = [
        PROJECT / "README.md",
        PROJECT / "RESULTS.md",
        PROJECT / "FINAL_REPORT_CN.md",
        PROJECT / "DATA_AND_LICENSES.md",
        PROJECT / "SUBMISSION.md",
        ROOT / "projects/nano_motion_cerebellum/README.md",
        ROOT / "projects/nano_motion_cerebellum/DATA_AND_LICENSES.md",
        ROOT / "projects/nano_motion_motionhub/README.md",
    ]
    forbidden = (
        "BEGIN " + "OPENSSH PRIVATE KEY",
        "AUTODL" + "_TOKEN=",
        "HF" + "_TOKEN=",
        "root@" + "connect.",
        "connect." + "bjb",
        "/root/" + "autodl-fs/",
        "/root/" + "autodl-tmp/",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    assert not any(value in combined for value in forbidden)
    assert "BONES-SEED" in combined and "SEED license" in combined


def test_nano_motion_t2m_recipe_uses_valid_named_supervision() -> None:
    config = NANO_MOTION_CONFIG.read_text(encoding="utf-8")
    assert "supervise: [motion_tokens, motion_end]" in config
    assert "supervise_tags:" not in config


def test_submission_allow_list_is_complete_and_scoped() -> None:
    entries = [
        line.strip()
        for line in SUBMISSION_FILES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == len(set(entries))
    assert entries == sorted(entries)
    assert all(
        path == "README.md"
        or path == "exemplars/nano_motion/configs/train_t2m.yaml"
        or path.startswith("projects/text2motion_cerebellum/")
        or path.startswith("projects/nano_motion_cerebellum/")
        or path.startswith("projects/nano_motion_motionhub/")
        for path in entries
    )
    assert all((ROOT / path).is_file() for path in entries)
    assert "README.md" in entries
    assert "projects/text2motion_cerebellum/results/main_results.json" in entries
    assert not any(path.startswith("remote/") or path.startswith("outputs/") for path in entries)
    media = {
        path
        for path in entries
        if Path(path).suffix.lower() in {".mp4", ".webm", ".png"}
    }
    assert media == {
        "projects/text2motion_cerebellum/assets/demo/turn-left.mp4",
        "projects/text2motion_cerebellum/assets/demo/turn-left.png",
        "projects/text2motion_cerebellum/assets/demo/turn-right.mp4",
        "projects/text2motion_cerebellum/assets/demo/turn-right.png",
        "projects/text2motion_cerebellum/assets/demo/walk-forward.mp4",
        "projects/text2motion_cerebellum/assets/demo/walk-forward.png",
        "projects/nano_motion_cerebellum/assets/demo/self_motion_nano-motion-self-forward.mp4",
        "projects/nano_motion_cerebellum/assets/demo/self_motion_nano-motion-self-forward.png",
        "projects/nano_motion_cerebellum/assets/demo/self_motion_nano-motion-self-left.mp4",
        "projects/nano_motion_cerebellum/assets/demo/self_motion_nano-motion-self-left.png",
        "projects/nano_motion_cerebellum/assets/demo/self_motion_nano-motion-self-right.mp4",
        "projects/nano_motion_cerebellum/assets/demo/self_motion_nano-motion-self-right.png",
    }


def load_tests(loader, tests, pattern):  # noqa: ARG001
    """Expose the dependency-free assertion functions to unittest discovery."""
    suite = unittest.TestSuite()
    for function in (
        test_tracked_main_result_matches_frozen_protocol_and_reports,
        test_tracked_expanded_prompt_result_matches_report,
        test_tracked_quality_diagnostics_match_report,
        test_post_hoc_reference_repair_matches_report,
        test_generator_stage_diagnosis_matches_reports,
        test_short_horizon_followup_matches_reports,
        test_long_horizon_followup_matches_reports,
        test_compact_results_retain_source_provenance,
        test_public_submission_docs_do_not_contain_machine_credentials,
        test_nano_motion_t2m_recipe_uses_valid_named_supervision,
        test_submission_allow_list_is_complete_and_scoped,
    ):
        suite.addTest(unittest.FunctionTestCase(function))
    return suite
