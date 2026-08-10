"""Compare the frozen expanded-prompt baseline with reference repair v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--repair-status", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    protocol = load(args.protocol)
    baseline = load(args.baseline)
    repaired = load(args.repaired)
    status = load(args.repair_status)
    for label, payload in (("baseline", baseline), ("repaired", repaired)):
        if payload.get("schema") != "text2motion-expanded-prompt-review-v1":
            raise ValueError(f"unexpected {label} summary schema")
    expected_run_schema = protocol["schema"].replace("-protocol-", "-run-")
    if status.get("repair_schema") != expected_run_schema:
        raise ValueError("unexpected repair status schema")
    if baseline["protocol"] != repaired["protocol"]:
        raise ValueError("baseline and repaired evaluation protocols differ")

    new_records = [row for row in status["prompts"] if row["source"] == "new_generation"]
    original_records = [row for row in status["prompts"] if row["source"] == "frozen_existing"]
    if len(new_records) != 9 or len(original_records) != 3:
        raise ValueError("repair status does not contain the frozen 3+9 prompt split")
    if len({row["source_sha256"] for row in new_records}) != 9:
        raise ValueError("new source hashes are missing or duplicated")

    baseline_quality = int(baseline["generation"]["new_quality_passed"])
    repaired_quality = sum(row["quality_gate"] == "passed" for row in new_records)
    baseline_e2e = float(baseline["end_to_end_success_across_training_seeds"]["mean"])
    repaired_e2e = float(repaired["end_to_end_success_across_training_seeds"]["mean"])
    acceptance = protocol["acceptance"]
    criteria = {
        "minimum_new_quality_passed": repaired_quality >= int(
            acceptance["minimum_new_quality_passed"]
        ),
        "original_three_preserved": all(
            row["quality_gate"] == "passed" and not row["repair"]["applied"]
            for row in original_records
        ),
        "no_regeneration_or_rerolls": all(
            row["generation"] == "passed" and "source_sha256" in row
            for row in new_records
        ),
        "all_seeds_tracking_floor": bool(
            repaired["decision"]["all_seeds_track_quality_passing_prompts"]
        ),
        "mean_end_to_end_improved": repaired_e2e > baseline_e2e,
    }
    payload = {
        "schema": protocol["schema"].replace("-protocol-", "-comparison-"),
        "quality_gate": {
            "baseline_new_passed": baseline_quality,
            "repaired_new_passed": repaired_quality,
            "total_new": 9,
        },
        "tracking_success_mean": {
            "baseline": baseline["tracking_across_training_seeds"]["succ"]["mean"],
            "repaired": repaired["tracking_across_training_seeds"]["succ"]["mean"],
        },
        "end_to_end_success_mean": {
            "baseline": baseline_e2e,
            "repaired": repaired_e2e,
            "absolute_delta": repaired_e2e - baseline_e2e,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
