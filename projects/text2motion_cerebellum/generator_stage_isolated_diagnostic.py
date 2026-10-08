"""Run the proven stage probe in an isolated process for each frozen OMG motion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from projects.text2motion_cerebellum.generator_stage_diagnostic import (
    FOOT_SLIDE_LIMIT_AT_50HZ,
    HOVER_LIMIT,
    source_rate_evidence,
)


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def attribution(record: dict) -> str:
    reason = record["upstream_gate"]["reason"]
    raw = record["raw_omg_30hz"]
    if source_rate_evidence(raw, reason):
        return "omg_source_motion"
    if reason == "foot_slide" and raw["foot_slide_equivalent_at_50hz"] > FOOT_SLIDE_LIMIT_AT_50HZ:
        return "omg_source_motion"
    if reason == "hover" and raw["hover"] > HOVER_LIMIT:
        return "omg_source_motion"
    if reason == "ok":
        return "no_failure"
    return "bridge_or_gate_unresolved"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-status", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--probe-script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    status = json.loads(args.prompt_status.read_text(encoding="utf-8"))
    rows = [row for row in status["prompts"] if row["source"] == "new_generation"]
    if len(rows) != 9:
        raise ValueError("expected nine frozen OMG outputs")
    work = args.out.parent / "probes"
    work.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        tag = row["tag"]
        probe_out = work / f"{tag}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(args.probe_script),
                "--source",
                str(args.generated_root / tag / "reference_motion.npz"),
                "--tracker-repo",
                str(args.tracker_repo),
                "--tag",
                tag,
                "--out",
                str(probe_out),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            write(
                args.out,
                {
                    "schema": "text2motion-generator-stage-isolated-diagnostic-v1",
                    "result": "failed",
                    "failed_tag": tag,
                    "completed_records": records,
                    "probe_checkpoint": (
                        json.loads(probe_out.read_text(encoding="utf-8"))
                        if probe_out.is_file()
                        else None
                    ),
                },
            )
            raise RuntimeError(f"isolated probe failed for {tag}")
        probe = json.loads(probe_out.read_text(encoding="utf-8"))
        record = probe["record"]
        record["attribution"] = attribution(record)
        records.append(record)
        write(
            args.out,
            {
                "schema": "text2motion-generator-stage-isolated-diagnostic-v1",
                "result": "running",
                "completed_records": records,
            },
        )

    expected = {row["tag"]: row["quality_gate"] for row in rows}
    if any(record["upstream_gate"]["result"] != expected[record["tag"]] for record in records):
        raise RuntimeError("quality-gate replay mismatch")
    write(
        args.out,
        {
            "schema": "text2motion-generator-stage-isolated-diagnostic-v1",
            "result": "passed",
            "selection_rule": "same nine frozen OMG outputs; no regeneration or repair",
            "bridge_contract": "OMG emits G1 qpos_36; bridge validates and linearly resamples 30 to 50 Hz",
            "records": records,
            "counts": {
                "passed": sum(row["upstream_gate"]["result"] == "passed" for row in records),
                "rejected": sum(row["upstream_gate"]["result"] == "rejected" for row in records),
                "attribution": dict(sorted(Counter(row["attribution"] for row in records).items())),
            },
        },
    )


if __name__ == "__main__":
    main()
