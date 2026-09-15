"""Assemble three independently gated references into one tracker shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--ref-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tracker-commit", required=True)
    parser.add_argument("--gmr-commit", required=True)
    args = parser.parse_args()

    refs = []
    rows = []
    for category in args.categories:
        report = json.loads((args.work / f"{category}_retarget.json").read_text())
        if report["result"] != "passed" or report["quality_gate_reason"] != "ok":
            raise ValueError(f"{category} did not pass upstream gates")
        with np.load(args.work / f"{category}_ref.npz", allow_pickle=True) as data:
            loaded = list(data["refs"])
        if len(loaded) != 1 or str(loaded[0]["robot"]) != "g1":
            raise ValueError(f"invalid {category} reference shard")
        refs.append(loaded[0])
        rows.append({"category": category, **report})
    args.ref_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.ref_output, refs=np.array(refs, dtype=object))
    report = {
        "schema": "nano-motion-cerebellum-batch-retarget-v1",
        "result": "passed",
        "rule": "one_discarded_gmr_initialization_solve_and_unmodified_upstream_quality_gates",
        "tracker_commit": args.tracker_commit,
        "gmr_commit": args.gmr_commit,
        "references": rows,
        "reference_count": len(refs),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
