"""Validate and summarize native-vs-Text2Motion tracking evaluations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


METRICS = ("succ", "completion", "Empjpe", "Eg_mpjpe", "foot_slide", "jerk")


def load_rows(paths: Iterable[Path], domain: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "motion-tracking-episodes-v1":
            raise ValueError(f"{path}: unexpected episode schema")
        for row in payload.get("episodes", []):
            item = dict(row)
            item["domain"] = domain
            rows.append(item)
    return rows


def validate(rows: list[dict[str, Any]], expected_clips: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation contains no episode rows")
    keys = [(int(r["seed"]), int(r["repeat"]), str(r["clip"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (seed, repeat, clip) evaluation rows")
    seeds = sorted({key[0] for key in keys})
    repeats = sorted({key[1] for key in keys})
    if seeds != [0, 1, 2] or repeats != [0, 1, 2, 3]:
        raise ValueError(f"expected seeds 0..2 and repeats 0..3, got {seeds}/{repeats}")
    grouped: dict[tuple[int, int], set[str]] = defaultdict(set)
    for seed, repeat, clip in keys:
        grouped[(seed, repeat)].add(clip)
    bad = {
        f"s{seed}_r{repeat}": len(clips)
        for (seed, repeat), clips in grouped.items()
        if len(clips) != expected_clips
    }
    if bad:
        raise ValueError(f"clip count mismatch: {bad}")
    return {
        "rows": len(rows),
        "seeds": seeds,
        "repeats": repeats,
        "clips_per_seed_repeat": expected_clips,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: fmean(float(row[metric]) for row in rows) for metric in METRICS}


def grouped_aggregates(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: aggregate(value) for key, value in sorted(groups.items())}


def delta(omg: dict[str, float], native: dict[str, float]) -> dict[str, float]:
    return {metric: omg[metric] - native[metric] for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, nargs=3, required=True)
    parser.add_argument("--omg", type=Path, nargs=3, required=True)
    parser.add_argument("--tracker-commit", required=True)
    parser.add_argument("--model-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    native_rows = load_rows(args.native, "native")
    omg_rows = load_rows(args.omg, "omg")
    native_contract = validate(native_rows, 60)
    omg_contract = validate(omg_rows, 3)
    native = aggregate(native_rows)
    omg = aggregate(omg_rows)

    # This threshold is deliberately below the repository's reported 90.8--95%
    # range.  If it fails, OMG performance cannot diagnose the adapter because
    # the tracker is not yet a credible reproduction on its own distribution.
    credible_native = native["succ"] >= 0.80 and native["completion"] >= 0.90
    demo_ready = omg["succ"] >= 2 / 3 and omg["completion"] >= 0.85
    if not credible_native:
        route = "strengthen_original_tracker"
    elif demo_ready:
        route = "render_text2motion_demo"
    else:
        route = "adapt_original_tracker_to_text2motion_refs"

    report = {
        "schema": "text2motion-cerebellum-mainline-evaluation-v1",
        "contract": {"native": native_contract, "omg": omg_contract},
        "run": {
            "tracker_commit": args.tracker_commit,
            "model_commit": args.model_commit,
            "training_seeds": [0, 1, 2],
            "observation_noise": True,
            "repeats": 4,
            "native_clips": 60,
            "omg_prompts": 3,
        },
        "aggregate": {"native": native, "omg": omg},
        "omg_minus_native": delta(omg, native),
        "by_seed": {
            "native": grouped_aggregates(native_rows, "seed"),
            "omg": grouped_aggregates(omg_rows, "seed"),
        },
        "by_prompt": grouped_aggregates(omg_rows, "clip"),
        "decision": {
            "credible_native_baseline": credible_native,
            "text2motion_demo_ready": demo_ready,
            "next_route": route,
            "native_success_floor": 0.80,
            "native_completion_floor": 0.90,
            "demo_success_floor": 2 / 3,
            "demo_completion_floor": 0.85,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
