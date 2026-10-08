"""Review one staged full-scale nano_motion training run.

Only compact, structured evidence is emitted.  Raw training logs and licensed
artifacts remain on remote persistent storage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from report import parse_log


def latest_checkpoint(root: Path) -> tuple[Path, dict]:
    candidates = sorted(root.glob("step_*"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint under {root}")
    path = candidates[-1]
    return path, json.loads((path / "meta.json").read_text())


def sweep_summary(path: Path) -> dict:
    payload = json.loads(path.read_text())
    candidates = payload.get("candidates", [])
    return {
        "schema": payload.get("schema"),
        "prompt": payload.get("prompt"),
        "generated": len(candidates),
        "passing": int(payload.get("passing_seeds", 0)),
        "result": payload.get("result"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--prepare-report", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prepare = json.loads(args.prepare_report.read_text())
    curves = parse_log(args.log)
    checkpoint_path, checkpoint_meta = latest_checkpoint(args.checkpoints)
    validation = curves["validation"]
    training = curves["train"]
    sweeps = {
        "forward": sweep_summary(args.forward),
        "left": sweep_summary(args.left),
        "right": sweep_summary(args.right),
    }
    finite_train = bool(training) and all(math.isfinite(row["loss"]) for row in training)
    finite_validation = bool(validation) and all(
        math.isfinite(row["motion_ce"]) for row in validation
    )
    best_validation = min(
        (row["motion_ce"] for row in validation), default=math.inf
    )
    final_validation = validation[-1]["motion_ce"] if validation else math.inf
    prior = json.loads(args.prior.read_text()) if args.prior else None
    validation_non_regression = (
        prior is None
        or best_validation <= float(prior["training"]["best_validation_ce"]) * 1.02
    )
    gates = {
        "full_cache_quality": prepare.get("result") == "passed",
        "checkpoint_at_expected_step": int(checkpoint_meta["step"]) >= args.expected_step,
        "finite_train_curve": finite_train,
        "train_loss_decreased_within_stage": (
            finite_train and training[-1]["loss"] < training[0]["loss"]
        ),
        "finite_validation": finite_validation,
        "validation_better_than_untrained": best_validation < 8.0,
        "final_validation_close_to_stage_best": final_validation <= best_validation * 1.15,
        "validation_non_regression": validation_non_regression,
        "all_fixed_seed_sweeps_complete": all(
            row["generated"] == args.expected_seeds for row in sweeps.values()
        ),
    }
    total_generated = sum(row["generated"] for row in sweeps.values())
    total_passing = sum(row["passing"] for row in sweeps.values())
    report = {
        "schema": "nano-motion-motionhub-scale-stage-v1",
        "stage": args.stage,
        "result": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "data": {
            "train_clips": prepare["splits"]["train"]["clips"],
            "train_pairs": prepare["splits"]["train"]["pairs"],
            "val_clips": prepare["splits"]["val"]["clips"],
            "val_pairs": prepare["splits"]["val"]["pairs"],
            "codec_rootrel_mpjpe_cm": prepare["splits"]["train"][
                "codec_rootrel_mpjpe_cm"
            ],
        },
        "training": {
            "first": training[0] if training else None,
            "last": training[-1] if training else None,
            "validation": validation,
            "best_validation_ce": None if not math.isfinite(best_validation) else best_validation,
            "final_validation_ce": None if not math.isfinite(final_validation) else final_validation,
        },
        "checkpoint": {
            "directory": checkpoint_path.name,
            "step": int(checkpoint_meta["step"]),
            "model_config": checkpoint_meta.get("model_config", {}),
        },
        "fixed_seed_generation": {
            "total_generated": total_generated,
            "total_passing": total_passing,
            "pass_rate": total_passing / total_generated if total_generated else 0.0,
            "by_prompt": sweeps,
            "selection_note": "no tracker outcome was used; all fixed seeds are counted",
        },
        "prior_stage": args.prior.name if args.prior else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["result"] != "passed":
        raise SystemExit(f"scale-stage review failed: {gates}")


if __name__ == "__main__":
    main()
