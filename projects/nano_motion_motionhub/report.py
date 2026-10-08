"""Summarize MotionHub Text2Motion training without exporting raw logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


TRAIN_RE = re.compile(r"Step\s+(\d+)/(?:\s*)?(\d+).*?loss:\s*([0-9.eE+-]+)")
EVAL_RE = re.compile(
    r"Step\s+(\d+).*?val/motion_ce:\s*([0-9.eE+-]+).*?"
    r"val/motion_ce_best:\s*([0-9.eE+-]+)"
)


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    train = [
        {"step": int(m.group(1)), "max_steps": int(m.group(2)), "loss": float(m.group(3))}
        for m in TRAIN_RE.finditer(text)
    ]
    val = [
        {"step": int(m.group(1)), "motion_ce": float(m.group(2)), "best": float(m.group(3))}
        for m in EVAL_RE.finditer(text)
    ]
    return {"train": train, "validation": val}


def checkpoint_summary(root: Path) -> dict:
    candidates = sorted(root.glob("step_*"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint under {root}")
    latest = candidates[-1]
    meta = json.loads((latest / "meta.json").read_text())
    return {
        "step": int(meta["step"]),
        "model_config": meta.get("model_config", {}),
        "directory": latest.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-report", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["pilot", "formal"], required=True)
    args = parser.parse_args()

    prepare = json.loads(args.prepare_report.read_text())
    curves = parse_log(args.log)
    checkpoint = checkpoint_summary(args.checkpoints)
    generated_npz = sorted(args.generated.glob("*.npz")) if args.generated else []
    generated_gif = sorted(args.generated.glob("*.gif")) if args.generated else []
    generated_mp4 = sorted(args.generated.glob("*.mp4")) if args.generated else []
    train = curves["train"]
    val = curves["validation"]
    expected_checkpoint_step = train[-1]["max_steps"] - 1 if train else math.inf
    best_validation = min((x["motion_ce"] for x in val), default=math.inf)
    final_validation = val[-1]["motion_ce"] if val else math.inf
    gates = {
        "cache_quality": prepare.get("result") == "passed",
        "finite_train_curve": bool(train) and all(math.isfinite(x["loss"]) for x in train),
        "train_loss_decreased": bool(train) and train[-1]["loss"] < train[0]["loss"],
        "finite_validation": bool(val) and all(math.isfinite(x["motion_ce"]) for x in val),
        "validation_better_than_untrained": bool(val) and min(x["best"] for x in val) < 8.0,
        "checkpoint_complete": checkpoint["step"] >= expected_checkpoint_step,
    }
    if args.scope == "formal":
        gates["generated_samples"] = len(generated_npz) >= 3 and len(generated_gif) >= 3
        gates["generated_videos"] = len(generated_mp4) >= 3
        gates["final_validation_close_to_best"] = final_validation <= best_validation * 1.25
    report = {
        "schema": "nano-motion-motionhub-training-v1",
        "scope": args.scope,
        "result": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "data": {
            "source": prepare["source"],
            "source_revision": prepare["source_revision"],
            "train_clips": prepare["splits"]["train"]["clips"],
            "train_pairs": prepare["splits"]["train"]["pairs"],
            "val_clips": prepare["splits"]["val"]["clips"],
            "val_pairs": prepare["splits"]["val"]["pairs"],
            "codec_rootrel_mpjpe_cm": prepare["splits"]["train"]["codec_rootrel_mpjpe_cm"],
        },
        "training": {
            "first": train[0] if train else None,
            "last": train[-1] if train else None,
            "validation": val,
            "best_validation_ce": min((x["best"] for x in val), default=None),
            "best_validation_step": min(val, key=lambda x: x["motion_ce"])["step"] if val else None,
        },
        "checkpoint": checkpoint,
        "generated": {
            "npz": [p.name for p in generated_npz],
            "gif": [p.name for p in generated_gif],
            "mp4": [p.name for p in generated_mp4],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["result"] != "passed":
        raise SystemExit(f"{args.scope} training gate failed: {gates}")


if __name__ == "__main__":
    main()
