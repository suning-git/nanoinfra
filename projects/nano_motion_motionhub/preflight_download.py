"""Verify one uncached MotionHub file through the configured HF transport."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


REPO_ID = "ZeyuLing/MotionHub"
MOTION_PREFIX = "HumanML3D_AMASS/smplh_52"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    indexes = sorted(args.cache_dir.glob("motionhub_smplh52_shards_*.json"))
    if not indexes:
        raise SystemExit("MotionHub shard index is missing")
    inventory = json.loads(indexes[-1].read_text())
    available = set(inventory["ids"])
    cached = {path.stem for path in args.cache_dir.rglob("*.npz")}
    train_ids = (
        args.starter / "humanml3d" / "train.txt"
    ).read_text().splitlines()
    clip_id = next(
        (item.strip() for item in train_ids if item.strip() in available - cached),
        None,
    )
    if clip_id is None:
        raise SystemExit("no indexed uncached training clip found")

    revision = HfApi().dataset_info(REPO_ID).sha
    started = time.monotonic()
    path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"{MOTION_PREFIX}/{clip_id[:4]}/{clip_id}.npz",
            revision=revision,
            cache_dir=str(args.cache_dir),
        )
    )
    report = {
        "schema": "nano-motion-motionhub-download-preflight-v1",
        "clip_id": clip_id,
        "bytes": path.stat().st_size,
        "seconds": round(time.monotonic() - started, 3),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
