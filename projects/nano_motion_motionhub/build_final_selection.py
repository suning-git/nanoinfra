"""Choose final forward/left/right motions across generated prompt batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_candidates import quality, score, trajectory


def add_item(items: list[dict], category: str, text: str, pool: str, path: Path) -> None:
    metrics = trajectory(path)
    items.append(
        {
            "category": category,
            "text": text,
            "pool": pool,
            "stem": path.stem,
            "metrics": metrics,
            "score": score(category, metrics),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items: list[dict] = []
    prompt_items = json.loads(args.prompts.read_text())["flat"]
    for index, prompt in enumerate(prompt_items):
        matches = sorted(args.candidates.glob(f"{index:02d}_*.npz"))
        if len(matches) != 1:
            raise RuntimeError(f"candidate index {index} has {len(matches)} NPZ files")
        add_item(items, prompt["category"], prompt["text"], "native_prompt_candidates", matches[0])

    original = [
        ("forward", "A person walks forward at a steady pace and stops."),
        ("left", "A person walks forward and turns left."),
        ("right", "A person walks forward and turns right."),
    ]
    for index, (category, text) in enumerate(original):
        matches = sorted(args.original.glob(f"{index:02d}_*.npz"))
        if len(matches) != 1:
            raise RuntimeError(f"original index {index} has {len(matches)} NPZ files")
        add_item(items, category, text, "original_prompts", matches[0])

    selected = []
    for category in ("forward", "left", "right"):
        best = max((item for item in items if item["category"] == category), key=lambda x: x["score"])
        best = dict(best)
        best["quality_gate"] = quality(category, best["metrics"])
        selected.append(best)
    report = {
        "schema": "nano-motion-motionhub-final-selection-v1",
        "result": "passed" if all(item["quality_gate"] for item in selected) else "failed",
        "selected": selected,
        "pool_size": len(items),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
