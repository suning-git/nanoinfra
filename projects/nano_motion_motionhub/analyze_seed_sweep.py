"""Select the best valid motion from a repeated-prompt seed sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_candidates import quality, score, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--category", choices=("forward", "left", "right"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for path in sorted(args.generated.glob("*.npz")):
        metrics = trajectory(path)
        candidates.append(
            {
                "category": args.category,
                "prompt": args.prompt,
                "stem": path.stem,
                "metrics": metrics,
                "score": score(args.category, metrics),
                "quality_gate": quality(args.category, metrics),
            }
        )
    if not candidates:
        raise RuntimeError("seed sweep produced no NPZ files")
    passing = [item for item in candidates if item["quality_gate"]]
    selected = max(passing or candidates, key=lambda item: item["score"])
    report = {
        "schema": "nano-motion-motionhub-seed-sweep-v1",
        "result": "passed" if passing else "failed",
        "category": args.category,
        "prompt": args.prompt,
        "seeds": len(candidates),
        "passing_seeds": len(passing),
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
