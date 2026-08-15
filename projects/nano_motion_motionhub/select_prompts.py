"""Select concise in-distribution HumanML3D prompt candidates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def captions(texts: Path, split: Path | None):
    if split:
        paths = (texts / f"{item.strip()}.txt" for item in split.read_text().splitlines())
    else:
        paths = texts.rglob("*.txt")
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            caption = line.split("#", 1)[0].strip()
            if caption:
                yield caption


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = {"forward": [], "left": [], "right": []}
    for caption in captions(args.texts, args.split):
        low = caption.lower()
        words = re.findall(r"[a-z']+", low)
        if not 4 <= len(words) <= 16:
            continue
        if "walk" not in low and "walking" not in low and "walks" not in low:
            continue
        if "turn" in low and "left" in low:
            groups["left"].append(caption)
        elif "turn" in low and "right" in low:
            groups["right"].append(caption)
        elif any(word in low for word in ("forward", "straight", "ahead")) and not any(
            word in low for word in ("left", "right", "turn")
        ):
            groups["forward"].append(caption)

    groups_out = {}
    for name, values in groups.items():
        unique = sorted(set(values), key=lambda item: (len(item.split()), len(item), item.lower()))
        groups_out[name] = unique[: args.limit]
    result = {
        "schema": "nano-motion-motionhub-prompts-v1",
        "groups": groups_out,
        "flat": [
            {"category": category, "text": text}
            for category in ("forward", "left", "right")
            for text in groups_out[category]
        ],
    }
    if any(len(groups_out[name]) < args.limit for name in groups_out):
        raise SystemExit(f"not enough prompt candidates: {groups_out}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": {k: len(v) for k, v in groups_out.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
