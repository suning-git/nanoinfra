"""Generate a repeated-prompt seed range without rendering every candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from core.training.model_setup import load_system
    from exemplars.nano_motion import generate as generation
    from exemplars.nano_motion import spec
    from modalities.motion.tokenizers._convae import MotionCodec

    codec = MotionCodec(args.codec, device="cuda")
    layout, resolver, tokenizer = generation.assemble_vocab(codec)
    setup = load_system(args.ckpt, sequence_len=spec.SEQ_LEN)
    system = setup["system"]
    tags = {
        name: resolver.resolve(name)
        for name in ("bos", "text_start", "text_end", "motion_start", "motion_end")
    }
    prefix = [
        tags["bos"],
        tags["text_start"],
        *tokenizer.encode(args.prompt)[: spec.MAX_TEXT_TOKENS],
        tags["text_end"],
        tags["motion_start"],
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    generated = []
    with torch.no_grad():
        for seed in range(args.seed_start, args.seed_start + args.count):
            codes = generation.sample_motion(
                system,
                layout,
                prefix,
                tags["motion_end"],
                seq_len=spec.SEQ_LEN,
                temperature=spec.TEMPERATURE,
                top_k=spec.TOP_K,
                seed=seed,
            )
            if len(codes) < 2:
                continue
            features = codec.decode(np.asarray(codes, dtype=np.int64))
            path = args.out / f"seed_{seed:04d}.npz"
            np.savez(path, features=features)
            generated.append({"seed": seed, "frames": len(features), "file": path.name})
    (args.out / "generation.json").write_text(
        json.dumps(
            {
                "schema": "nano-motion-motionhub-seed-generation-v1",
                "prompt": args.prompt,
                "seed_start": args.seed_start,
                "requested": args.count,
                "generated": generated,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
