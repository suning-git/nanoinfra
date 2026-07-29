"""
build_cache.py — one-shot: data/encode.py's npz shards -> a fixed-stride memmap.

WHY (this is the whole reason the exemplar exists in this shape):

Codes arrive as compressed npz shards, a few thousand rows each. A shard-at-a-time
loader has to hold one decompressed shard resident and serve its rows before moving
on, so the sampling order is "shuffle within the shard we happen to have open". That
is cheap, but it makes the loader's checkpoint state hardware-dependent (which shard,
which rank, which position) — and a rank cannot resume another rank's shard list.
Exact per-sample resume becomes impossible, and resuming on a different number of
GPUs is out of the question.

Every row of a given geometry is exactly the same width, so the fix is to stop
compressing along that axis: one flat binary per field, `np.memmap`ed. Then
`dataset[i]` is a slice at `i * row_bytes`, random access costs one page fault
(~17KB), and the sampler is free to hand out a globally shuffled index stream.
That is what lets core's ResumableDistributedSampler — whose entire state is
{seed, epoch, index}, with no rank in it — drive video training.

Size check, so nobody worries: 539k rows x 1280 codes x 2B = 1.4GB at the default
geometry, i.e. it lives in page cache.

Run (17-frame / 128px, the default geometry):
    python -m exemplars.nano_world_model.build_cache

A different clip length is a different geometry, built beside this one:
    python -m exemplars.nano_world_model.build_cache --frames 33
"""

import argparse
import json
import os
import time

import numpy as np

from exemplars.nano_world_model import spec

# Code ids ride in uint16 and action ids in uint8. Asserted per shard rather than
# assumed: a codec swap that widened the codebook past 65535 would otherwise wrap
# silently and poison the cache.
CODE_DTYPE = np.uint16
ACTION_DTYPE = np.uint8


def shards_for(geom, split):
    """Encoded shards of THIS geometry, from the manifest, sorted and deduped.

    The manifest is the contract: data/encode.py records what each file actually
    contains, so selection here is a match on geometry rather than a guess from the
    filename. That matters because one codes/ directory can hold several geometries
    at once (a 17-frame cache and a 129-frame one), and rows of the wrong width do
    not announce themselves — they surface much later as a shape error, or worse, as
    a silently mis-sliced row.

    Sorted, because THE ORDER IS THE CACHE'S ROW ORDER and must not vary between
    machines or runs.
    """
    names = []
    with open(spec.MANIFEST) as f:
        for line in f:
            e = json.loads(line)
            if e.get("status") != "encoded" or not e.get("rows", {}).get(split):
                continue
            if e.get("geometry") != geom:
                continue
            names.append(e["shard"] + ("" if split == "train" else "_val"))
    return [n for n in sorted(set(names)) if (spec.CODES_DIR / f"{n}.npz").exists()]


def _append(name, shards, out_dir, code_len, n_action_tokens):
    """Stream shards into <name>_codes.u16 / <name>_actions.u8. Returns (rows, provenance)."""
    codes_path = out_dir / f"{name}_codes.u16"
    acts_path = out_dir / f"{name}_actions.u8"
    prov, total = [], 0
    with open(codes_path, "wb") as fc, open(acts_path, "wb") as fa:
        for i, shard in enumerate(shards):
            d = np.load(spec.CODES_DIR / f"{shard}.npz", allow_pickle=True)
            codes, acts = d["codes"], d["actions"]
            assert codes.shape[1] == code_len, \
                f"{shard}: code_len {codes.shape[1]} != geometry's {code_len}"
            assert acts.shape[1] == n_action_tokens, \
                f"{shard}: {acts.shape[1]} action tokens != geometry's {n_action_tokens}"
            assert len(codes) == len(acts), f"{shard}: codes/actions row mismatch"
            assert codes.min() >= 0 and codes.max() < np.iinfo(CODE_DTYPE).max, \
                f"{shard}: code id out of uint16 range (codec vocab grew?)"
            assert acts.min() >= 0 and acts.max() < spec.N_ACTIONS, \
                f"{shard}: action id outside 0..{spec.N_ACTIONS - 1}"
            fc.write(np.ascontiguousarray(codes, dtype=CODE_DTYPE).tobytes())
            fa.write(np.ascontiguousarray(acts, dtype=ACTION_DTYPE).tobytes())
            prov.append({"shard": shard, "rows": int(len(codes))})
            total += len(codes)
            if i % 25 == 0 or i == len(shards) - 1:
                print(f"  [{name}] {i + 1}/{len(shards)} shards, {total} rows", flush=True)
    return total, prov


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--frames", type=int, default=spec.FRAMES)
    ap.add_argument("--res", type=int, default=spec.RES)
    ap.add_argument("--force", action="store_true", help="rebuild even if meta.json exists")
    args = ap.parse_args()

    geom = spec.clip_geometry(args.frames, args.res)
    out_dir = spec.cache_dir(args.frames, args.res)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not args.force:
        raise SystemExit(f"{meta_path} exists — pass --force to rebuild")
    if not spec.MANIFEST.exists():
        raise SystemExit(f"no {spec.MANIFEST} — run data/encode.py first")
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va = shards_for(geom, "train"), shards_for(geom, "val")
    if not tr or not va:
        raise SystemExit(f"manifest has {len(tr)} train / {len(va)} val shards at this "
                         f"geometry ({args.frames}f/{args.res}px) — encode more, or "
                         f"build the geometry you actually encoded")
    print(f"building {out_dir}\n  geometry: {geom}\n  {len(tr)} train shards, {len(va)} val shards",
          flush=True)

    t0 = time.time()
    n_train, prov_train = _append("train", tr, out_dir, geom["code_len"], geom["n_action_tokens"])
    n_val, prov_val = _append("val", va, out_dir, geom["code_len"], geom["n_action_tokens"])

    meta = {
        "geometry": geom,
        "codec": spec.CODEC_NAME,
        "code_dtype": np.dtype(CODE_DTYPE).name,
        "action_dtype": np.dtype(ACTION_DTYPE).name,
        "rows": {"train": n_train, "val": n_val},
        # Repo-relative, so the record is about the data rather than about this box.
        "source": {"root": str(spec.DATASET_ROOT.relative_to(spec.REPO)),
                   "train": prov_train, "val": prov_val},
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    gb = sum(os.path.getsize(out_dir / f) for f in os.listdir(out_dir)) / 1e9
    print(f"\ndone in {time.time() - t0:.0f}s: {n_train} train + {n_val} val rows, "
          f"{gb:.2f} GB\n  -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
