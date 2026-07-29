"""download.py — the fast path to pixels: fetch a public VizDoom dataset and the codec.

There are two ways to get pixels into this exemplar. This is the one that does not
need a game installed:

    download.py  ->  encode.py  ->  build_cache.py  ->  train_wm.py

It pulls shards of `P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-XLrg`, a recorded PPO agent
playing VizDoom deathmatch. Each row is one 10-frame sliding window: `images`
(base64 PNG), `actions` (one id per frame), `episode_id`, `step_id`. Windows within
an episode overlap and chain exactly, which is how encode.py builds clips longer
than 10 frames — see its `_episode_frames`.

Shards are ~1GB / ~1600 windows each, so START SMALL. Two shards is enough to build
a cache and watch the loss come down; the numbers in RESULTS.md used far more.

    python -m exemplars.nano_world_model.data.download              # codec + 2 shards
    python -m exemplars.nano_world_model.data.download --shards 8

The other path — recording your own games — is data/record.py. It gives you ground
truth actions and unlimited data, at the cost of installing vizdoom.
"""

import argparse
import shutil

from huggingface_hub import hf_hub_download

from exemplars.nano_world_model import spec

DATA_REPO = "P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-XLrg"
CODEC_REPO = "nvidia/Cosmos-0.1-Tokenizer-DV4x8x8"
CODEC_FILES = ["encoder.jit", "decoder.jit", "config.json"]


def fetch(repo, filename, dest, repo_type=None):
    """Download one file to `dest` unless it is already there. Returns True if fetched."""
    if dest.exists():
        print(f"  [have] {dest.relative_to(spec.REPO)}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get ] {repo}/{filename} ...", flush=True)
    cached = hf_hub_download(repo, filename, repo_type=repo_type,
                             cache_dir=str(spec.DATASET_ROOT / "_hf"))
    shutil.copyfile(cached, dest)          # copy, so clearing the HF cache is safe
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--shards", type=int, default=2,
                    help="how many ~1GB parquet shards to fetch (default 2)")
    ap.add_argument("--codec-only", action="store_true")
    args = ap.parse_args()

    print(f"codec -> {spec.CODEC_DIR.relative_to(spec.REPO)}")
    for f in CODEC_FILES:
        fetch(CODEC_REPO, f, spec.CODEC_DIR / f)

    if args.codec_only:
        return

    print(f"\nvizdoom shards -> {spec.PIXEL_PARQUET_DIR.relative_to(spec.REPO)}")
    for i in range(args.shards):
        name = f"shard_{i}.parquet"
        fetch(DATA_REPO, name, spec.PIXEL_PARQUET_DIR / name, repo_type="dataset")

    print("\nnext: python -m exemplars.nano_world_model.data.encode")


if __name__ == "__main__":
    main()
