"""encode.py — pixels to discrete codes: the one step that needs a GPU and real time.

    download.py OR record.py  ->  [encode.py]  ->  build_cache.py  ->  train_wm.py

Two pixel sources feed one encoder. Both yield the same thing — an F-frame clip and
the F action ids that drove it — so everything downstream is source-blind:

    parquet   downloaded a16z shards; rows are 10-frame sliding windows that chain
              within an episode, so longer clips are stitched from them
    recorded  data/record.py's shards; whole episodes, windows cut here

Output is what build_cache.py reads:

    datasets/nano_world_model/codes/<shard>.npz        train rows
    datasets/nano_world_model/codes/<shard>_val.npz    frozen val rows
    datasets/nano_world_model/manifest.jsonl           one line per encoded shard

Each npz holds `codes` [rows, code_len] uint16 and `actions` [rows, frames] uint8.
The manifest line records the geometry, so build_cache.py never has to guess what a
file contains — see the note there about caches of different geometries colliding.

TWO THINGS THIS FILE IS CAREFUL ABOUT, both easy to get wrong when re-targeting:

  1. VAL IS HELD OUT BY EPISODE, NEVER BY ROW. Clips cut at stride < F share frames,
     and even at stride == F two clips from the same episode are seconds apart in the
     same corridor. Splitting rows at random puts near-duplicates of training frames
     in the val set and every number you measure afterwards is optimistic.
  2. ACTIONS STAY ALIGNED TO GAME FRAMES, not to latent frames. The codec is causal
     with temporal /4, so the row layout later pairs each latent frame with the 4
     actions that produced it (row_layout.py). Averaging or subsampling actions here
     would destroy exactly the signal the world model is supposed to learn.

Run:
    python -m exemplars.nano_world_model.data.encode                 # all new shards
    python -m exemplars.nano_world_model.data.encode --limit 200     # a quick taste
    python -m exemplars.nano_world_model.data.encode --source recorded

Re-runnable: shards already in the manifest are skipped, so an interrupted encode
resumes by re-running it.
"""

import argparse
import base64
import io
import json
import time
import zlib
from pathlib import Path

import numpy as np

from exemplars.nano_world_model import spec
from exemplars.nano_world_model.data.codec import CosmosDV

WINDOW = 10          # frames per stored window in the a16z parquet schema
VAL_EVERY = 10       # 1 contiguous run in 10 is held out for val


def split_of(run_key, every=VAL_EVERY):
    """train / val for a whole contiguous RUN of frames.

    A "run" is a maximal span of consecutive frames — the unit both sources produce.
    Holding out whole runs is what makes the guarantee exact: runs are disjoint by
    construction, so no frame can reach both splits. A random split of ROWS cannot
    promise that, because clips cut at stride < clip_len overlap, and even at equal
    stride two neighbouring clips are a second apart in the same corridor.

    Why runs and not whole episodes, which sound stricter? Because an episode is not
    the unit this data actually has. The downloaded set is sub-sampled — its stored
    windows sit 1, 3, 5, ... steps apart — so one episode fragments into ~150 runs of
    a few dozen frames each. Holding out episodes there means holding out entire
    shards, and a reader who fetched two of them would get no val set at all. Runs
    exist at every scale, including one shard. For self-recorded data, where episodes
    are unbroken, a run IS an episode and this reduces to episode holdout.

    crc32, not the builtin hash(): Python randomizes string hashing per process, so
    hash() would silently reshuffle the boundary on every run — and a val set that
    moves is a ruler that moves.

    What it does not buy, stated plainly: val runs come from the same sessions and
    the same map as training. This measures "is my training working" well and "does
    this generalize to a new game" not at all.
    """
    return "val" if zlib.crc32(run_key.encode()) % every == 0 else "train"


# --- source 1: downloaded parquet --------------------------------------------

def _decode_png(s, res):
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    return np.asarray(im.resize((res, res), Image.BILINEAR), np.uint8)


def parquet_clips(path, frames, res, stride):
    """Yield (clip, actions, run_key) from one a16z parquet shard.

    Rows are 10-frame windows at overlapping steps: the window at step s holds global
    frames s..s+9, and consecutive windows agree pixel-for-pixel on their overlap. So
    an episode's frame g can be read from any stored step s with s <= g < s+10, and
    clips longer than 10 frames are just a walk over that index.

    The set is sub-sampled — stored windows sit 1, 3, 5, 9, ... steps apart — so the
    frames it covers are NOT one continuous take. A typical shard's single episode
    expands into ~150 contiguous runs with a median length near 25 frames and a
    maximum near 60. Two consequences worth knowing before you plan anything:

      * clips never span a gap, because the frames across a gap are seconds apart and
        nothing connects them;
      * a clip longer than the longest run cannot be cut from this data AT ALL. The
        default 17-frame geometry fits comfortably; a 129-frame one does not, and no
        amount of downloading fixes that. Record your own (data/record.py) for those.
    """
    import pyarrow.parquet as pq

    episodes = {}        # episode_id -> {step: (images, actions)}
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=256, columns=["images", "actions", "episode_id", "step_id"]):
        d = batch.to_pydict()
        for im, ac, ep, st in zip(d["images"], d["actions"], d["episode_id"], d["step_id"]):
            episodes.setdefault(ep, {})[st] = (im, ac)

    for ep, by_step in episodes.items():
        # global frame -> (step, offset within that window); later steps win, and
        # they agree with earlier ones on the overlap, so which one wins is moot.
        src = {}
        for st in sorted(by_step):
            for k in range(WINDOW):
                src[st + k] = (st, k)

        present = sorted(src)
        run_start, run_no = 0, 0
        for i in range(1, len(present) + 1):
            broken = i == len(present) or present[i] != present[i - 1] + 1
            if not broken:
                continue
            run = present[run_start:i]
            run_start, run_no = i, run_no + 1
            key = f"{Path(path).stem}/{ep}/{run_no}"
            for g0 in range(0, len(run) - frames + 1, stride):
                clip, acts = [], []
                for g in run[g0:g0 + frames]:
                    st, k = src[g]
                    im, ac = by_step[st]
                    clip.append(_decode_png(im[k], res))
                    acts.append(ac[k])
                yield np.stack(clip), np.asarray(acts, np.uint8), key


# --- source 2: our own recordings --------------------------------------------

def recorded_clips(path, frames, res, stride):
    """Yield (clip, actions, run_key) from one shard written by data/record.py.

    Pixels are a raw uint8 memmap (no per-frame decode) and episodes are stored whole,
    so windowing is a slice and one episode is one run. This is the cheap source; the
    parquet one above pays for PNG decoding on every frame of every clip.
    """
    meta = np.load(str(path).replace(".bin", "_meta.npz"))
    actions, episode_id = meta["actions"], meta["episode_id"]
    h = w = int(meta["res"])
    px = np.memmap(path, dtype=np.uint8, mode="r").reshape(-1, h, w, 3)
    assert len(px) == len(actions), f"{path}: {len(px)} frames vs {len(actions)} actions"

    for ep in np.unique(episode_id):
        idx = np.where(episode_id == ep)[0]
        key = f"{Path(path).stem}/{ep}/0"          # unbroken episode = one run
        for i in range(0, len(idx) - frames + 1, stride):
            sl = idx[i:i + frames]
            clip = px[sl]
            if h != res:
                from PIL import Image
                clip = np.stack([np.asarray(Image.fromarray(f).resize((res, res),
                                 Image.BILINEAR), np.uint8) for f in clip])
            yield np.asarray(clip), actions[sl].astype(np.uint8), key


# --- the encode loop ----------------------------------------------------------

def encode_shard(name, clips, codec, geom, out_dir, batch, limit):
    """Drain `clips` through the codec into <name>.npz / <name>_val.npz."""
    buf = {"train": ([], []), "val": ([], [])}
    out = {"train": 0, "val": 0}
    pend_clips, pend_acts, pend_val = [], [], []
    n = 0

    def flush():
        if not pend_clips:
            return
        codes = codec.encode(pend_clips)
        # The geometry is derived from (frames, res, codec) in spec.py; if what the
        # encoder actually returned disagrees, the cache would be built to a width
        # nothing downstream expects. Cheap to check once per batch, so check.
        assert codes.shape[1] == geom["code_len"], \
            f"codec returned {codes.shape[1]} codes/clip, geometry says {geom['code_len']}"
        assert codes.max() < spec.CODEC_VOCAB, \
            f"code id {codes.max()} outside the codec's {spec.CODEC_VOCAB}-entry codebook"
        for row, act, split in zip(codes, pend_acts, pend_val):
            c, a = buf[split]
            c.append(row.astype(np.uint16)); a.append(act)
        pend_clips.clear(); pend_acts.clear(); pend_val.clear()

    for clip, acts, run_key in clips:
        pend_clips.append(clip); pend_acts.append(acts)
        pend_val.append(split_of(run_key))
        n += 1
        if len(pend_clips) >= batch:
            flush()
            print(f"  [{name}] {n} clips", end="\r", flush=True)
        if limit and n >= limit:
            break
    flush()

    for split, (c, a) in buf.items():
        if not c:
            continue
        suffix = "" if split == "train" else "_val"
        path = out_dir / f"{name}{suffix}.npz"
        np.savez(path, codes=np.stack(c), actions=np.stack(a))
        out[split] = len(c)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source", choices=["parquet", "recorded"], default="parquet")
    ap.add_argument("--frames", type=int, default=spec.FRAMES)
    ap.add_argument("--res", type=int, default=spec.RES)
    ap.add_argument("--stride", type=int, default=None,
                    help="clip stride in frames (default = frames, i.e. no overlap)")
    ap.add_argument("--batch", type=int, default=8, help="clips per encoder call")
    ap.add_argument("--limit", type=int, default=None, help="clips per shard (smoke test)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    stride = args.stride or args.frames
    geom = spec.clip_geometry(args.frames, args.res)

    if args.source == "parquet":
        shards = sorted(spec.PIXEL_PARQUET_DIR.glob("*.parquet"))
        reader, hint = parquet_clips, "data/download.py"
    else:
        shards = sorted(spec.PIXEL_SHARD_DIR.glob("*.bin"))
        reader, hint = recorded_clips, "data/record.py"
    if not shards:
        raise SystemExit(f"no {args.source} shards found — run {hint} first")

    done = set()
    if spec.MANIFEST.exists():
        with open(spec.MANIFEST) as f:
            done = {json.loads(line)["shard"] for line in f}

    spec.CODES_DIR.mkdir(parents=True, exist_ok=True)
    codec = CosmosDV(spec.CODEC_DIR, device=args.device)
    print(f"encoding {len(shards)} {args.source} shard(s) -> {spec.CODES_DIR}\n"
          f"  geometry: {geom}\n  stride {stride}, batch {args.batch}", flush=True)

    for path in shards:
        name = Path(path).stem
        if name in done:
            print(f"  [skip] {name} (in manifest)")
            continue
        t0 = time.time()
        rows = encode_shard(name, reader(path, args.frames, args.res, stride),
                            codec, geom, spec.CODES_DIR, args.batch, args.limit)
        with open(spec.MANIFEST, "a") as f:
            f.write(json.dumps({
                "shard": name, "status": "encoded", "source": args.source,
                "geometry": geom, "codec": spec.CODEC_NAME, "stride": stride,
                "rows": rows, "seconds": round(time.time() - t0, 1),
            }) + "\n")
        print(f"  [{name}] {rows} in {time.time() - t0:.0f}s" + " " * 20, flush=True)

    totals = {"train": 0, "val": 0}
    with open(spec.MANIFEST) as f:
        for line in f:
            for k, v in json.loads(line)["rows"].items():
                totals[k] = totals.get(k, 0) + v
    print(f"\nmanifest totals: {totals}")
    if not totals["val"]:
        # Whole runs go to val, so with few runs it is luck whether any lands there.
        # That is fine at the intended scale (a parquet shard is ~150 runs) and it
        # bites on a tiny self-recorded smoke test (one episode = one run).
        raise SystemExit(
            f"no val rows: with 1-in-{VAL_EVERY} runs held out, none of the runs "
            f"encoded landed in val. Encode more data — a longer recording, or more "
            f"shards. (If you are just smoke-testing, --limit still needs enough runs.)")

    print("next: python -m exemplars.nano_world_model.build_cache")


if __name__ == "__main__":
    main()
