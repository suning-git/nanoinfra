# data — from game frames to a trainable cache

```
                  download.py ─┐
                               ├─→ encode.py ─→ ../build_cache.py ─→ ../train_wm.py
     scenarios/ ─→ record.py ──┘
        (a map)      (pixels)      (codes)         (memmap)            (training)
```

Two ways to get pixels, one way to turn them into tokens. Pick a path, then run
`encode.py` and `build_cache.py` the same way either way.

| | `download.py` | `record.py` |
|---|---|---|
| needs | ~2GB of network | `pip install vizdoom` |
| gives | someone else's play, someone else's action labels | ground truth actions, unlimited data |
| speed | one 1GB shard ≈ 165 clips | ~1700 frames/s, 200k frames ≈ 2 min |
| longest clip | **~59 frames** (see below) | as long as an episode |

## What it costs

Measured on one RTX 5090, at the default 17-frame / 128px geometry.

| step | rate | size |
|---|---|---|
| record | ~1700 frames/s | 49KB/frame → 200k frames = **9.8GB** |
| encode, recorded pixels | ~330 clips/s | |
| encode, downloaded parquet | ~30 clips/s (PNG decode bound) | |
| codes | | 2.5KB/clip — **~60x smaller than the pixels** |
| cache | one pass, seconds | same as the codes |

Two consequences. First, **pixels are the whole storage cost and they are
disposable**: once `encode.py` has consumed a recording you can delete the `.bin`,
which is why `record.py` streams to a plain memmap instead of anything clever.
Second, **encoding downloaded data is ten times slower than encoding your own**,
entirely because every frame arrives as a base64 PNG.

## Things that will bite you

**Train the text tokenizer first.** The shared vocabulary's text band is sized by
whatever tokenizer is on disk, and every band offset after it — including the video
codes — shifts with that size. With no artifact in `outputs/tokenizer`, core falls
back to a generic gpt2 tokenizer (vocab 50257 instead of 32768) and says so loudly.
Training still runs; it is just a different vocabulary, so nothing you measure is
comparable to the reference numbers, and no checkpoint crosses the boundary.

```bash
python -m modalities.text.train_tokenizer     # seconds
```

**The downloaded set is fragmented, and that limits clip length.** Its stored windows
are sub-sampled — they sit 1, 3, 5, 9, ... steps apart — so expanding them back into
frames yields not one continuous take but roughly 150 contiguous runs per episode,
median length about 25 frames and maximum about 60. The default 17-frame geometry
fits. A 129-frame one cannot be cut from this data at all, and downloading more does
not help, because every shard fragments the same way. Record your own for long
windows.

**Encoding is resumable, and mixing sources is fine.** `encode.py` skips any shard
already in `manifest.jsonl`, so an interrupted run continues by re-running it. Codes
from `download.py` and `record.py` can sit in the same directory and be built into
one cache; `build_cache.py` selects by geometry, from the manifest, not by filename.

**Val is held out by RUN, not by row.** A run is a maximal span of consecutive
frames. Clips cut from the same run overlap, or are a second apart in the same
corridor, so splitting rows at random puts near-duplicates of training frames into
val and every number afterwards reads better than it is. Runs are disjoint, so the
guarantee is exact. What it does not buy: val comes from the same sessions and the
same map as training, which measures "is my training working" well and "does this
generalize to a new game" not at all.

## The format contract

Plugging in a different game means writing one generator. Everything downstream is
source-blind.

**A source yields** `(clip, actions, run_key)`:

| | |
|---|---|
| `clip` | `[frames, res, res, 3]` uint8 |
| `actions` | `[frames]` uint8, **one id per game frame**, ids `0 .. spec.N_ACTIONS-1` |
| `run_key` | any string, unique per contiguous run — it decides train/val |

Actions stay aligned to game frames, not latent frames. The codec is causal with
temporal /4, and `row_layout.py` pairs each latent frame with the four actions that
produced it; averaging or subsampling here destroys exactly the signal the model is
meant to learn.

**`encode.py` writes** `codes/<shard>.npz` and `codes/<shard>_val.npz`, each holding
`codes` `[rows, code_len]` uint16 and `actions` `[rows, frames]` uint8, plus one
`manifest.jsonl` line per shard recording geometry, codec, stride and row counts.

**`build_cache.py` reads** the manifest, selects the shards matching the geometry it
was asked for, and concatenates them into flat `.u16` / `.u8` binaries — the
fixed-stride memmap the loader slices. Its own docstring explains why that shape is
what makes exact resume possible.

## Files

| | |
|---|---|
| `download.py` | fetch public VizDoom parquet shards + the Cosmos codec from HuggingFace |
| `record.py` | play the game and write pixel shards |
| `scenarios/` | the map recorded on — see `../NOTICE` for its provenance |
| `codec.py` | the frozen Cosmos DV4x8x8 encoder, wrapped to one method |
| `encode.py` | both sources → codes + manifest |
