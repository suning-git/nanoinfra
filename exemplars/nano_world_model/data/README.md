# data — from game frames to a trainable cache

```
                    download.py ─┐
                                 ├─→ encode.py ─→ ../build_cache.py ─→ ../train_wm.py
   scenarios/ ─→ record/run.py ──┘
      (a map)   (pixels+sidecar)     (codes)         (memmap)            (training)
```

Two ways to get pixels, one way to turn them into tokens. Pick a path, then run
`encode.py` and `build_cache.py` the same way either way.

| | `download.py` | `record/run.py` |
|---|---|---|
| needs | ~2GB of network | `pip install vizdoom` |
| gives | someone else's play, someone else's action labels | ground truth actions, unlimited data |
| speed | one 1GB shard ≈ 165 clips | ~900-1700 frames/s |
| longest clip | **~59 frames** (see below) | as long as an episode (4000-21000 frames) |
| corpus design | fixed | a YAML recipe (`../data/recipes/minrec.yaml`) |

The recorder is not a random-walk script: it is the minimal subset of the research
line's corpus machine, ported and verified bit-exact (same recipe, same seed, both
recorders — identical shards). What it records is decided by the RECIPE: layer
shares (bot-world episodes vs pans episodes), a segment library with coverage
quotas (committed turns held 8-40 tics, true noop stillness, symmetric L/R
shares), wall avoidance so dwells don't stare into walls, and a stuck-escape
ladder. Each of those exists because a corpus without it taught the model
something wrong — the recipe header says which.

## What it costs

Measured on one RTX 5090, at the default 17-frame / 128px shape contract.

| step | rate | size |
|---|---|---|
| record | ~900 f/s (bots + labels), ~1700 f/s ceiling | 230KB/frame native 240x320 → 1M frames = **230GB** |
| encode, recorded pixels | ~110 clips/s (resize included) | |
| encode, downloaded parquet | ~30 clips/s (PNG decode bound) | |
| codes | | 2.5KB/clip — **~90x smaller than the pixels** |
| cache | one pass, seconds | same as the codes |

Two consequences. First, **pixels are the whole storage cost and they are
disposable**: the durable assets are the codes and the sidecar (actions, poses,
episode table, replay schedule — everything needed to re-render the pixels
later). Once `encode.py` has consumed a `.bin` you delete it. For a corpus
bigger than your disk, work in chunks: record a million frames, encode, delete
the `.bin`, repeat with the next seed — the recorder also pauses itself when
un-encoded pixels exceed `--buffer_cap_gb`. Second, **encoding downloaded data
is several times slower than encoding your own**, because every frame arrives
as a base64 PNG.

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
median length about 25 frames and maximum about 60. The default 17-frame contract
fits. A 129-frame one cannot be cut from this data at all, and downloading more does
not help, because every shard fragments the same way. Record your own for long
windows.

**A short recording has no val rows.** The recorder marks every 8th EPISODE val
(recipe `val.episode_every`), so a smoke-scale recording of fewer than 8
episodes contributes only train rows. That is by design — whole-episode holdout
is the guarantee — but it means the first val rows appear only once the
recording is ~10 episodes long. (The downloaded shards carry their own val
rows, so a mixed cache has a ruler either way.)

**Encoding is resumable, and mixing sources is fine.** `encode.py` skips any shard
already in `manifest.jsonl`, so an interrupted run continues by re-running it. Codes
from both sources can sit in the same directory and be built into one cache;
`build_cache.py` selects by shape contract, from the manifest, not by filename.
Old shards encoded under action-table v1 remain valid — v2 appended NOOP at the
tail, so v1 ids kept their meaning.

**Val is held out by whole spans, never by row.** Clips cut from the same span
overlap, or are a second apart in the same corridor, so splitting rows at random
puts near-duplicates of training frames into val and every number afterwards
reads better than it is. Recorded data holds out whole EPISODES (marked at
record time, above); downloaded data holds out whole contiguous RUNS (hashed —
`split_of` in encode.py says why crc32 and not hash()). What neither buys: val
comes from the same map as training, which measures "is my training working"
well and "does this generalize to a new game" not at all.

## The format contract

Plugging in a different game means writing one generator. Everything downstream is
source-blind.

**A source yields** `(clip, actions, split)`:

| | |
|---|---|
| `clip` | `[frames, res, res, 3]` uint8 |
| `actions` | `[frames]` uint8, **one id per game frame**, ids `0 .. spec.N_ACTIONS-1` |
| `split` | `"train"` or `"val"` — decided per whole episode/run, never per row |

Actions stay aligned to game frames, not latent frames. The codec is causal with
temporal /4, and `row_layout.py` pairs each latent frame with the four actions that
produced it; averaging or subsampling here destroys exactly the signal the model is
meant to learn.

**`encode.py` writes** `codes/<shard>_<F>f.npz` and `codes/<shard>_<F>f_val.npz`,
each holding `codes` `[rows, code_len]` uint16, `actions` `[rows, frames]` uint8
and the action-table version stamp, plus one `manifest.jsonl` line recording the
shape contract, codec, stride and row counts. The clip length is in the name
because one pixel shard is encoded once per length you train at, and Cosmos is
only ~90% prefix-invariant — a 129-frame encoding cannot be sliced down to
17-frame rows, so each length is encoded from its own windows.

**`build_cache.py` reads** the manifest, selects the shards matching the contract it
was asked for, and concatenates them into flat `.u16` / `.u8` binaries — the
fixed-stride memmap the loader slices. Its own docstring explains why that shape is
what makes exact resume possible.

## Files

| | |
|---|---|
| `download.py` | fetch public VizDoom parquet shards + the Cosmos codec from HuggingFace |
| `record/` | the corpus machine: engine, actors, shard writer, episode runner |
| `recipes/minrec.yaml` | the corpus this exemplar ships — layers, quotas, and why |
| `scenarios/` | the map recorded on — see `../NOTICE` for its provenance |
| `codec.py` | the frozen Cosmos DV4x8x8 encoder, wrapped to one method |
| `encode.py` | both sources → codes + manifest |
