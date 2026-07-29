# data — from motion capture to a trainable code stream

```
download.py ─→ prepare.py ─→ ../train_codec.py ─→ encode.py ─→ ../train_t2m.py
  (raw)        (rot139)        (a tokenizer)       (codes)       (a model)
```

Four steps, and the middle two are the ones worth understanding: **rot139** is the
representation everything speaks, and the **codec** is what turns it into integers.

## Which dataset, and what it lets you train

| | LAFAN1 | AMASS + HumanML3D |
|---|---|---|
| how to get it | `download.py` | by hand — see below |
| captions | none | yes, from HumanML3D |
| trains | a tokenizer, an unconditional motion model | the same, plus **text→motion** |
| size | ~1GB raw | ~26GB raw |

LAFAN1 is the default because it needs no registration, and it exercises the entire
pipeline. Text conditioning needs captions, and the only motion caption set of real
size is HumanML3D, which is written against AMASS. Both require accepting a license
in person:

- **AMASS** — https://amass.is.tue.mpg.de. Register, accept the licence per subset,
  take the "SMPL+H G" archives, unpack to `datasets/amass/<Subset>/<subject>/*.npz`.
  Which subsets is up to you; the index references about a dozen.
- **HumanML3D** — https://github.com/EricGuo5513/HumanML3D. Follow its README to the
  point where it has produced `index.csv`, `texts/`, `train.txt` and `val.txt`, then
  copy those four into `datasets/humanml3d/`.

HumanML3D is **captions only** — it says which frame range of which AMASS file each
caption describes. Without AMASS there is nothing to caption; without HumanML3D there
is nothing to condition on.

```bash
python -m exemplars.nano_motion.data.download --check
```

reports what is present and, for anything missing, what it blocks.

### Three details the pairing gets right

Worth knowing if you re-target this, because none of them is visible in the index file
and each one fails silently:

1. **The index is in 20 fps.** HumanML3D annotates a 20fps resampling of AMASS, while
   this package decimates by an integer stride toward 30 — so the rate it lands on is
   not always exactly 30 (a 100fps source lands on 33.3). The crop is therefore mapped
   by FRACTION OF THE CLIP, which is right whatever either rate turns out to be.
2. **The split is HumanML3D's own** (`train.txt` / `val.txt`), not a fresh random one.
   Its test set is held back.
3. **Mirrored ids (`M****`) are skipped.** HumanML3D augments with left-right flipped
   motion; the captions for those exist but the flipped features do not, so pairing
   them would caption motion that is not there.

Rows that do not resolve — `humanact12` is a different dataset, and any subset you did
not download — are counted and reported rather than dropped quietly.

## rot139, and why the root is a displacement

BVH skeletons and SMPL parameter files describe the same thing incompatibly.
`prepare.py` converts both into one representation: per frame, the joint rotations,
the root's **displacement** and height, and four foot-contact flags. 139 numbers.

The root is a displacement rather than a position on purpose. A model given absolute
positions has to memorise where in the capture volume each recording happened; the
same walk in the far corner of the room looks like different data. As a displacement
it is the same walk anywhere, and each clip's starting root is kept beside the
features (`root0s`) for putting it back into world space when rendering.

This output is **tokenizer-independent** — it does not change when you retrain a
codec — so it lives beside the dataset, at `datasets/<source>/rot139/<split>.npz`.

## What it costs

| step | LAFAN1 | AMASS |
|---|---|---|
| `prepare.py` | ~2 min | an hour or more (LAFAN1 path only — `humanml3d` crops inside `encode.py`) |
| `train_codec.py` | ~30 min for the full 30k steps on one GPU | same |
| `encode.py` | seconds | minutes |

`prepare.py` and `train_codec.py` are each done once. Only `encode.py` needs
repeating, and only when you change codec.

## Things that will bite you

**Train the text tokenizer first.** The shared vocabulary's text band is sized by
whatever tokenizer is on disk, and every band offset after it — including the motion
codes — moves with it. With nothing in `outputs/tokenizer`, core falls back to a
generic gpt2 tokenizer (vocab 50257 instead of 96786) and says so loudly. Training
still runs; it is a different vocabulary, so no checkpoint and no number crosses the
boundary.

```bash
python -m modalities.text.train_tokenizer     # seconds
```

**Codes do not carry their codec.** A code stream is a list of small integers, and
streams from two different codecs are indistinguishable on disk. Train an AR model
against the wrong one and it will train perfectly happily, then decode into motion
that has nothing to do with what it learned. `encode.py` writes a sidecar `.json`
recording which codec produced each file; `train_t2m.py` re-reads the codec to size
the motion band. Do not hand-copy code caches between codecs.

**The codec sets the ceiling.** The AR model can only ever generate motion the codec
can represent. If the samples look mushy, check the codec's reconstruction on real
data before touching the AR model — decoding a clip's own codes shows you the best
the AR model could possibly do.

**Reconstruction numbers are only comparable within one metric.** These are MPJPE in
a **global canonical frame**. An earlier round of this work crowned a different
representation using a heading-blind metric, which flattered anything that discarded
heading — the winner changed when the metric was fixed. A number from a different
evaluation protocol is not comparable to these, however similar it looks.

## The format contract

Plugging in a different motion source means writing one loader that yields, per clip,
a `[T, 139]` float32 array plus its starting root. Register it in
`modalities/motion/data/loaders/` and `prepare.py` picks it up. Everything after
that is source-blind.

| file | what it holds |
|---|---|
| `datasets/<source>/rot139/<split>.npz` | `clips` (ragged `[T,139]`), `root0s` |
| `outputs/motion_caches/<source>_<split>_codes.npz` | `codes` (ragged int32) |
| `outputs/motion_caches/<source>_<split>_codes.json` | which codec made them |
| `outputs/motion_caches/t2m_<source>_<split>.npz` | `codes` + `captions` |
