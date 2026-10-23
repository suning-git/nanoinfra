# RESULTS — what a working run looks like

Reference numbers, so you can tell whether your run is training or broken. All on
RTX 5090s, 12 layers / 768 dim / 12 heads (233M params), the default 17-frame 128px
geometry.

> **Storage precision.** These numbers were measured with the parameters, their gradients and the AdamW moments
> stored in bf16, core's only storage when they ran (code up to v0.2.0). Training now stores them in fp32 by default
> (the `param_dtype` config key; the arithmetic stays under bf16 autocast), so a run on current code differs from them
> a little in the first minute and more over a long run. `param_dtype=bf16` returns to the storage they used; to
> reproduce them exactly, use v0.2.0.

> **Pipeline.** Since 2026-10 the exemplar trains block diffusion only, on core's data pipeline, with each block's
> mean CE over its masked cells as the loss. The readings below the quickstart were measured before that, on the
> exemplar's own fixed-row loader with MDLM's 1/t weight. Side by side on the same card, the same data and the same
> seeds (2000 steps of 4 rows, val every 500, two seeds each) the two read the same within seed noise: at step 1999
> 9.4120 / 9.4084 (now) against 9.3965 / 9.4101 (before), while the two seeds of one version differ by up to 0.044
> along the way; a step takes 76 ms in both. With the old code's masks injected and the same loss weights on both
> paths, the new path's inputs, per-token CE, loss and every gradient are the old path's bit for bit.

## Is the pipeline alive? (under a minute)

Straight after `build_cache.py`, on a few thousand clips:

```bash
python -m exemplars.nano_world_model.train_wm \
    max_steps=15 optimizer.scheduler.warmup_steps=5 \
    evaluation.interval_steps=15 evaluation.val_rows=100 checkpoint.enabled=false
```

Training runs through flex attention, which must run compiled, so the run first
compiles each block (`compile_trunk: true`, the default; core's trunk refuses an
uncompiled flex call with gradients on). It evaluates 100 of the val clips, at the
first step and the last. On 4733 training clips, one RTX 5090, 17 s from launch to exit:

```
Step 00000  loss 11.48   val/nelbo 11.472
Step 00014  loss 11.14   val/nelbo 10.969
```

The absolute values depend on how much data you built; what matters is that the
NELBO moves down immediately and the four `val/nelbo_t*` levels stay within about
0.01 nat of each other this early. A NELBO pinned at its starting value means the
model is not receiving gradients; levels that disagree wildly at step 0 mean the row
layout and the objective disagree about where blocks start.

Throughput at this size, single GPU: **~149k tok/s, 76 ms a step** (4 rows of 2816
positions, two micro-batches), MFU ~51% by block_diffusion.py's corrected estimate.
An evaluation of the 420 val clips at four noise levels takes about 8 s.

## Does it actually learn? (a real run)

10k steps, batch 4 rows, LR 3e-4, warmup 300:

| step | val NELBO |
|---:|---:|
| 1000 | 9.556 |
| 2000 | 9.430 |
| 5000 | 9.293 |
| 10000 | **8.999** |

Measured on 539k clips, on the pre-port stack (1D rope, action table v1, batch
4 rows — the 2026-07 exemplar). The stack has since moved to 3D rope and table
v2; the quickstart above was re-verified to the digit on the new stack, this
10k table was not re-run. A cache built from a handful of downloaded shards is
a different regime anyway — expect higher numbers and earlier overfitting; the
curve's shape is the thing to compare, not its height.

NELBO is in nats per predicted token.

### The optimizer finding behind the config

core's `build_optimizers` uses role-based multi-LR groups, `embedding_lr` 0.2 and
`unembedding_lr` 0.004, tuned on text. On this model they cost **0.34 nat at 10k
steps** — 9.336 against 8.999 for a single LR. The plausible reason is that the video
band is 64000 codes that each appear rarely, so 0.2 on the embedding is far too hot,
but that is a hypothesis; the gap is the measurement. `configs/train_wm.yaml`
therefore sets all three LRs equal, which reduces `build_optimizers` to plain AdamW.

Whether core's defaults should be per-modality is still open.

## The real run (README's curriculum) — reference points

These come from the research twin this exemplar was crystallized out of (not in
the public release), whose training stack is what this exemplar now is: core
machinery, 3D rope, action table v2. Corpora differ from yours, and **NELBO is
corpus-relative — never compare absolute numbers across corpora**; what should
reproduce is the shape and the regime effects.

### Why the volume unit below is "supervised tokens", not steps

A step of this model touches three different counts, and only one of them is
training volume:

| | what it counts | 17-frame recipe |
|---|---|---|
| `total_batch_size` (core's) | sequence positions: clean stream + noised stream + pad + the given frame | 2816 / row |
| row length | one row | 1408 |
| **supervised tokens** | **positions actually predicted** | **1024 / row** |

The ratio between them is not a constant — it moves with window length. So a curve
plotted against steps, or against `total_batch_size`, compares nothing once two runs
have different windows, and comparing a 17-frame run with a 129-frame one is exactly
what this project exists to do. The research twin learned this the hard way: a
two-nat discrepancy that took a long time to explain had a steps-as-volume comparison
as its final wrong turn.

So: **`supervised = rows x n_blocks x codes_per_frame`** (1024 per row here), and every
number below is quoted per that. Core cannot compute it for you — `total_batch_size` is
the only volume it knows, and for this objective that is the wrong one.

| leg | regime | val NELBO |
|---|---|---|
| 17f from scratch | b32 rows, lr 3e-4, 57k steps, constant LR | ~5.16 at end |
| 129f warm-started | b32 rows (262k supervised tok/update), lr 6e-4, 9k steps, warmdown | **4.195** |

Regime effects you should see on your own corpus, whatever its absolute level:

* equal-token batch scaling 4→32 rows: ~0.15 nat free, knee between 16k and 32k
  supervised tokens per update;
* the same equal-token comparison at 129f (9k x 32 rows vs 72k x 4): every
  tracked metric improved, val 4.343 → 4.195 on the twin's corpus;
* warmdown: a roughly uniform ~-0.37 nat endpoint shift, order-preserving
  across arms;
* the shipped recipe's corpus (bots + pans, no monsters) trained to a curve
  within ±0.024 nat of the twin's reference corpus at equal volume — the
  minimal recorder is not the bottleneck.

## Generating

`inference/benchmark.py`, one RTX 5090, 12 layers / 768 dim, random weights, 17-frame
clips, 8 denoising rounds per latent frame, the machine otherwise idle:

```
block 1: max |logit diff| 0.0234, argmax agrees on 256/256
block 2: max |logit diff| 0.0233, argmax agrees on 254/256, top1-top2 margins where not: [0.0153, 0.0]
block 3: max |logit diff| 0.0233, argmax agrees on 256/256
block 4: max |logit diff| 0.0232, argmax agrees on 256/256

36.3 ms per latent frame (4 game frames): 110 game frames/s, eager trunk, dynamic KV cache
```

Agreement comes before speed on purpose: a sampler that is fast and subtly wrong is
worse than a slow one. The comparison is with the training forward on the same block —
the sampler's SDPA over a KV cache against flex over the two-stream row — so under bf16
the logits agree to rounding, not bit for bit — the two cells of block 2 whose argmax
differs are near-ties (margins 0.015 and 0.0); in float32 on the CPU
(`tests/test_sampler.py`) they agree to 1e-5.

A latent frame costs one forward for its actions, eight over its 256 cells and one to
commit it. With an eager trunk most of that is launching kernels, so the number depends
on the CPU as much as on the GPU: with a training job starting up on the other GPU of
the same box (compiling, preparing batches) the same run took 83 ms.

## Multi-GPU

2×RTX 5090, PCIe, no NVLink:

| | tok/s | memory/GPU |
|---|---:|---:|
| single GPU | 57.0k | 20GB |
| **replicated (NanoDDP)** | **106.2k** | **19GB** |
| torch DDP | 105.9k | 19GB |
| FSDP, sharded | 93.6k | 30GB |

(Measured on an earlier version of the exemplar, whose single-GPU throughput was below
today's ~149k tok/s; read the rows against each other, not against that line.) The model
fits, so the bottleneck is activations rather than parameters and sharding buys nothing. `tests/test_ddp.py` checks that the faster path is also the correct
one — see the README for what each check would catch.

Communication is about 6% of a step at this clip length. That is not a general fact:
`comm/compute ≈ 2087 / (tokens per GPU per step)`, independent of model size, so a
language model at 4096–8192 tokens per step spends 25–44% there and overlap matters
far more.
