# RESULTS — what a working run looks like

Reference numbers, so you can tell whether your run is training or broken. All on
RTX 5090s, 12 layers / 768 dim / 12 heads (233M params), the default 17-frame 128px
geometry.

## Is the pipeline alive? (one minute)

Straight after `build_cache.py`, on a few thousand clips:

```bash
python -m exemplars.nano_world_model.train_wm \
    max_steps=30 optimizer.scheduler.warmup_steps=5 \
    evaluation.interval_steps=15 use_compile=false checkpoint.enabled=false
```

```
Step 00000  loss 11.39   val/nelbo 11.474
Step 00015               val/nelbo 10.974
Step 00029  loss 10.94   val/nelbo 10.791
```

The absolute values depend on how much data you built; what matters is that the
NELBO moves down immediately and the four `val/nelbo_t*` levels stay within about
0.01 nat of each other this early. A NELBO pinned at its starting value means the
model is not receiving gradients; levels that disagree wildly at step 0 mean the row
layout and the objective disagree about where blocks start.

Throughput at this size, single GPU, no compile: **~130k tok/s, MFU ~79%**.

## Does it actually learn? (a real run)

10k steps, batch 4 rows, LR 3e-4, warmup 300:

| step | val NELBO |
|---:|---:|
| 1000 | 9.556 |
| 2000 | 9.430 |
| 5000 | 9.293 |
| 10000 | **8.999** |

Measured on 539k clips. A cache built from a handful of downloaded shards is a
different regime — expect higher numbers and earlier overfitting; the curve's shape
is the thing to compare, not its height.

NELBO is in nats per predicted token and is one-directionally comparable to an
autoregressive nll: NELBO ≥ NLL, so a lower number definitively beats AR and a
higher one is inconclusive.

### The optimizer finding behind the config

core's `create_optimizers` uses role-based multi-LR groups, `embedding_lr` 0.2 and
`unembedding_lr` 0.004, tuned on text. On this model they cost **0.34 nat at 10k
steps** — 9.336 against 8.999 for a single LR. The plausible reason is that the video
band is 64000 codes that each appear rarely, so 0.2 on the embedding is far too hot,
but that is a hypothesis; the gap is the measurement. `configs/train_wm.yaml`
therefore sets all three LRs equal, which reduces `build_optimizers` to plain AdamW.

Whether core's defaults should be per-modality is still open.

## Decoding the autoregressive model

`inference/benchmark.py`, one RTX 5090, 12 layers / 768 dim, 256-token segments:

```
teacher-forced argmax: 518/518 agree
greedy segment 0/1/2:  256/256 identical   (2 and 3 continue on the live cache)

static, eager          3.958 ms/token   0.99 latent frames/s ( 3.9 game frames/s)
static, cuda graphs    1.128 ms/token   3.46 latent frames/s (13.9 game frames/s)
                                                    3.5x
```

Equivalence is reported before speed on purpose: a decode path that is fast and
subtly wrong is worse than a slow one. The checks are the ones that survive a kernel
change — teacher-forced argmax (no cascade, so a disagreement is pure numerics) and
greedy decode. Draw-for-draw identity at finite temperature is not a meaningful
target; bf16 noise flips near-tied candidates either way.

The speedup is overhead, not arithmetic. At this size a decode step spends most of
its time launching kernels, and CUDA graphs replay a recorded launch sequence instead.
That is also why the number will look different on a larger model: the same absolute
overhead against more work per token.

## Multi-GPU

2×RTX 5090, PCIe, no NVLink:

| | tok/s | memory/GPU |
|---|---:|---:|
| single GPU | 57.0k | 20GB |
| **replicated (NanoDDP)** | **106.2k** | **19GB** |
| torch DDP | 105.9k | 19GB |
| FSDP, sharded | 93.6k | 30GB |

The model fits, so the bottleneck is activations rather than parameters and sharding
buys nothing. `tests/test_ddp.py` checks that the faster path is also the correct
one — see the README for what each check would catch.

Communication is about 6% of a step at this clip length. That is not a general fact:
`comm/compute ≈ 2087 / (tokens per GPU per step)`, independent of model size, so a
language model at 4096–8192 tokens per step spends 25–44% there and overlap matters
far more.
