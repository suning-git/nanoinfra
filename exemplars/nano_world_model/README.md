# nano_world_model — a discrete video world model, at exemplar quality

Train a transformer to answer: *given what I have seen and the buttons I pressed,
what does the world look like next?* The world is VizDoom; the model sees the game
as discrete tokens, exactly the way this framework sees text.

Everything from raw game frames to interactive generation is here — recording or
downloading pixels, tokenizing them, training the model two different ways, and
decoding it fast enough to play against. It is also the first user of
`core.parallel.NanoDDP`, the replicated data parallelism this project built and
merged into core.

**Two objectives over one pipeline.** The same data, the same rows and the same trunk
train either an autoregressive model or a block-diffusion one; they differ in `loss()`
and nothing else. That is not indecision — they trade against each other. Diffusion
trains cheaply and needs several forwards per frame to generate; autoregression needs
one forward per TOKEN, which is expensive in the obvious way and is also exactly what
a KV cache and CUDA graphs make fast. `inference/` decodes the autoregressive one at
**1.1 ms/token**, about 14 game frames per second on one GPU.

**Why this exemplar is self-contained, when `text_pretrain` next door is a 41-line
driver.** That one is thin because the text ORCHESTRATOR lives in `modalities/text/`,
where several projects share it — and what makes a modality package worth having is a
shared TOKENIZER. Video has no such thing: every video project either borrows a frozen
codec or trains its own, and the codec's facts (vocabulary size, spatial and temporal
downsampling) change with it. Extracting a `modalities/video/` would have moved a
four-line `Modality(...)` declaration whose only interesting argument comes from the
project anyway, and split code that belongs together. So the whole pipeline lives
here, and the rule to take away is: **extract a modality when there is a tokenizer to
share, not when two projects merely look like the same modality.**

**Which codec this uses.** The frozen Cosmos DV4x8x8, borrowed — chosen so the
exemplar teaches world-model training without also requiring you to train a tokenizer
first. It is not the best one available: a causal FSQ tokenizer trained on this data
beats it substantially (+6dB reconstruction, and 2.3x the end-to-end action signal).
That is a separate project.

---

## What it trains

A 17-frame 128px clip becomes 5 latent frames of 256 discrete codes each (frozen
Cosmos DV4x8x8 tokenizer, temporal /4, spatial /8), interleaved with the action ids
that drive them:

```
[bos, vstart, L0 (given), a a a a, L1, a a a a, L2, a a a a, L3, a a a a, L4, vend, eos, pad..]
        ^ 256 codes = one latent frame = one diffusion BLOCK
```

Frame 0 is the given observation; the other four are predicted. HOW they are
predicted is the `objective` knob.

**`objective=ar`** predicts the next token causally, over the code positions of the
predicted frames plus the closing tag. Nothing supervises the given frame (it is the
observation) or the action tokens (they are the control input, and learning which
button a player presses is a different problem).

**`objective=diffusion`** uses **absorbing-state (masked) diffusion**. The forward
process masks each token of a
block with probability `t ~ U[0.2, 1]`; the model sees the clean prefix plus the
partially masked block and predicts the clean token at each masked position. One
forward pass covers every block, using BD3-LM's two-stream trick: the sequence is
`[clean row | noisy row]` and the attention mask lets noisy block *k* read the clean
prefix before its own start plus itself bidirectionally.

Both report nats per predicted token on the same frozen val rows — nll for AR, NELBO
for diffusion. The comparison is ONE-DIRECTIONAL: NELBO ≥ NLL for the same model
class, so a diffusion number below an AR number is a real win and one above it is
inconclusive. Saying so is the only honest way to put the two side by side.

## Layout

| file | what it owns |
|---|---|
| `spec.py` | the knob panel: protocol ids (action table v2), the shape contract, recipe. One edit re-targets everything. |
| `data/` | **pixels to codes** — download or record (a recipe-driven corpus machine), tokenize, and the format contract. Has its own README. |
| `build_cache.py` | one-shot: encoded shards → a fixed-stride memmap. |
| `dataset.py` | map-style dataset + the infinite, resumable, rank-sharded loader. |
| `row_layout.py` | where every token sits; the two-stream attention mask; mirrored rope. |
| `rope3d.py` | the position regime: true (t, y, x) rotary coordinates, one table swap. |
| `block_diffusion.py` | one objective: noising, weighted masked CE, val NELBO — and the `System` that carries it. |
| `autoregressive.py` | the other: next-token CE over the predicted future, and its `System`. |
| `inference/` | the real-time decode path — static shapes, CUDA graphs, and the equivalence checks that justify them. |
| `evaluator.py` | the frozen ruler, on core's Evaluator contract. |
| `train_wm.py` | **the orchestrator** — read this first. |
| `configs/train_wm.yaml` | this project's knobs; mechanism defaults come from core. |
| `tests/test_ddp.py` | multi-GPU correctness (below). |

## Run it

```bash
# 0. the text tokenizer, which sizes the shared vocabulary (seconds)
python -m modalities.text.train_tokenizer

# 1. pixels: download a public VizDoom recording + the codec (~2GB)
python -m exemplars.nano_world_model.data.download

# 2. pixels -> discrete codes  (the one step that wants a GPU)
python -m exemplars.nano_world_model.data.encode

# 3. codes -> fixed-stride memmap
python -m exemplars.nano_world_model.build_cache

# 4. train, on one GPU — objective=diffusion (default) or objective=ar
python -m exemplars.nano_world_model.train_wm
python -m exemplars.nano_world_model.train_wm objective=ar

# ... or on two, replicated, with NanoDDP
torchrun --nproc_per_node=2 --standalone \
    -m exemplars.nano_world_model.train_wm parallel=ddp

# 5. how fast can the autoregressive model be decoded, and is the fast path right?
python -m exemplars.nano_world_model.inference.benchmark
```

Step 1 has an alternative: record your own —

```bash
python -m exemplars.nano_world_model.data.record.run --tag rec1 --seed 0 --minutes 30
python -m exemplars.nano_world_model.data.encode --source recorded
```

which gives ground-truth actions and unlimited data at the cost of
`pip install vizdoom`. The recorder is driven by a corpus RECIPE
([data/recipes/minrec.yaml](data/recipes/minrec.yaml)) — layer shares, coverage
quotas, wall avoidance — because what you record decides what the model can
learn; the recipe header carries the reasons. See
[data/README.md](data/README.md) for what each path costs and what it can and
cannot produce.

## The real run: a 129-frame model you can play

The five-command quickstart above trains a REAL model on a TOY budget. The
regime below is the settled one from the research line — every number in it was
measured, not guessed (see RESULTS.md):

```bash
# a real corpus: ~8M frames of the shipped recipe, recorded in chunks
#   (~2h of recording; pixels are disposable, delete each .bin after encoding)
python -m exemplars.nano_world_model.data.record.run --tag c0 --seed 0 --max_frames 1000000 --minutes 999
python -m exemplars.nano_world_model.data.encode --source recorded               # 17f rows
python -m exemplars.nano_world_model.data.encode --source recorded --frames 129  # 129f rows
python -m exemplars.nano_world_model.build_cache
python -m exemplars.nano_world_model.build_cache --frames 129

# stage 1 — 17 frames, from scratch, constant LR (it will be warm-started from,
# and a checkpoint taken mid-schedule warm-starts cleaner than one mid-decay)
python -m exemplars.nano_world_model.train_wm \
    total_batch_rows=32 max_steps=57000 optimizer.scheduler.warmdown_ratio=0

# stage 2 — 129 frames, warm-started, equal tokens at 8x the update size,
# warmdown over the final stretch (the config default, warmdown_ratio 0.2)
python -m exemplars.nano_world_model.train_wm \
    clip.frames=129 device_batch_size=2 total_batch_rows=32 \
    optimizer.lr_max=6e-4 max_steps=9000 \
    checkpoint.init_model_from=<stage-1 checkpoint dir>
```

Three regime facts carry this, each paid for once on the research line:

* **≥32k supervised tokens per update** (`total_batch_rows=32`). At 4 rows the
  same token budget spends itself in updates that are individually too noisy;
  raising the batch 8x at equal tokens improved val by ~0.15 nat for free, with
  the knee between 16 and 32k. That is also why stage 2 is 9000 large steps
  rather than 72000 small ones.
* **Curriculum, not scratch.** Warm-starting 129f from a 17f parent reaches the
  same loss as direct 129f training for ~1.5x less compute. The 3D rope
  (rope3d.py) is what makes the window change free: coordinates are absolute
  (t, y, x) with bases fixed across window lengths, so the parent's tables
  simply extend.
* **Warmdown at the end, judgments before it.** The end-of-run LR decay moves
  the endpoint by roughly -0.37 nat uniformly across arms — so it buys quality
  but reorders nothing, and any comparison you run mid-training is honest as
  long as both arms are pre-warmdown.

To play the result, decode it interactively — `inference/` holds the fast path
(1.1 ms/token) and `inference/benchmark.py` proves it faithful before it
reports speed.

---

## Three decisions worth knowing about

### 1. The data is a memmap, so the sampler can be core's

A loader that walks compressed shards has to keep one shard resident and serve its
rows before moving on, so its checkpoint state is **which shard, which rank, which
position** — hardware-dependent. A resumed rank 1 cannot read rank 0's shard list
(only rank 0 writes `meta.json`), so under DDP the position restore has to be skipped
entirely.

Every row of a given geometry has the same width, so `build_cache.py` writes one
flat binary per field and `dataset[i]` becomes a slice. Random access costs a page
fault, which lets core's `ResumableDistributedSampler` drive the run — and its state
is `{seed, epoch, index}`, three integers with no rank in them. Consequences:
resume is exact to the sample on every rank, and **a run checkpointed on 2 GPUs
resumes on 1**. The loader was the problem, not the parallelism.

### 2. The diffusion noise is seeded from the data, not from the step

`VideoRowLoader` derives each micro-batch's mask seed from *which rows are in it*.
So resume replays the same noise for the same rows, different ranks get independent
masks, and a 2-GPU step and a 1-GPU 2-accumulation step over the same micro-batches
draw identical noise — which is what makes the DDP equivalence test tight enough to
be worth running.

### 3. Multi-GPU replicates rather than shards

At this size the model fits and the bottleneck is activations, not parameters, so
sharding buys nothing and costs an all-gather per block per step. Measured on
2×RTX 5090 (233M params): replicated 106.2k tok/s / 19GB vs FSDP 93.6k / 30GB.
`build_system(parallel="ddp")` places replicas; `NanoDDP` reduces one bucket per
transformer block from a gradient hook during backward, so communication overlaps
the remaining compute. Both are in core — `core/training/model_setup.py` and
`core/parallel/nano_ddp.py`.

On a dense trunk this requires **per-block** `torch.compile`: a whole-graph compile
makes AOTAutograd finalize every gradient at the very end of backward (PyTorch
#109774), so every all_reduce piles up after the compute instead of hiding inside it.
It does not generalize — a trunk that already graph-breaks has the seams anyway; see
`compile_blocks`.

### 4. Generating fast is an engineering problem, not a modelling one

An autoregressive world model emits 256 codes per latent frame, one forward each, so
whether it is interactive is decided by per-token OVERHEAD rather than by FLOPs. At
this size most of a token goes into launching kernels.

CUDA graphs remove that by recording the launch sequence once and replaying it — but
they demand that every shape and every address stay fixed, which an ordinary decode
path violates in three specific ways: the KV cache returns a growing slice, the
attention mask shrinks with the sequence, and the rotary tables are sliced with a
Python integer. `core.model.kv_cache.StaticKVCache` fixes each one, and
`torch.compile(mode="reduce-overhead")` then captures the graphs by itself:

```python
trunk.attach_kv_cache(StaticKVCache.for_model(trunk.config, 1, seq_len))
trunk(idx, kv_cache=STATIC)
```

Two steps rather than one argument, and that is load-bearing: a cache passed THROUGH
forward is a mutated graph input, so dynamo declines to capture — silently, with the
same answers and a third of the speed gone.

Measured on one RTX 5090, 12 layers / 768 dim: **3.958 -> 1.128 ms/token, 3.5x**.

The interesting part is not the speedup, it is what it costs to trust it. A decode
path that is fast and subtly wrong is worse than a slow one, so `inference/benchmark.py`
checks agreement with core's ordinary path first and only then reports time — and it
checks it in the way that survives kernel changes. Draw-for-draw identity at finite
temperature is not a meaningful target, because bf16 noise flips near-tied candidates
either way; teacher-forced argmax and greedy decode are.

This began as a monkeypatch of core's GPT in this project, and moved into core once
both consumers here — the autoregressive model and block diffusion — had exercised it.
That order was the point: proving the three changes against real users is what earns
the right to put a decode-shaped constraint into core.

---

## Tests

```bash
torchrun --nproc_per_node=2 --standalone \
    -m exemplars.nano_world_model.tests.test_ddp
```

Two ways of being wrong about multi-GPU training, both silent:

| check | what it would catch |
|---|---|
| NanoDDP == backward-then-all_reduce, bitwise | an overlapped reduction that drops or misorders a bucket |
| 2 ranks == 1 rank with `grad_accum=2`, over the same micro-batches | a batch that is split but not averaged the way you think |
| a bucket that receives no gradient **raises** | replicas drifting apart with training happily continuing |

Neither shows up as a crash, which is the reason to test them rather than trust them.
See `RESULTS.md` for what the numbers should look like when it is working.

## Open question

**core's optimizer defaults are text-tuned.** `create_optimizers` uses role-based
multi-LR groups (`embedding_lr` 0.2, `unembedding_lr` 0.004). Measured head to head
on this model they cost **0.34 nat at 10k steps**, so `configs/train_wm.yaml` sets all
three LRs equal — which reduces `build_optimizers` to plain AdamW — and says why.
Whether core's defaults should be per-modality is still open.
