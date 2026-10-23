# nano_world_model — a discrete video world model, at exemplar quality

Train a transformer to answer: *given what I have seen and the buttons I pressed,
what does the world look like next?* The world is VizDoom; the model sees the game
as discrete tokens, exactly the way this framework sees text.

Everything from raw game frames to generated video is here — recording or downloading
pixels, tokenizing them, training a block-diffusion world model on core's data
pipeline, and generating clips from it action by action. It is also the first user of
`core.parallel.NanoDDP`, the replicated data parallelism this project built and
merged into core.

**Block diffusion, on core's pipeline.** Each latent frame is one diffusion block:
masked cells are predicted from the clean past and the block's other cells, so a whole
frame is filled in a handful of forwards instead of one per token. Training runs on
core's per-sample data path — a source, a per-clip preparation, a batcher, core's
loader — and a custom System, with no Trainer subclass; generation asks the model the
question training scored (`inference/`). An autoregressive model over the same rows is
`projects/nano_multimodal`'s video line.

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

Frame 0 is the given observation; the other four are predicted, by **absorbing-state
(masked) diffusion**. The forward process masks each token of a block with probability
`t ~ U[0.2, 1]`; the model sees the clean prefix plus the partially masked block and
predicts the clean token at each masked position. One forward pass covers every block,
using BD3-LM's two-stream trick: the sequence is `[clean row | noisy row]` and the
attention mask lets noisy block *k* read the clean prefix before its own start plus
itself bidirectionally. Nothing supervises the given frame (it is the observation) or
the action tokens (they are the control input, and learning which button a player
presses is a different problem).

The training loss is each block's mean CE over the cells its draw masked, averaged over
the blocks; the val number is the same quantity at a fixed grid of noise levels on
frozen val clips (`val/nelbo`), in nats per predicted token.

## Layout

| file | what it owns |
|---|---|
| `spec.py` | the knob panel: protocol ids (action table v2), the shape contract, recipe. One edit re-targets everything. |
| `data/` | **pixels to codes** — download or record (a recipe-driven corpus machine), tokenize, and the format contract. Has its own README. |
| `build_cache.py` | one-shot: encoded shards → a fixed-stride memmap. |
| `dataset.py` | the memmap as a map-style dataset, and `ClipSource`: the DataSource core's loader reads — resumable on any number of GPUs. |
| `row_layout.py` | where every token sits. |
| `rope3d.py` | the position regime: true (t, y, x) rotary coordinates, one table swap. |
| `block_diffusion.py` | the objective: a clip prepared into its two-stream sample, the mask built from a batch, the loss — and the `System` that carries it. |
| `evaluator.py` | the frozen ruler, on core's loss evaluation. |
| `train_wm.py` | **the orchestrator** — read this first. |
| `configs/train_wm.yaml` | this project's knobs; mechanism defaults come from core. |
| `inference/` | generation: the sampler, a script that renders generated clips next to the truth, and a benchmark that checks the sampler against training before timing it. |
| `tests/` | the sampler against training, the pipeline's parts, evaluation and multi-GPU correctness (below). |

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

# 4. train, on one GPU
python -m exemplars.nano_world_model.train_wm

# ... or on two, replicated, with NanoDDP
torchrun --nproc_per_node=2 --standalone \
    -m exemplars.nano_world_model.train_wm parallel=ddp

# 5. what did it learn? generated val clips next to the truth, as GIFs
python -m exemplars.nano_world_model.inference.rollout --ckpt <checkpoint step dir>

# 6. does the sampler compute what training scored, and how fast does it generate?
python -m exemplars.nano_world_model.inference.benchmark
```

Step 1 has an alternative: record your own —

```bash
python -m exemplars.nano_world_model.data.record.run --tag rec1 --seed 0 --minutes 30
python -m exemplars.nano_world_model.data.encode --source recorded
python -m exemplars.nano_world_model.build_cache --force   # the cache is rebuilt from the manifest; --force if step 3 already ran
```

which gives ground-truth actions and unlimited data at the cost of
`pip install vizdoom`. The recorder is driven by a corpus RECIPE
([data/recipes/minrec.yaml](data/recipes/minrec.yaml)) — layer shares, coverage
quotas, wall avoidance — because what you record decides what the model can
learn; the recipe header carries the reasons. See
[data/README.md](data/README.md) for what each path costs and what it can and
cannot produce.

## The real run: a 129-frame model

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

A checkpoint written before this exemplar ran on core's data pipeline (2026-10) still
loads its weights — `checkpoint.init_model_from`, as in stage 2 — but cannot be resumed
in place: its data position is the old loader's, which core's loader refuses.

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

To see the result, `inference/rollout.py --ckpt <dir> --frames 129` generates val
clips from their first frame and recorded actions and writes them next to the truth;
`inference/sampler.py` is the loop to build an interactive player on.

---

## Four decisions worth knowing about

### 1. The data is a memmap, so the sampler can be core's

A loader that walks compressed shards has to keep one shard resident and serve its
rows before moving on, so its checkpoint state is **which shard, which rank, which
position** — hardware-dependent. A resumed rank 1 cannot read rank 0's shard list
(only rank 0 writes `meta.json`), so under DDP the position restore has to be skipped
entirely.

Every row of a given geometry has the same width, so `build_cache.py` writes one
flat binary per field and `dataset[i]` becomes a slice. Random access costs a page
fault, which lets core's `ResumableDistributedSampler` drive the run — and its
position is `{seed, epoch, index}`, global, with no rank in it. `ClipSource` is a
DataSource over it, so core's loader keeps the snapshot: at every save it merges the
ranks' positions into the furthest one plus the clips behind it that no rank has
trained. Consequences: **a run checkpointed on 2 GPUs resumes on 1**, or on 4, and
every clip is trained once, none skipped or repeated. The loader was the problem, not
the parallelism.

### 2. A clip's noise is drawn with the clip

The masks are drawn when a clip is prepared, before batching, from a generator seeded
by the clip itself — `(seed, epoch, row)`. So a clip gets the same noise on any rank,
in any batch and after a resume, and a different one in its next epoch; and a 2-GPU
step and a 1-GPU 2-accumulation step over the same clips draw identical noise — which
is what makes the DDP equivalence test tight enough to be worth running. The prepared
sample carries what the step needs (targets, loss weights, token types, and the
per-position facts the attention rules read), so the mask is built from the batch, not
from offsets fixed in advance.

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

### 4. Generating asks the training question

Training asks one question: given the clean past and a block with some cells masked,
what are the masked cells? Generating latent frame *k* asks exactly that, starting
from a fully masked block (`inference/sampler.py`): write the actions that drive it and
run the prefix into a KV cache; then K times run the block over the cache —
bidirectionally inside it, `attention="bidirectional"` — draw a code for every masked
cell and commit the most confident ones on a cosine schedule (MaskGIT), rewinding the
cache so the half-finished block never persists (`KVCache.rewind`); finally run the
finished block causally into the cache, as the clean copy of a training row holds it.

That forward sees what the noisy copy of a training row sees — the clean past up to the
block's start, the block itself both ways — at the same rotary positions, so its logits
are the ones training scored. `tests/test_sampler.py` checks it in float32 on the CPU,
against the training forward, to rounding; `inference/benchmark.py` checks it at full
size under bf16 before it reports speed. A sampler that is fast and subtly wrong is
worse than a slow one.

## Tests

```bash
pytest exemplars/nano_world_model/tests                       # CPU
torchrun --nproc_per_node=2 --standalone \
    -m exemplars.nano_world_model.tests.test_ddp               # two GPUs
```

| test | what it would catch |
|---|---|
| `test_sampler.py` | a sampler whose forward sees other positions, rotations or token types than training did |
| `test_block_diffusion.py` | a mask from the batch's fields that differs from the two-stream rules anywhere, padding that leaks, targets or weights in the wrong cells, a loss that is not the mean of block means |
| `test_clip_source.py` | a clip tagged with the wrong epoch (wrong noise), a state that does not read back from JSON, a resume on another number of GPUs that skips or repeats clips |
| `test_wm_eval_shard.py` | an evaluation that reads differently on two processes than on one |
| `test_ddp.py`: NanoDDP == backward-then-all_reduce, bitwise | an overlapped reduction that drops or misorders a bucket |
| `test_ddp.py`: 2 ranks == 1 rank with `grad_accum=2`, over the same micro-batches | a batch that is split but not averaged the way you think |
| `test_ddp.py`: a bucket that receives no gradient **raises** | replicas drifting apart with training happily continuing |

None of these shows up as a crash, which is the reason to test them rather than trust
them. See `RESULTS.md` for what the numbers should look like when it is working.

## Open question

**core's optimizer defaults are text-tuned.** `build_optimizers` uses role-based
multi-LR groups (`embedding_lr` 0.2, `unembedding_lr` 0.004). Measured head to head
on this model they cost **0.34 nat at 10k steps**, so `configs/train_wm.yaml` sets all
three LRs equal — which reduces `build_optimizers` to plain AdamW — and says why.
Whether core's defaults should be per-modality is still open.
