# exemplar: text_pretrain

A **complete, minimal text-LM project** — the reference you fork to start your own.
It trains a 135M-param model to val CE 3.81, measures its scaling law, and runs it
as a language model: the whole lifecycle in one small folder.

This is the `exemplars/` tier — distinct from `projects/` (research, throwaway) and
from the reusable building blocks in `core/` and `modalities/`. An exemplar is what
you point a newcomer at: *"this is how we train a text model, end to end, and here
is exactly what it produces."*

## The pipeline — one knob, three stages

Edit **`spec.py`** to re-target the whole project: change the model there and every
stage follows (you never hunt for the model definition scattered across scripts).

| file | stage | what it does |
|------|-------|--------------|
| **`spec.py`** | — | ★ the model + recipe — **the one place you edit** |
| **`pretrain.py`** | 1 · train | trains the champion |
| **`scaling.py`** | 2 · measure | compute-optimal scaling law (per-model curves → frontier) |
| **`inference.py`** | 3 · run | samples text from the champion (core KV-cache engine) |
| *(posttrain.py)* | 4 · align | *future slot — SFT / preference, same rhythm* |

Training itself is **not reimplemented here**. `pretrain.py` and `scaling.py` drive
the blessed text Orchestrator **`modalities.text.train_text`** — a maintained
building block that assembles the modality manifests into a shared vocab, wires the
data source + evaluator into the core `Trainer`, and runs. **To read the full
assemble→train flow, open that file**; this project only picks knobs and drives it.

## How to run

From the repo root. One-time prerequisites (fresh clone):

```bash
python exemplars/text_pretrain/data/download_shards.py   # FineWeb shards -> outputs/base_data/
python -m modalities.text.train_tokenizer                # tokenizer artifact (seconds) -> outputs/tokenizer/
```

```bash
# 1 · train the champion  (~4.4 h on one 5090; the checkpoint may already exist)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exemplars/text_pretrain/pretrain.py

# 2 · the compute-optimal scaling law — split the depths across both GPUs, then fit:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exemplars/text_pretrain/scaling.py run --depths 8 6
CUDA_VISIBLE_DEVICES=1 .venv/bin/python exemplars/text_pretrain/scaling.py run --depths 4 3 2
.venv/bin/python exemplars/text_pretrain/scaling.py fit        # -> scaling_law.png + scaling.json

# 3 · sample from the champion
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exemplars/text_pretrain/inference.py
```

## Multi-GPU: ddp vs fsdp, measured

The launcher decides the world size — the orchestrator reads `RANK`, which only
torchrun sets — and `--parallel` picks the placement:

```bash
.venv/bin/python exemplars/text_pretrain/pretrain.py --nproc 2 --parallel ddp             # full run
.venv/bin/python exemplars/text_pretrain/pretrain.py --nproc 2 --parallel ddp --smoke 30  # 30-step smoke
```

`ddp` replicates the model on every rank and all_reduces gradients DURING
backward (`core/parallel/nano_ddp.py`); `fsdp` shards parameters per block. How
much of a step is communication depends on the gradient-accumulation count,
which `total_batch_size` sets: at the default 131072 (dbs 16 · seq 512 ·
2 ranks) that is grad_accum 8 — gradients cross the link once per 8
micro-batches. `total_batch_size=16384` makes it 1, where the overlap has real
work to do.

Measured on this recipe (d12 / 135M, 2× RTX 5090 PCIe no-NVLink, 30-step smokes,
median of steps 5–29; run 2026-07-29 in a since-retired fork of this driver —
the numbers, not the fork, are the artifact):

| arm | total_batch_size | grad_accum | ms/step | tok/s (global) | MFU/GPU |
|---|---|---|---|---|---|
| **ddp, 2 GPU** | 131072 | 8 | 382 | **343,454** | 58.8% |
| ddp, 2 GPU | 16384 | 1 | 52 | 317,011 | 54.3% |
| fsdp, 2 GPU | 131072 | 8 | 428 | 306,447 | 52.5% |
| fsdp, 2 GPU | 16384 | 1 | 54 | 300,769 | 51.5% |
| 1 GPU (baseline) | 131072 | 16 | 750 | 174,732 | 59.8% |

Read off it: scaling efficiency **98.3%** at grad_accum 8 (343,454 / 2×174,732),
92.7% at grad_accum 1 — the difference is the once-per-step all_reduce becoming
visible as amortization goes away. And **ddp beats fsdp by ~12%** on the same
recipe when the model fits comfortably on one card — the replicate-while-it-fits
rule, reproduced on the text family.

**A trap, not a result**: `--nproc 1 --parallel ddp` measures 129,746 tok/s and
it is NOT a ddp fact — that combination runs the trunk uncompiled (whole-graph
compile is off for ddp at any world size; the per-block replacement installs
only at world > 1). `pretrain.py` warns if you ask for it; baseline = leave
`--parallel` unset.

To see WHERE the overlap happens — per-bucket all_reduce timing against
per-block backward compute, CUDA events on both streams — run the probe
`modalities/tests/ddp_timeline.py` (1 and 2 GPU variants in its docstring).
Headline from its measurement: 19.7 ms of link traffic hides behind compute at a
cost of only 4.4 ms of extra backward — 83% hidden; the un-hideable tail (block 0
+ embedding gradients finalize last) is structural, not an implementation flaw.

## What it produces

Pinned capability numbers live in [`RESULTS.md`](RESULTS.md); [`provenance.md`](provenance.md)
records how the recipe's `lr_max` was chosen. In short: a d12 / 135M model at
**val CE 3.81**, a compute-optimal **frontier** over per-model training curves recovering the
exponent **a ≈ 0.52** (Chinchilla ~0.5; see RESULTS §2), and coherent English continuations.

## Layout

```
spec.py          the model + recipe — the one knob you turn
pretrain.py      stage 1 · train the champion
scaling.py       stage 2 · compute-optimal scaling law (per-model curves + frontier)
scaling_fit.py   the frontier fitter: lower envelope + N_opt ∝ C^a slope (used by scaling.py)
inference.py     stage 3 · sample the champion (loads via core load_system —
                 the checkpoint self-describes its architecture)
provenance.md    how lr_max=3e-4 was chosen (the LR bracket) + how to re-tune
inference_compare.md  the budget ladder: one spec, four budgets, samples side by side
data/            download_shards.py — fetch the FineWeb shards
scaling_law.png  the headline figure (stage 2 output)
results/         scaling.json · samples.md · bracket.json
RESULTS.md       the pinned capability log
```

## Data

FineWeb `sample-10BT` parquet shards, streamed on the fly through
`modalities/text/fineweb.py`. `data/download_shards.py` fetches them into
`outputs/base_data/`. Six shards (≈4.36B tokens) back this project: 5 train
(≈3.63B, 1.34× the 2.705B Chinchilla budget → single-epoch) + 1 val.
