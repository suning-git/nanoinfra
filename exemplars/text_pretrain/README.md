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

From the repo root:

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
