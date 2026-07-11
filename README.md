# nanoinfra

Modality-agnostic training infrastructure for decoder-only transformers — small
enough to read end to end, real enough to train a 135M-parameter language model
and measure its compute-optimal scaling law.

The core is **zero-modality**: it holds only the mechanisms — the GPT trunk, a
pluggable-head factory, the `Trainer`, FSDP + resumable checkpointing, tokenizer
plumbing, and a KV-cache inference engine — and never imports a modality.
Modalities (text, and your own) plug in as declarative manifests that an assembler
stacks into one shared vocabulary. Adding a modality — or a new head, or a
different model trunk — is a plug-in, not a fork of the core.

## What's inside

| dir | what |
|-----|------|
| `core/` | the mechanisms, zero modality knowledge — model (GPT / attention / RoPE / KV-cache), pluggable heads, `Trainer`, tokenization, data pipeline, evaluation. **`core` never imports a modality.** |
| `modalities/` | per-modality implementations + the `assembler` that wires them into a shared vocab. Ships with `text`. |
| `exemplars/text_pretrain/` | a complete, minimal text-LM project — train a 135M model to val CE 3.81, measure the compute-optimal scaling law (exponent a ≈ 0.52), sample from it. The reference you fork. |
| `projects/` | where your own work goes — a fork of an exemplar with one thing changed. |

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Requires Python ≥ 3.12 and a CUDA GPU for training.

## Quickstart — the text exemplar

```bash
# 1 · fetch a few FineWeb shards (into outputs/base_data/)
python exemplars/text_pretrain/data/download_shards.py

# 2 · train the model (see the exemplar README for the full train / measure / sample recipe)
CUDA_VISIBLE_DEVICES=0 python exemplars/text_pretrain/pretrain.py
```

The exemplar's [`README.md`](exemplars/text_pretrain/README.md) walks the whole
lifecycle — train → compute-optimal scaling law → inference — and
[`RESULTS.md`](exemplars/text_pretrain/RESULTS.md) pins exactly what it produces.

## Layout — three roots

Code stays separate from data. Three top-level roots hold everything that isn't
source: `datasets/` (raw corpora), `models/` (checkpoints), `outputs/` (run
artifacts). Set `NANOINFRA_BASE_DIR` to relocate `outputs/`.

## Credits

The model and training loop descend from Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) / nanochat lineage.

## License

MIT — see [LICENSE](LICENSE).
