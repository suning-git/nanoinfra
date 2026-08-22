# nanoinfra

Modality-agnostic training infrastructure for decoder-only transformers — small
enough to read end to end, real enough to train a 135M-parameter language model
and measure its compute-optimal scaling law.

The core is **zero-modality**: it holds only the mechanisms — the GPT trunk, a
pluggable-head factory, the `Trainer`, data and tensor parallelism, resumable
checkpointing, tokenizer plumbing, and a KV-cache inference path that CUDA graphs
can replay — and never imports a modality. Modalities (text, motion, and your own)
plug in as declarative manifests that an assembler stacks into one shared
vocabulary. Adding a modality — or a new head, or a different model trunk — is a
plug-in, not a fork of the core.

Three worked examples ship with it, and they are the point: one language model, one
video world model, one text-to-motion model, over the same core, the same vocabulary
assembler and the same `Trainer`.

## What's inside

| dir | what |
|-----|------|
| `core/` | the mechanisms, zero modality knowledge — model (GPT / attention / RoPE / KV-cache), pluggable heads, `Trainer`, tokenization, data pipeline, evaluation. **`core` never imports a modality.** |
| `modalities/` | per-modality implementations + the `assembler` that wires them into a shared vocab. Ships with `text` and `motion`. |
| `exemplars/text_pretrain/` | a complete, minimal text-LM project — train a 135M model to val CE 3.81, measure the compute-optimal scaling law (exponent a ≈ 0.52), sample from it. The reference you fork. |
| `exemplars/nano_world_model/` | a video world model: VizDoom as discrete tokens, trained either autoregressively or with block diffusion over one pipeline, and decoded at ~1.1 ms/token. |
| `exemplars/nano_motion/` | text → human motion: train the motion tokenizer, then a model that turns a caption into a moving skeleton. |
| `projects/` | where your own work goes — a fork of an exemplar with one thing changed. |

**Why three.** A framework that claims to be modality-agnostic and ships one modality
has not been tested, it has been asserted. Video and motion do not resemble text and
do not resemble each other — different tokenizers, different data legs, different
objectives — and everything they share is core.

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

# 2 · train YOUR tokenizer on those shards (seconds; writes outputs/tokenizer/)
python -m modalities.text.train_tokenizer

# 3 · train the model (see the exemplar README for the full train / measure / sample recipe)
CUDA_VISIBLE_DEVICES=0 python exemplars/text_pretrain/pretrain.py
```

Yes, you train the tokenizer yourself — it's a from-scratch framework all the way
down. Skipping step 2 falls back to the generic gpt2 vocab with a loud warning:
training still runs, but in a different world than the exemplar's numbers.

The exemplar's [`README.md`](exemplars/text_pretrain/README.md) walks the whole
lifecycle — train → compute-optimal scaling law → inference — and
[`RESULTS.md`](exemplars/text_pretrain/RESULTS.md) pins exactly what it produces.

The other two start from their own data. `nano_world_model` can borrow a public
VizDoom recording, or make its own: install the game and it records from a corpus
RECIPE — which behaviours to spend frames on, how long a turn is held, how much
true stillness — because what a world model can learn is decided by what the
recording contains. Recording is also the only way to the long-window model the
exemplar's README builds toward: the public set is sub-sampled into fragments of
at most ~59 frames, so a 129-frame clip cannot be cut from it at all. `nano_motion`
runs on freely downloadable motion capture, and adds text conditioning if you
accept the licences for AMASS and HumanML3D. Each has a `data/README.md` covering
what to fetch, what it costs, and what will bite you.

Every exemplar's `RESULTS.md` opens with numbers you can reproduce in about a minute,
so you can tell a broken setup from a slow one before spending a GPU-day on it.

## Multi-GPU

`parallel=ddp` replicates the model and reduces gradients during backward;
`parallel=fsdp` shards parameters. Which one wins is a property of the model rather
than a preference: while the model fits, the bottleneck is activations, so sharding
parameters costs an all-gather per block and buys nothing. Measured on 2x RTX 5090,
a 1.45B text model at sequence length 8192 reaches 93.5% MFU replicated.

```bash
torchrun --nproc_per_node=2 --standalone \
    -m modalities.text.train_text --config-name train_text_1p4b
```

## Layout — three roots

Code stays separate from data. Three top-level roots hold everything that isn't
source: `datasets/` (raw corpora), `models/` (checkpoints), `outputs/` (run
artifacts). Set `NANOINFRA_BASE_DIR` to relocate `outputs/`.

## Credits

The model and training loop descend from Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) / nanochat lineage.

## License

MIT — see [LICENSE](LICENSE).
