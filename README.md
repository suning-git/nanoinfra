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
| `projects/text2motion_cerebellum/` | a Text2Motion → Motion Cerebellum demo that connects a pretrained OMG generator to a Unitree G1 motion tracker. |
| `projects/nano_motion_cerebellum/` | the same G1 control path driven by a 140M `nano_motion` Text2Motion model trained with NanoInfra. |

**Why three.** A framework that claims to be modality-agnostic and ships one modality
has not been tested, it has been asserted. Video and motion do not resemble text and
do not resemble each other — different tokenizers, different data legs, different
objectives — and everything they share is core.

## Install

```bash
# recommended  (uv: https://docs.astral.sh/uv/ , or `pip install uv`)
uv venv && uv pip install -e '.[liger]'

# or, without uv
python -m venv .venv && . .venv/bin/activate && pip install -e '.[liger]'
```

**uv is recommended for a reason other than speed.** It resolves against platform
tags, so on an older glibc it picks package versions that ship a wheel your machine
can use. pip resolves newest-first and only discovers the mismatch when the source
build fails — and that failure names `cython`, or `rustc`, neither of which points at
glibc. On CentOS 7 (glibc 2.17, still common on academic clusters) `pip install -e .`
cannot finish without hand-written constraints, and every new package you add can
reintroduce the problem; `uv pip install -e .` needs none. One caveat: if the machine
has no suitable interpreter, `uv venv` fetches one from GitHub, which a firewalled
network will not allow — supply your own instead, with
`uv venv --python /path/to/python3.12`.

The `liger` extra is what the exemplars pin as their `head_ce` — the
implementation their recorded numbers were measured on (the world model's block
diffusion arm is the one exception: it needs an unreduced per-token loss and uses
`compiled`). It is an extra rather than a
base dependency because nothing in the framework's default path needs it: core's
`head_ce` defaults to `naive`, which depends on no optional package and runs
anywhere. Plain `pip install -e .` installs and runs fine; the exemplars will then
stop with a message naming this line, rather than quietly training a different arm
than the one their numbers came from. See
[`head_ce`](core/training/model_setup.py) for what the three arms are and what each
costs.

Requires Python ≥ 3.12 and a CUDA GPU for training. `compile_trunk` is on by
default and torch.compile's inductor backend compiles C++17, so a **gcc ≥ 9**
toolchain has to be on PATH (set `CC`/`CXX` if the system compiler is older —
CentOS 7 ships 4.8.5, for instance). Train with `compile_trunk=false` to skip it.

`torch>=2.6` is a floor, not a pin — a fresh install resolves to whatever is current,
which need not match the version an exemplar's recorded numbers came from (this tree
develops against 2.12.1). Expect small numeric differences from the numbers in each
`RESULTS.md`. That is deliberate: a framework you are meant to fork should not choose
your torch for you.

## Quickstart — the text exemplar

Run these from the repo root: the exemplars are addressed as modules
(`python -m exemplars.…`), which resolves them relative to the working directory.

```bash
# 1 · fetch a few FineWeb shards (into outputs/base_data/)
python -m exemplars.text_pretrain.data.download_shards

# 2 · train YOUR tokenizer on those shards (seconds; writes outputs/tokenizer/)
python -m modalities.text.train_tokenizer

# 3 · train the model (see the exemplar README for the full train / measure / sample recipe)
CUDA_VISIBLE_DEVICES=0 python -m exemplars.text_pretrain.pretrain
```

Yes, you train the tokenizer yourself — it's a from-scratch framework all the way
down. Step 2 is not optional: without the trained artifact the orchestrator falls
back to the generic gpt2 vocab, warns, and then stops at vocabulary assembly
(`layout/artifact vocab mismatch`) rather than training in a different world than
the exemplar's numbers. Step 2 also writes `token_bytes.pt`, the per-token byte
table behind `val/bpb`; a stream that declares bpb refuses to start without it.

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

The Text2Motion → Motion Cerebellum projects use the
[`suning-git/motion_tracking`](https://github.com/suning-git/motion_tracking)
controller. The baseline uses a pretrained OMG generator; the extension trains
NanoInfra's own `nano_motion` generator and compares both under the same frozen
tracker protocol. Training data includes Motion Data by
[Bones Studio](https://bones.studio/) and separately licensed HumanML3D/AMASS
material. Raw data, checkpoints, codec weights, and derived reference shards are
not redistributed. Compact skeleton comparison videos are included with both
demos.

## License

MIT — see [LICENSE](LICENSE).
