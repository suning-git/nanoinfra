"""
spec.py — THE knob panel: what this exemplar trains, on what data, at what size.

Everything downstream (cache build, dataset, orchestrator, evaluation) reads its
facts from here, so re-targeting the project is a one-line edit rather than a hunt
through five scripts. The pattern follows exemplars/text_pretrain/spec.py.

Three kinds of fact live here, and the distinction matters:

  1. PROTOCOL — the shared-vocab contract (band type ids, delimiter slots, action
     count). Changing one silently invalidates every checkpoint ever trained, so
     train_wm.py asserts the assembled layout still agrees with these constants.
  2. GEOMETRY — clip length, codec grid, and hence the row layout. Derived, not
     guessed: `clip_geometry()` computes the whole row shape from three numbers.
  3. RECIPE — model size, LR, batch. Free to tune; nothing breaks.

What is NOT here: the training loop (train_wm.py) or the objective
(block_diffusion.py). This file only says WHICH.
"""

from pathlib import Path

# --- roots -------------------------------------------------------------------
# The repo's three data roots: datasets/ is input, models/ is weights, outputs/ is
# derived-and-regenerable. Everything this exemplar makes is one of those three.
PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]

DATASET_ROOT = REPO / "datasets" / "nano_world_model"    # everything data/ produces
PIXEL_PARQUET_DIR = DATASET_ROOT / "vizdoom_ppo"         #   downloaded (data/download.py)
PIXEL_SHARD_DIR = DATASET_ROOT / "recorded"              #   self-recorded (data/record.py)
CODES_DIR = DATASET_ROOT / "codes"                       #   encoded (data/encode.py)
MANIFEST = DATASET_ROOT / "manifest.jsonl"               #   one line per encoded shard

CACHE_ROOT = PROJECT / "outputs" / "cache"          # fixed-stride memmaps (build_cache.py)
MODELS_ROOT = PROJECT / "models"                    # checkpoints (gitignored)
CODEC_DIR = REPO / "models" / "video" / "cosmos_dv4x8x8"

TOKENIZER_DIR = REPO / "outputs" / "tokenizer"


def pin_tokenizer():
    """Point modalities.text at the repo's trained tokenizer. Every entry point calls
    this BEFORE importing modalities.text.

    The shared vocab's text band is sized by whatever tokenizer gets loaded, and every
    band offset after it — including the video codes — moves with that size. Without
    the pin, modalities.text looks in its own default location and, finding nothing,
    falls back to gpt2 (vocab 50257 instead of 96786). It says so loudly, but training
    still runs: a different vocabulary, incomparable numbers, and no checkpoint that
    crosses the boundary. Hence a function every entry point calls, not a comment.
    """
    import os
    os.environ.setdefault("NANOINFRA_TOKENIZER_DIR", str(TOKENIZER_DIR))


# --- 1. PROTOCOL: the shared-vocab contract (changing these breaks checkpoints) --
VIDEO_START = "ctrl0"    # delimiters reuse reserved control slots — zero core changes
VIDEO_END = "ctrl1"
MASK_SLOT = "ctrl2"      # [MASK], the absorbing state of the diffusion forward process
VIDEO_TYPE_ID = 4        # 0=text, 1=motion, 2=control, (3=audio reserved), 4=video
ACTION_TYPE_ID = 5       # discrete game actions (world-model conditioning)
N_ACTIONS = 18           # VizDoom deathmatch action set (ids 0..17)


# --- 2. GEOMETRY: the clip and the codec that turns it into tokens -------------
FRAMES = 17              # frames per training clip
RES = 128                # square clips, 128px

# Facts about the frozen Cosmos-Tokenizer DV4x8x8, stated rather than imported.
# Training needs exactly three numbers from a video codec; loading the real
# encoder/decoder here would put a ~1GB TorchScript pair on the GPU to read them.
# The codec ITSELF is only needed to make codes (cache build) or pixels (sampling).
CODEC_NAME = "cosmos_dv4x8x8"
CODEC_VOCAB = 64000      # FSQ codebook
CODEC_SPATIAL_DS = 8     # H/8 x W/8 per latent frame
CODEC_TEMPORAL_DS = 4    # causal: T frames -> 1 + (T-1)/4 latent frames


def clip_geometry(frames=FRAMES, res=RES):
    """Derive the whole row shape from (frames, res, codec). Returns a dict of
    integers; every consumer takes its numbers from here so they cannot disagree.

      codes_per_frame  tokens for one latent frame (the diffusion BLOCK size)
      n_latent         latent frames per clip (frame 0 is the given observation)
      n_blocks         predicted latent frames = n_latent - 1
      td               game frames per latent frame = actions before each block
      code_len         codes per row, i.e. the width of the cache
    """
    cpf = (res // CODEC_SPATIAL_DS) ** 2
    n_latent = 1 + (frames - 1) // CODEC_TEMPORAL_DS
    td = (frames - 1) // (n_latent - 1)
    return {
        "frames": frames, "res": res,
        "codes_per_frame": cpf,
        "n_latent": n_latent,
        "n_blocks": n_latent - 1,
        "td": td,
        "n_given": cpf,                 # exactly one given latent frame (the initial obs)
        "code_len": cpf * n_latent,
        "n_action_tokens": frames,      # one action id per game frame in the cache
    }


def cache_dir(frames=FRAMES, res=RES):
    """Where build_cache.py writes the fixed-stride memmap for a given geometry."""
    return CACHE_ROOT / f"dv{res}_{frames}f"


# --- 3. RECIPE: model size + optimization (tune freely) -----------------------
DEPTH = 12               # transformer depth
DIM = 768                # hidden size
N_HEAD = 12
LR_MAX = 3e-4
SEED = 42

DEVICE_BATCH_SIZE = 4    # rows per GPU per micro-step
TOTAL_BATCH_ROWS = 4     # rows per optimizer step (grad_accum = this / (bs * world))

# Masked-diffusion noise schedule. t is clipped away from 0 because the MDLM 1/t
# weight has unbounded gradient variance as t -> 0 (BD3-LM §4).
T_MIN = 0.2
T_MAX = 1.0

VAL_ROWS = 500           # frozen val rows (capped at what the cache holds)
VAL_T_GRID = (0.3, 0.5, 0.7, 0.9)


def run_name(depth=DEPTH, lr=LR_MAX):
    return f"nwm_{FRAMES}f_d{depth}_lr{lr:g}"


def ckpt_dir(depth=DEPTH, lr=LR_MAX):
    return str(MODELS_ROOT / run_name(depth, lr))
