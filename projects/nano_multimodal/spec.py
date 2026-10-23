"""
spec.py — THE knob panel: which three models this project trains, on what, at what size.

Everything downstream (assembly, configs, decoders, the web) reads its facts from
here, so re-targeting the project is a one-line edit rather than a hunt through
eight files. Same pattern as exemplars/text_pretrain/spec.py.

Three kinds of fact live here, and the distinction matters:

  1. PROTOCOL — the shared-vocab contract: the band table and its canonical type
     ids. Changing one silently invalidates every checkpoint ever trained, so
     assembly.py asserts the assembled layout still agrees with it.
  2. SHAPE — the video clip contract and hence the row length. DERIVED from three
     numbers, never written down twice.
  3. RECIPE — model size, LR, batch. Free to tune; nothing breaks.

THE ONE IDEA THIS FILE ENCODES. There is ONE band table. Each of the three lines
ACTIVATES a subset of it (LINES below). A line pays only for the bands it
activates — the model's vocab_size is the sum of its active bands, not of all of
them. Measured on this box (RTX 5090, d12/768): carrying one dead 32750-wide text
band costs 1.43x the step time — 2.2-2.7x on a naive head, where the un-embedding
has to materialise a [B, T, V] logit tensor. The cost is bandwidth, not FLOPs.
"""

from pathlib import Path

# --- roots -------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]

TOKENIZER_DIR = REPO / "outputs" / "tokenizer"        # the BPE artifact (sizes the text band)

# EVERYTHING THIS PROJECT WRITES GOES UNDER outputs/, and the name is not a
# preference. `projects/*/outputs/` is what this repo's .gitignore already covers —
# the public one too — and it is where the sibling exemplar puts the same kind of
# thing (exemplars/nano_world_model/spec.py: PROJECT / "outputs" / "cache"). A run's
# checkpoints, the video cache, the scaling curves and the eyeball galleries all
# resolve from here, so nothing lands in the project root by accident and nothing
# lands somewhere git is not already told to leave alone.
#
# The rule behind the name: a directory that holds gigabytes of generated weights
# must be one the SHIPPED .gitignore already covers. Inventing a different name for
# it moves the files somewhere git offers to commit them, and the first person to
# find out is whoever runs `git add -A` after a training run.
OUTPUTS = PROJECT / "outputs"
MODELS_ROOT = OUTPUTS / "models"                      # the three checkpoints
SCALING_DIR = OUTPUTS / "scaling"                     # curves + the fitted figure
GALLERY_DIR = OUTPUTS / "gallery"                     # rendered clips, for eyeballing

MOTION_CACHE_DIR = REPO / "outputs" / "motion_caches"  # t2m_*.npz (given, pre-encoded)
# The motion tokenizer. The shelf holds two, both 512 codes, and the rule that
# decides between them turned out to be simple: EACH ONE ONLY WORKS ON THE CORPUS IT
# WAS TRAINED ON. Measured here (root-relative MPJPE, and a jitter reading = the norm
# of joint acceleration in cm/frame^2, which is what "the body shakes" actually is):
#
#   features      codec              height   jitter   MPJPE
#   bones_seed    (raw, no codec)     1.47     0.250     —
#   bones_seed    rot139_kin_fsq2     1.48     0.244    3.67 cm   <- its own corpus
#   bones_seed    rot139_vqvae        1.32     4.161   80.50 cm
#   amass         (raw, no codec)     1.24     1.205     —
#   amass         rot139_kin_fsq2     1.35     3.694   62.26 cm
#   amass         rot139_vqvae        1.26     3.103   16.29 cm   <- its own corpus
#
# So the choice is really a choice of CORPUS. We take Bones-SEED + fsq2, because that
# pair adds no jitter at all (0.244 against the raw 0.250 — the conv autoencoder
# smooths slightly) and reconstructs to 3.67 cm. The AMASS pair is the honest
# alternative and it is four times shakier before any codec touches it: real mocap
# carries real noise, while Bones-SEED is SOMA-retargeted and smooth by construction.
#
# MPJPE alone would not have chosen correctly: 16.4 vs 10.09 cm looks like a 1.6x
# difference, while what the eye reacts to is the TIME DERIVATIVE of that error.
MOTION_CODEC = "rot139_kin_fsq2"
# The video subset. WRITTEN by data/build_video_cache.py and also SHIPPED to students,
# so it sits under outputs/ by the rule above (it is generated, and it must never be
# committed) while being one of the things the release actually packages.
VIDEO_CACHE_DIR = OUTPUTS / "cache"
VIDEO_CODEC_DIR = REPO / "models" / "video" / "cosmos_dv4x8x8"   # decoder only; training never loads it


def pin_tokenizer():
    """Point modalities.text at this repo's trained tokenizer. EVERY entry point
    calls this BEFORE importing modalities.text.

    The text band's width is whatever tokenizer gets loaded, and every band offset
    after it moves with that width. Without the pin, modalities.text finds nothing
    in its default location and falls back to gpt2 (vocab 50257 instead of 32768).
    It says so loudly, but training still runs: a different vocabulary, incomparable
    numbers, and no checkpoint that crosses the boundary. Hence a function every
    entry point calls, not a comment.
    """
    import os
    os.environ.setdefault("NANOINFRA_TOKENIZER_DIR", str(TOKENIZER_DIR))


# --- 1. PROTOCOL: the band table (changing these breaks checkpoints) ----------
# Canonical type ids are GLOBAL across all three lines and never renumbered. That
# is the property the browser needs: token IDs are per-line (a compact layout pays
# only for its own bands), but a token of type 4 is a video token on every line, so
# ONE colour scheme is correct everywhere.
#
# n_token_types is DERIVED per line, not fixed at 6 — VocabLayout sizes it as
# max(active type_id) + 1 and allows gaps, so the text line gets 3 (ids 0 and 2,
# with row 1 unused) and the video line gets 6 (ids 2, 4, 5). The gap rows are
# zero-initialised and never indexed; they cost n_embd floats each.
TEXT_TYPE_ID = 0
MOTION_TYPE_ID = 1
CONTROL_TYPE_ID = 2
# 3 = audio, reserved
VIDEO_TYPE_ID = 4
ACTION_TYPE_ID = 5
N_TOKEN_TYPES = 6

# Band widths that are FACTS OF AN ARTIFACT are read off that artifact at assembly
# time (text from the BPE tokenizer, motion from the codec). The two below are
# stated instead, because reading them would mean loading a ~600MB TorchScript
# pair (video) or a table that exists only in code (action).
VIDEO_VOCAB = 64000          # frozen Cosmos-Tokenizer DV4x8x8, FSQ codebook
N_ACTIONS = 19               # VizDoom action table v2 (ids 0..18; 18 = NOOP)

# Action-table version 2, and the names BY ID. The ORDER is data protocol: every
# cache's action ids index this list, and it must match whatever produced the data.
# v1 was the downloaded PPO set's 18 button combos; v2 appends exactly one id —
# NOOP = 18, a true stand-still (without it a recorded policy can never hold still,
# so the model never learns what an unpressed world does). Appending at the tail is
# what makes the break cheap: ids 0..17 keep their meaning.
# TL/TR = turn, ML/MR = strafe, FWD = forward, ATK = attack.
ACTION_TABLE_VERSION = 2
ACTION_NAMES = ["TL", "TR", "MR", "MR+TL", "MR+TR", "ML", "ML+TL", "ML+TR",
                "FWD", "FWD+TL", "FWD+TR", "FWD+MR", "FWD+MR+TL", "FWD+MR+TR",
                "FWD+ML", "FWD+ML+TL", "FWD+ML+TR", "ATK", "NOOP"]
assert len(ACTION_NAMES) == N_ACTIONS

# Widths that ARE readable off an artifact are declared here too, and assembly.py
# asserts the artifact against the declaration every time it loads one. The
# declaration is what lets a line that does not activate a band still draw the full
# table (the browser's vocab panorama); the assert is what keeps the two honest.
TEXT_VOCAB = 32750           # BPE content ids; = tokenizer vocab (32768) - 18 specials
MOTION_VOCAB = 512           # rot139_kin_fsq2, FSQ2 [8,8,8]; asserted against the cache sidecar
MOTION_DOWNSAMPLE = 4        # frames per motion code at 30fps — "one integer = 1/8 second"

# How big each line's media is rendered. ONE place, because both panels must draw the
# same picture: the browser shows a row from the dataset and the inference panel shows
# a row the model wrote, and a student compares them. They already share the renderer
# — but sharing a FUNCTION does not share its ARGUMENTS, and for a while the
# inference panel quietly passed size=192 while the browser used 300. At 192 the
# matplotlib 3D box's panes and ticks crowd out the figure, so the same model looked
# worse on one tab than on the other. A caller that wants a different size (an offline
# gallery) passes it explicitly; the two panels never do.
RENDER_SIZE = 300

# Delimiters. The control registry (modalities/control) ships four unnamed reserved
# slots, ctrl0..ctrl3, precisely so a project can claim one without editing core.
# Video claims two.
VIDEO_START = "ctrl0"
VIDEO_END = "ctrl1"
MOTION_START = "motion_start"
MOTION_END = "motion_end"

# ...but `ctrl0` is not a name a student should have to decode while reading a row
# template. This project's recipes say what they mean, and assembly.py wraps the
# control resolver with this table so the yaml stays readable and the protocol stays
# core's. The alias is one-way and local: the registry is unchanged, and a checkpoint
# does not know these names exist.
CONTROL_ALIASES = {
    "video_start": VIDEO_START,
    "video_end": VIDEO_END,
}

# THE DECLARATION ORDER. A line stacks its active bands in THIS order, so band
# offsets are a function of (order, active set) and nothing else.
#
# control sits DIRECTLY AFTER text, and that is load-bearing: the BPE artifact
# physically reserves its 18 special tokens at the tail of its own id space
# (32750..32767). Putting any other band between them would give a control token a
# different id here than the tokenizer itself assigns it — self-consistent, but a
# trap for anything that reads the tokenizer directly. With this order the text
# line's layout matches the artifact exactly (vocab 32768) and the motion band
# lands at 32768, which is also where exemplars/nano_motion puts it.
BAND_ORDER = ["text", "control", "motion", "video", "action"]

# Which bands each line activates. This IS the difference between the three
# models. What is identical: the GPT trunk, the LM head, next-token cross-entropy,
# MixedDataLoader and core's Trainer — diff the three configs and nothing else
# moves.
LINES = {
    "text":   ["text", "control"],
    "motion": ["text", "control", "motion"],
    "video":  ["control", "video", "action"],
}


# --- 2. SHAPE: the video clip contract ---------------------------------------
# Facts about the frozen Cosmos DV4x8x8, stated rather than imported.
CODEC_NAME = "cosmos_dv4x8x8"  # the tag a video cache must carry in its meta.json
CODEC_SPATIAL_DS = 8         # H/8 x W/8 per latent frame
CODEC_TEMPORAL_DS = 4        # causal: T frames -> 1 + (T-1)/4 latent frames
FRAMES = 17                  # game frames per training clip
RES = 128                    # square clips, 128px


def video_shape(frames=FRAMES, res=RES):
    """Derive the whole video row from (frames, res, codec). Every consumer takes
    its numbers from here, so they cannot disagree.

      codes_per_frame  tokens for one latent frame
      n_latent         latent frames per clip (frame 0 is the GIVEN observation)
      n_blocks         predicted latent frames = n_latent - 1
      td               game frames per latent frame = action ids between frames
      row_len          the assembled row, delimiters included
    """
    cpf = (res // CODEC_SPATIAL_DS) ** 2                 # 256
    n_latent = 1 + (frames - 1) // CODEC_TEMPORAL_DS     # 5
    td = (frames - 1) // (n_latent - 1)                  # 4
    n_action = (n_latent - 1) * td                       # 16 action ids actually used
    # [bos, video_start, L0, a*td, L1, a*td, L2, a*td, L3, a*td, L4, video_end, eos]
    row_len = 4 + n_latent * cpf + n_action
    return {"frames": frames, "res": res, "codes_per_frame": cpf, "n_latent": n_latent,
            "n_blocks": n_latent - 1, "td": td, "n_action": n_action,
            "code_len": n_latent * cpf, "row_len": row_len}


def video_cache_dir(frames=FRAMES, res=RES):
    return VIDEO_CACHE_DIR / f"dv{res}_{frames}f"


# --- 3. RECIPE: model size per line (tune freely) ----------------------------
# Sizes are a starting point, not a measured optimum. The motion model is
# deliberately small, and the reason is measured: Bones-SEED holds 128,679 clips x 3
# captions = 450,594 rows = 30.4M supervised motion tokens per epoch, while a 36.2M
# d6 wants ~720M by Chinchilla — about 24 passes over the same clips. What that
# buys is visible on this line: d12 reaches its floor at step 5,500 and drifts up
# for the rest of the run, while d6 is still descending at 16,000 and never gets
# beaten. The corpus is the ceiling, not the model — that is the lesson of this
# line, not a defect.
RECIPES = {
    "text":   {"depth": 12, "dim": 768, "seq": 512},
    "motion": {"depth": 6,  "dim": 384, "seq": 256},
    "video":  {"depth": 12, "dim": 768, "seq": None},   # None -> video_shape()["row_len"] - 1 positions
}

SEED = 42


def run_name(line, depth):
    return f"nmm_{line}_d{depth}"


def ckpt_dir(line, depth):
    return str(MODELS_ROOT / run_name(line, depth))


# Prompts the inference web offers for the motion line. The first three are the ones
# the archived reference gallery used on this exact corpus and codec, so they are
# known to produce something worth looking at; the fourth is a Bones-SEED caption verbatim. The web
# marks which of them appear in the TRAINING set, because a training caption that
# generates beautifully is memorisation, and that distinction is the lesson.
MOTION_PROMPTS = [
    "a person walks forward",
    "a man kicks with his left leg",
    "a person jumps",
    "a person quickly hops forward on their right foot.",
]
