"""spec.py — THE knob panel: what this exemplar trains, on what data, at what size.

Everything downstream (data preparation, codec training, encoding, the AR
orchestrator, sampling) reads its facts from here. The pattern follows
exemplars/text_pretrain/spec.py and exemplars/nano_world_model/spec.py.

Three kinds of fact live here, and the distinction matters:

  1. PROTOCOL — the shared-vocab contract (band type ids, delimiter slots). Changing
     one invalidates every checkpoint ever trained against it.
  2. REPRESENTATION — rot139, and the codec facts that follow from a tokenizer
     choice. Derived from the tokenizer, not guessed: `codec_facts()` reads them
     off the artifact so this file and the weights cannot disagree.
  3. RECIPE — model size, LR, batch, prompts. Free to tune; nothing breaks.

What is NOT here: the training loop or the objective. This file only says WHICH.
"""

from pathlib import Path

# --- roots -------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]

DATASETS = REPO / "datasets"                 # raw + rot139 format versions
MODELS = REPO / "models" / "motion"          # promoted artifacts (codecs)
CKPTS = PROJECT / "models"                   # this project's AR checkpoints
RESULTS = PROJECT / "results"                # generated .npz / .gif

TOKENIZER_DIR = REPO / "outputs" / "tokenizer"


def pin_tokenizer():
    """Point modalities.text at the repo's trained tokenizer, BEFORE importing it.

    The shared vocab's text band is sized by whatever tokenizer is on disk, and every
    band offset after it — including the motion codes — moves with that size. Without
    a trained artifact, modalities.text falls back to gpt2 (vocab 50257 instead of
    96786) and says so loudly; training still runs, on a different vocabulary, and no
    checkpoint crosses the boundary. Hence a function every entry point calls.
    """
    import os
    os.environ.setdefault("NANOINFRA_TOKENIZER_DIR", str(TOKENIZER_DIR))


# --- 1. PROTOCOL: the shared-vocab contract ----------------------------------
MOTION_TYPE_ID = 1       # 0=text, 1=motion, 2=control, (3=audio reserved), 4=video
MOTION_START = "motion_start"
MOTION_END = "motion_end"


# --- 2. REPRESENTATION --------------------------------------------------------
SOURCE = "lafan1"        # "lafan1" (no captions) or "amass" (captions via HumanML3D)
D_FEAT = 139             # rot139: joint rotations + root displacement/height + contacts
FPS = 30.0

TOKENIZER = "rot139_kin_fsq2"    # which shelf tokenizer; see modalities/motion/tokenizers


def codec_facts(name=TOKENIZER):
    """The facts the AR side needs from a codec, read off the artifact.

    Stated by the checkpoint rather than duplicated here on purpose: a codec swap
    that changed the codebook size while this file still claimed the old one would
    place every motion token at the wrong vocabulary offset, and nothing would raise.
    """
    import importlib
    mod = importlib.import_module(f"modalities.motion.tokenizers.{name}")
    codec = mod.load(device="cpu")
    return {"vocab_size": codec.vocab_size, "downsample": codec.downsample,
            "d_feat": codec.d_feat, "rep": codec.rep}


# --- 3. RECIPE ----------------------------------------------------------------
DEPTH = 12
DIM = 768
N_HEAD = 12
SEQ_LEN = 256            # caption (<= MAX_TEXT_TOKENS) + generated motion codes
MAX_TEXT_TOKENS = 32
LR_MAX = 3e-4
SEED = 42

DEVICE_BATCH_SIZE = 16
TOTAL_BATCH_ROWS = 64

# Sampling
TEMPERATURE = 1.0
TOP_K = 40

# Prompts to generate at the end of a run. Full sentences on purpose: a model trained
# on full-sentence captions goes out of distribution on terse ones, which is a
# measured property of these caption sets rather than a stylistic preference.
PROMPTS = [
    "A person is walking backwards at a normal pace.",
    "A person starts to jog forward at a steady pace.",
    "A person gets down from a box and stands idle.",
    "A person standing idle takes a small hop backward.",
    "The person is in a kneeling posture with both knees resting on the floor.",
    "A person walking forward with their hands behind their back comes to a stop.",
]


def run_name(depth=DEPTH):
    return f"nmo_{SOURCE}_{TOKENIZER}_d{depth}"


def ckpt_dir(depth=DEPTH):
    return str(CKPTS / run_name(depth))
