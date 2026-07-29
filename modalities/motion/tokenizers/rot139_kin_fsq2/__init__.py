"""
rot139_kin_fsq2 — a shelf tokenizer (façade).

rot139 features -> 512 FSQ2 codes (downsample 4). Reconstruction champion of the
matched-budget arbitration (fair global 10.09 cm / 4.2°); GENERATION not yet
eyeball-verified. Family: _convae. Recipe + certificate: recipe.yaml / card.md.

This is the user entry — call load(); you do not touch _convae/.
"""

import os

import yaml

from modalities.motion.tokenizers._convae import MotionCodec

_HERE = os.path.dirname(os.path.abspath(__file__))
# Weights live under the repo's models/ root (gitignored, large); this folder holds
# only a pointer. The default is derived from where this package sits, so it is not
# tied to any one checkout.
_MODELS = os.environ.get("NANOINFRA_MODELS_DIR") or os.path.join(
    os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")), "models")
WEIGHTS = os.path.join(_MODELS, "motion", "codec_rot139_kin_fsq2_bones_seed_30k.pt")


def load(device: str = "cpu") -> MotionCodec:
    """Load this tokenizer. Returns a codec satisfying the shelf contract
    (.vocab_size / .downsample / encode / decode). See card.md."""
    return MotionCodec(WEIGHTS, device=device)


def recipe() -> dict:
    """This tokenizer's training recipe (from recipe.yaml)."""
    with open(os.path.join(_HERE, "recipe.yaml")) as f:
        return yaml.safe_load(f)


def train(out_path: str = WEIGHTS, device: str = "cuda", **overrides):
    """Reproduce this tokenizer from recipe.yaml (override any field via kwargs,
    e.g. steps=200 for a smoke run). Saves to out_path; returns fair-eval metrics."""
    from modalities.motion.tokenizers._convae.train import train_codec
    return train_codec({**recipe(), **overrides}, out_path, device=device)
