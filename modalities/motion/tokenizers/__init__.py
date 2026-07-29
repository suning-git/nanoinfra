"""
motion.tokenizers — the classic-tokenizer SHELF (R7, 2026-07-07).

Layout:
  <recipe>/     a tokenizer = a self-serve FAÇADE (thin load()/train()) + recipe
                + card (birth certificate). Open the folder, call load(), use it —
                you never navigate the architecture machinery.
  _convae/       an architecture FAMILY (conv autoencoder): nets, quantizers, the
                self-describing MotionCodec, the training harness. A different
                architecture is a sibling family folder, not an edit inside _convae.
  REGISTRY.md   the human index of artifacts + tokenizer-dependent caches.

THE CONTRACT — every folder's load() returns an object exposing:
    .vocab_size            int    (code count = the motion band width)
    .downsample            int    (frames per code)
    encode(features)       -> list[int]  local codes
    decode(codes)          -> np.ndarray native features
That is all the assembler / data sources / evaluators rely on.

Usage:
    from modalities.motion.tokenizers import rot139_kin_fsq2
    codec = rot139_kin_fsq2.load(device="cuda")
    codes = codec.encode(features139)

Back-compat: the class-level API (MotionCodec, MotionFSQ2, MotionVQVAE,
save_checkpoint) is re-exported here from _convae so existing importers
(`from modalities.motion import MotionCodec`) keep working — but new user code
should go through a tokenizer folder's load(), not MotionCodec directly.
"""

from modalities.motion.tokenizers._convae import (
    FSQ2,
    FSQ_LEVELS,
    MotionCodec,
    MotionFSQ2,
    MotionVQVAE,
    VectorQuantizerEMA,
    save_checkpoint,
)
from modalities.motion.tokenizers import rot139_kin_fsq2, rot139_vqvae

__all__ = [
    "rot139_kin_fsq2", "rot139_vqvae",
    "MotionCodec", "save_checkpoint",
    "MotionFSQ2", "MotionVQVAE",
    "FSQ2", "VectorQuantizerEMA", "FSQ_LEVELS",
]
