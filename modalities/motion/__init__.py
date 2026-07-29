"""
motion — the motion content modality.

A self-sufficient modality package (R6, 2026-07-07):
  - tokenizers/  the classic-tokenizer shelf (codec networks + MotionCodec
                 loader + REGISTRY.md of trained artifacts w/ birth certificates)
  - data/        the reusable data leg (SMPL/BVH<->rot139 converters, FK,
                 dataset loaders, windowing, the core DataSources)
  - manifest()   the assembly registration form (this file)

Research lines USE these as a user (import to load/encode/benchmark); they
promote MATURE work back in as a developer, but do day-to-day research on
COPIES in their own project (DESIGN §1). FakeMotionCodec stays for CPU-only
tests (no checkpoint / no torch weights needed to exercise a third band).
"""

from modalities.assembler import Modality

from modalities.motion.tokenizers import (
    FSQ2,
    MotionCodec,
    MotionFSQ2,
    MotionVQVAE,
    VectorQuantizerEMA,
    save_checkpoint,
)

TYPE_ID = 1  # canonical (0=text, 1=motion, 2=control); see D1.2


class FakeMotionCodec:
    """Stand-in LOCAL-ID producer: local code space [0, n_codes)."""

    def __init__(self, n_codes: int = 64):
        self.vocab_size = n_codes

    def encode(self, codes):
        return [int(c) for c in codes]   # codes already are local ints

    def decode(self, codes):
        return list(codes)


def manifest(codec) -> Modality:
    """codec = a LOCAL-ID producer with .vocab_size (MotionCodec or the fake)."""
    return Modality(name="motion", type_id=TYPE_ID,
                    vocab_size=codec.vocab_size, tokenizer=codec)
