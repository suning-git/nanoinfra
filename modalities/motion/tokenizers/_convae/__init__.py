"""
convae — the conv-autoencoder tokenizer FAMILY (architecture machinery).

Fixed backbone (nets), quantizer bottlenecks (quantizers), the two assemblies
(models), and the self-describing artifact loader (codec). A tokenizer folder's
thin load()/train() calls into here; a normal user never imports convae directly.

A different architecture (transformer / diffusion codec) is a SIBLING family
folder under tokenizers/, not an edit here — this is not "common", it is one
family among future families.
"""

from modalities.motion.tokenizers._convae.codec import MotionCodec, save_checkpoint
from modalities.motion.tokenizers._convae.models import MotionFSQ2, MotionVQVAE
from modalities.motion.tokenizers._convae.nets import Decoder, Encoder, ResConv1d
from modalities.motion.tokenizers._convae.quantizers import (
    FSQ_LEVELS,
    FSQ2,
    VectorQuantizerEMA,
)

__all__ = [
    "MotionCodec", "save_checkpoint",
    "MotionFSQ2", "MotionVQVAE",
    "Encoder", "Decoder", "ResConv1d",
    "FSQ2", "VectorQuantizerEMA", "FSQ_LEVELS",
]
