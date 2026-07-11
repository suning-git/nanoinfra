"""
text — the text content modality.

Implementation lives here (moved out of core): the BPE codec wrappers
(tokenizer.py), the FineWeb streaming loader (fineweb.py), TextDataSource +
the conventional text recipes (data_source.py), and TextEvaluator
(evaluator.py). Core keeps only mechanisms.
"""

from modalities.assembler import Modality

from modalities.text.tokenizer import (
    ByteTokenizer,
    HuggingFaceTokenizer,
    RustBPETokenizer,
    get_token_bytes,
    get_tokenizer,
)
from modalities.text.fineweb import token_data_loader, list_parquet_files
from modalities.text.data_source import TEXT_RECIPE, TextDataSource, create_recipe
from modalities.text.evaluator import TextEvaluator

TYPE_ID = 0  # canonical (0=text, 1=motion, 2=control)

# The modality's data-source fragment: orchestrators compose these into
# MixedDataLoader's source_types.
SOURCE_TYPES = {
    'text': TextDataSource,
}


def manifest(tokenizer) -> Modality:
    """The trained artifact physically bundles the control band at its tail
    (BPE build-time reservation, historical residency). Text's OWN band is
    what remains — the artifact self-describes how many specials it carries,
    so no registry import is needed here."""
    content_size = tokenizer.get_vocab_size() - len(tokenizer.get_special_tokens())
    return Modality(name="text", type_id=TYPE_ID,
                    vocab_size=content_size, tokenizer=tokenizer)
