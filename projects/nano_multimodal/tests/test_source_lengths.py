"""The sources' length keys since 2026-10: `sample_len` (text, motion, video rows) and `max_sample_len` (t2m) count
raw tokens per sample, T + 1. The old `sequence_len` is refused, and the refusal names the key that source takes.

    pytest projects/nano_multimodal/tests/test_source_lengths.py
"""
import pytest

from projects.nano_multimodal import assembly
from projects.nano_multimodal.sources import VideoRowSource


def test_video_row_source_refuses_the_old_key_when_built_directly():
    with pytest.raises(KeyError, match="`sequence_len` is now `sample_len`"):
        VideoRowSource({"sample_len": 1300, "sequence_len": 999, "cache": "/nonexistent"}, {})


@pytest.mark.parametrize("kind, key", [("t2m", "max_sample_len"), ("motion", "sample_len"), ("text", "sample_len"),
                                       ("video", "sample_len")])
def test_a_declared_sequence_len_is_refused_naming_the_key_that_source_takes(kind, key):
    config = {"data": {"recipes": {"r": {}}, "sources": [{"type": kind, "recipe": "r", "sequence_len": 128}]}}
    with pytest.raises(KeyError, match=f"it takes `{key}`"):
        assembly.resolve_sources(config, vocab=None, split="train", device="cpu")
