"""token_bytes.pt — the bpb denominator — must count a token's REAL UTF-8 bytes.

The table shipped 2026-07-10 was derived with `len(tok.decode([i]).encode("utf-8"))`.
Decoding one token that holds a partial UTF-8 sequence (a byte-fallback id, or a
merge that ends mid-character) yields U+FFFD, whose encoding is 3 bytes — so 216 ids
of the 32768 vocab were overstated. This biased bpb low by a corpus-dependent amount
(about 0.08% on one FineWeb sample; core queue #12).
`build_token_bytes` uses tiktoken's `decode_single_token_bytes`, which is right; these
tests pin that and keep the wrong recipe from coming back.

The first test trains a tiny BPE on ASCII only, so Chinese and emoji fall back to
byte tokens: exactly the case the old recipe miscounts. The second demonstrates the
old recipe's overcount; the third checks the artifact the text line actually loads
(skipped where it is absent).
"""

import os
from pathlib import Path

import pytest
import torch

from core.utils import get_base_dir
from modalities.control import CONTROL_TOKENS, display_form
from modalities.text.tokenizer import RustBPETokenizer, get_token_bytes
from modalities.text.train_tokenizer import build_token_bytes

SAMPLES = [
    "plain ascii text, nothing special.",
    "中文三字节一字：位置、词表、分母。",
    "emoji are four bytes: 🙂🚀 and a flag 🇨🇳",
    "mixed: café résumé naïve — 日本語 — 😀",
]


def _tiny_tokenizer():
    ascii_corpus = ["the quick brown fox jumps over the lazy dog. " * 20,
                    "pack my box with five dozen liquor jugs! " * 20]
    specials = [display_form(n) for n in CONTROL_TOKENS]
    return RustBPETokenizer.train_from_iterator(iter(ascii_corpus), 320, specials)


def _old_recipe(tok):
    """The table the 2026-07 artifact was built with. Kept only to show it is wrong."""
    tb = torch.zeros(tok.get_vocab_size(), dtype=torch.int64)
    for i in range(tok.control_token_offset):
        tb[i] = len(tok.decode([i]).encode("utf-8"))
    return tb


def test_content_bytes_sum_to_the_utf8_length_and_controls_are_zero():
    tok = _tiny_tokenizer()
    tb = build_token_bytes(tok)
    assert tb.dtype == torch.int64 and tb.shape == (tok.get_vocab_size(),)
    for text in SAMPLES:
        ids = tok.encode(text)
        assert int(tb[ids].sum()) == len(text.encode("utf-8")), text
    assert (tb[tok.control_token_offset:] == 0).all()
    assert (tb[:tok.control_token_offset] >= 1).all()      # every content id is ≥ 1 byte


def test_the_decode_reencode_recipe_overstates_partial_utf8_tokens():
    tok = _tiny_tokenizer()
    good, old = build_token_bytes(tok), _old_recipe(tok)
    assert (old >= good).all()                              # only ever inflates
    assert (old > good).any()                               # and does inflate here
    non_ascii = "中文😀"
    ids = tok.encode(non_ascii)
    assert int(old[ids].sum()) > len(non_ascii.encode("utf-8"))   # the wrong denominator


ARTIFACT = Path(os.environ.get("NANOINFRA_TOKENIZER_DIR") or os.path.join(get_base_dir(), "tokenizer"))


@pytest.mark.skipif(not (ARTIFACT / "tokenizer.pkl").exists() or not (ARTIFACT / "token_bytes.pt").exists(),
                    reason="no tokenizer artifact on this box")
def test_the_shipped_table_matches_the_builder(monkeypatch):
    tok = RustBPETokenizer.from_directory(str(ARTIFACT))
    expected = build_token_bytes(tok)
    monkeypatch.setenv("NANOINFRA_TOKENIZER_DIR", str(ARTIFACT))   # the explicit directory
    shipped = get_token_bytes()
    assert shipped.dtype == torch.int64 and shipped.shape == (tok.get_vocab_size(),)
    assert torch.equal(shipped, expected)
    assert (shipped[128:256] == 1).all()                    # byte-fallback ids are one byte
    assert (shipped[tok.control_token_offset:] == 0).all()


# ---- fail closed (queue #16, public issue #6) ------------------------------------------

def test_missing_table_raises_instead_of_reporting_bits_per_token(tmp_path, monkeypatch):
    """No token_bytes.pt -> error. The old ones-table fallback kept the `bpb` name on
    a bits-per-TOKEN number (the 5.5-vs-1.24 episode in exemplars/text_pretrain)."""
    _tiny_tokenizer().save(str(tmp_path))
    monkeypatch.setenv("NANOINFRA_TOKENIZER_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="token_bytes.pt"):
        get_token_bytes()


def test_table_length_must_match_the_vocabulary(tmp_path, monkeypatch):
    tok = _tiny_tokenizer()
    tok.save(str(tmp_path))
    torch.save(build_token_bytes(tok)[:-1], tmp_path / "token_bytes.pt")   # one entry short
    monkeypatch.setenv("NANOINFRA_TOKENIZER_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="vocab"):
        get_token_bytes()
    torch.save(build_token_bytes(tok), tmp_path / "token_bytes.pt")        # the right one loads
    assert get_token_bytes().shape == (tok.get_vocab_size(),)
