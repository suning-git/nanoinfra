"""The text source reads the share of the row groups it is told; told nothing, the launcher's.

The FineWeb loader reads row groups rank, rank + world_size, ... of each file. By default that is the launcher's
process (core.utils.process_identity), what training wants. A validation set every rank freezes in full is read as
rank 0 of 1: nano_multimodal's resolve_sources gives every source a rank and a world size, the validation sources
0 and 1. Pinned with a launcher environment of rank 1 of 2 and no process group, through the real parquet path with
a stand-in tokenizer whose every document is one token naming its row group.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import modalities.control
import modalities.text
import modalities.text.tokenizer
from modalities.assembler import build_layout
from modalities.control import make_control_resolver
from modalities.text.data_source import TextDataSource
from modalities.text.fineweb import token_data_loader

N_GROUPS = 6


class _Tokenizer:
    def get_bos_token_id(self):
        return 0

    def get_vocab_size(self):
        return 64

    def get_special_tokens(self):
        return ["<bos>"]

    def encode(self, texts, prepend, num_threads):
        return [[prepend, 1 + int(t)] for t in texts]          # the document "g" -> [bos, 1 + g]


@pytest.fixture
def shard(tmp_path):
    """Six row groups of two documents each; the documents of row group g read "g"."""
    path = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": [str(i // 2) for i in range(2 * N_GROUPS)]}), path, row_group_size=2)
    return str(path)


@pytest.fixture
def rank_1_of_2(monkeypatch):
    for name, value in (("RANK", 1), ("LOCAL_RANK", 1), ("WORLD_SIZE", 2)):
        monkeypatch.setenv(name, str(value))


def _groups(tokens):
    """[bos, 1 + g, bos, 1 + g', ...] -> [g, g', ...]"""
    assert tokens[0::2] == [0] * (len(tokens) // 2)
    return [t - 1 for t in tokens[1::2]]


def _loader_groups(path, n_docs, **share):
    batch = next(token_data_loader(B=1, T=2 * n_docs - 1, files=[path], device="cpu", tokenizer=_Tokenizer(),
                                   tokenizer_batch_size=1, **share))
    return _groups(batch["idx"][0].tolist() + batch["targets"][0, -1:].tolist())


def test_the_fineweb_loader_reads_the_share_it_is_given(shard, rank_1_of_2):
    assert _loader_groups(shard, 6) == [1, 1, 3, 3, 5, 5]                          # the launcher's: rank 1 of 2
    assert _loader_groups(shard, 12, rank=0, world_size=1) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert _loader_groups(shard, 6, rank=0, world_size=2) == [0, 0, 2, 2, 4, 4]
    assert _loader_groups(shard, 6, rank=1, world_size=2) == [1, 1, 3, 3, 5, 5]


def _source(path, **share):
    tok = _Tokenizer()
    bands = [modalities.text.manifest(tok), modalities.control.manifest()]
    layout = build_layout(bands)
    config = {"recipe": {"template": ["bos", "text_start", "text_tokens", "text_end", "eos"]}, "recipe_name": "text",
              "files": [path], "sample_len": 4 + 6, "buffer_batch_size": 1, "tokenizer_batch_size": 1, **share}
    return TextDataSource(config, {"text": tok, "layout": layout, "control_resolver": make_control_resolver(bands[1],
                                                                                                          layout)})


def _source_groups(src, n_samples):
    it = iter(src)
    tokens = []
    for _ in range(n_samples):
        tokens += next(it)["tokens"][src._text_start:src._text_end].tolist()      # 6 text tokens: 3 documents
    return _groups(tokens)


def test_the_text_source_passes_its_configured_share_down(shard, rank_1_of_2, monkeypatch):
    monkeypatch.setattr(modalities.text.tokenizer, "get_tokenizer", _Tokenizer)    # the loader encodes with this
    assert _source_groups(_source(shard, rank=0, world_size=1), 4) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert _source_groups(_source(shard), 2) == [1, 1, 3, 3, 5, 5]                   # told nothing: the launcher's
