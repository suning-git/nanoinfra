"""The modality sources' states read back from JSON unchanged (see core.data.DataSource.get_state).

The text source's state is read through the real parquet path with a stand-in tokenizer: TextDataSource's exact
position (text_standins.py), and the raw stream's of token_data_loader. The motion and T2M sources are given only
what get_state reads (their sampler and item count), so no data is needed.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from modalities.motion.data.sources import MotionDataSource, T2MDataSource, _index_sampler
from modalities.tests import text_standins
from modalities.text.fineweb import token_data_loader


def _reads_back(state):
    assert json.loads(json.dumps(state)) == state


class _Tokenizer:
    def get_bos_token_id(self):
        return 0

    def get_vocab_size(self):
        return 64

    def get_special_tokens(self):
        return ["<bos>"]

    def encode(self, texts, prepend, num_threads):
        return [[prepend] + [1 + ord(c) % 60 for c in t] for t in texts]


def _shard(tmp_path):
    path = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": [f"{i} document" for i in range(6)]}), path, row_group_size=2)
    return str(path)


def _loader(path, state=None):
    return token_data_loader(B=1, T=8, files=[path], device="cpu", resume_state_dict=state, tokenizer=_Tokenizer(),
                             tokenizer_batch_size=1)


def test_the_text_source_state_reads_back(tmp_path):
    it = _loader(_shard(tmp_path))
    for _ in range(4):                                     # into a later row group
        batch = next(it)
    _reads_back(batch["state_dict"])


def test_the_text_source_starts_from_its_starting_state(tmp_path):
    """Before its first window TextDataSource reports where it will start (the loader records that as its starting
    point); restored from it, reading begins at the first document."""
    files = text_standins.write_files(tmp_path, [[[3, 4]] * 3])
    start = text_standins.source(files, 4, rank=0, world_size=1).get_state()
    _reads_back(start)
    restored = text_standins.source(files, 4, rank=0, world_size=1)
    restored.set_state(start)
    first = next(iter(restored))
    assert first["meta"]["from"][0][:5] == [1, 0, 0, 0, 0]
    assert text_standins.text_of(first, 4) == text_standins.text_of(
        next(iter(text_standins.source(files, 4, rank=0, world_size=1))), 4)


def test_the_text_sources_exact_state_reads_back(tmp_path):
    files = text_standins.write_files(tmp_path, [[[3, 4]] * 3])
    src = text_standins.source(files, 4, rank=1, world_size=2)
    it = iter(src)
    for _ in range(3):
        next(it)
    _reads_back(src.get_state())
    _reads_back(src.merge_rank_states([text_standins.source(files, 4, rank=0, world_size=2).get_state(),
                                       src.get_state()]))


def _drawn(sampler, n=5):
    it = iter(sampler)
    for _ in range(n):
        next(it)
    return sampler


def test_the_motion_source_state_reads_back():
    src = object.__new__(MotionDataSource)
    src._global = np.zeros((8, 4), dtype=np.int64)
    src.sampler = _drawn(_index_sampler(len(src._global), seed=0, rank=0, world_size=1))
    _reads_back(src.get_state())


def test_the_t2m_source_state_reads_back():
    src = object.__new__(T2MDataSource)
    src._tokens = np.zeros((8, 4), dtype=np.int64)
    src.sampler = _drawn(_index_sampler(len(src._tokens), seed=0, rank=0, world_size=1))
    _reads_back(src.get_state())


def test_the_motion_sources_restore_their_share_and_merge_with_their_count():
    """_restore hands the sampler the whole state (its leftover too, split by rank); the merge keeps the item count."""
    src = object.__new__(T2MDataSource)
    src._tokens = np.zeros((8, 4), dtype=np.int64)
    src.sampler = _index_sampler(8, seed=0, rank=1, world_size=2)
    src.set_state({"seed": 0, "epoch": 0, "index": 4, "leftover": [[0, 5], [0, 6], [0, 7]], "n_pairs": 8})
    mine = src.get_state()
    assert mine["leftover"] == [[0, 6]] and mine["n_pairs"] == 8 and mine["shard"] == [1, 2]
    other = object.__new__(T2MDataSource)
    other._tokens, other.sampler = src._tokens, _index_sampler(8, seed=0, rank=0, world_size=2)
    other.set_state({"seed": 0, "epoch": 0, "index": 4, "leftover": [[0, 5], [0, 6], [0, 7]], "n_pairs": 8})
    run = src.merge_rank_states([other.get_state(), mine])
    assert run["n_pairs"] == 8 and "shard" not in run
    assert sorted(map(tuple, run["leftover"])) == [(0, 5), (0, 6), (0, 7)]
    _reads_back(run)
