"""TextDataSource's exact resume: positions down to the document and the token, merged over the ranks.

Small parquet files and a stand-in tokenizer (text_standins.py): every token names its file, row group, document and
place, so a window's text, and the pieces its meta says it came from, can be checked against each other and counted
by position. N ranks are N sources (or loaders) in one process, told their rank in the config.

    pytest modalities/tests/test_text_exact_resume.py
"""

import collections
import json
import os
import random
import signal
from functools import partial

import pytest

import torch

from core.data.batchers import PackingBatcher, PaddingBatcher
from core.data.data_source import DataSource
from core.data.dist_sampler import ResumableDistributedSampler
from core.data.mixed_dataloader import MixedDataLoader
from core.data.supervision import build_NTP_supervised_sample
from core.tokenization.vocab_layout import VocabLayout
from modalities.tests import text_standins as ts
from modalities.text.fineweb import RowGroups, merge_exact_states, token_data_loader


def _windows(src, k, T):
    it = iter(src)
    out = []
    for _ in range(k):
        s = next(it)
        out.append((ts.text_of(s, T), s["meta"]["from"], src.get_state()))
    return out


def _tokens_of(pieces, lengths):
    out = []
    for e, f, g, d, tok, n in pieces:
        out += ts.doc_tokens(lengths, f, g, d)[tok:tok + n]
    return out


# ------------------------------------------------------------------ a fresh reading is the old one

@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_a_fresh_reading_is_token_data_loaders(tmp_path, monkeypatch, world_size):
    """From the start, rank r of N reads what the old reader read: row groups r mod N of every file, the same
    consecutive windows of T tokens (the old loader's [B, T] rows)."""
    rng = random.Random(world_size)
    lengths = [[[rng.randint(0, 9) for _ in range(rng.randint(1, 4))] for _ in range(rng.randint(3, 6))]
               for _ in range(3)]
    files, T = ts.write_files(tmp_path, lengths), 5
    for r in range(world_size):
        for name, value in (("RANK", r), ("LOCAL_RANK", r), ("WORLD_SIZE", world_size)):
            monkeypatch.setenv(name, str(value))
        old = token_data_loader(B=2, T=T, files=files, device="cpu", tokenizer=ts.Tokenizer(),
                                tokenizer_batch_size=2)
        want = []
        while len(want) < 40:
            want += next(old)["idx"].tolist()
        got = _windows(ts.source(files, T, rank=r, world_size=world_size), 40, T)
        assert [w for w, _, _ in got] == want[:40]
        assert all(w == _tokens_of(p, lengths) for w, p, _ in got)          # the meta says where it came from


# ------------------------------------------------------------------ the state follows the cursor

def test_the_state_speaks_of_the_cursor_not_of_what_was_read_ahead(tmp_path):
    """Row group 0 has one document of 4 tokens + BOS, row group 1 three such (read in one batch): after the first
    window (4 tokens) the cursor is in row group 0, row group 1 not yet entered; after the second it is in row group
    1's first document, whose other two documents were read but are not trained."""
    lengths = [[[4], [4, 4, 4], [4]]]
    files = ts.write_files(tmp_path, lengths)
    src = ts.source(files, 4, rank=0, world_size=1, tokenizer_batch_size=3)
    (w1, p1, s1), (w2, p2, s2) = _windows(src, 2, 4)
    assert p1 == [[1, 0, 0, 0, 0, 4]] and s1["taken"] == 1 and s1["leftover"] == [[1, 0, 0, 0, 4]]
    assert p2 == [[1, 0, 0, 0, 4, 1], [1, 0, 1, 0, 0, 3]]
    assert s2["taken"] == 2 and s2["leftover"] == [[1, 0, 1, 0, 3]]


def test_a_leftover_segment_read_ahead_stays_in_the_state_until_the_cursor_enters_it(tmp_path):
    lengths = [[[2], [9], [9], [9]]]
    files = ts.write_files(tmp_path, lengths)
    src = ts.source(files, 4, rank=0, world_size=1)
    src.set_state({"files": [os.path.basename(f) for f in files], "next": [1, 0, 3],
                   "leftover": [[1, 0, 0, 0, 0], [1, 0, 1, 0, 6]]})
    (w1, p1, s1), (w2, p2, s2) = _windows(src, 2, 4)
    # window 1: row group 0's 3 tokens and the first of row group 1's leftover (from token 6): the cursor is there
    assert p1 == [[1, 0, 0, 0, 0, 3], [1, 0, 1, 0, 6, 1]]
    assert s1["taken"] == 0 and s1["leftover"] == [[1, 0, 1, 0, 7]]
    assert p2 == [[1, 0, 1, 0, 7, 3], [1, 0, 3, 0, 0, 1]]      # the rest of that segment, then the main stream
    assert s2["taken"] == 1 and s2["leftover"] == [[1, 0, 3, 0, 1]]


@pytest.mark.parametrize("where", ["document", "row group", "file", "epoch"])
def test_a_window_ending_on_a_boundary_puts_the_cursor_on_the_next_start(tmp_path, where):
    """3 + 1 tokens per document, windows of 4: every window ends where a document does."""
    lengths = [[[3, 3], [3]], [[3]]]
    files = ts.write_files(tmp_path, lengths)
    src = ts.source(files, 4, rank=0, world_size=1)
    k = {"document": 1, "row group": 2, "file": 3, "epoch": 4}[where]
    state = _windows(src, k, 4)[-1][2]
    expect = {"document": (1, [[1, 0, 0, 1, 0]]), "row group": (1, []), "file": (2, []), "epoch": (3, [])}[where]
    assert (state["taken"], state["leftover"]) == expect
    # resumed from there (one rank), the next window is the next document's
    nxt = _windows(src, 1, 4)[0][1][0][:5]
    resumed = ts.source(files, 4, rank=0, world_size=1)
    resumed.set_state(merge_exact_states([state], RowGroups(files)))
    assert _windows(resumed, 1, 4)[0][1][0][:5] == nxt


# ------------------------------------------------------------------ the merge

def test_the_new_front_follows_the_furthest_main_stream_row_group(tmp_path):
    """Two ranks from the start: rank 0 entered its row groups 0, 2; rank 1 its 1, 3, 5, 7. The front moves past 7;
    rank 0's 4 and 6 are whole leftover; each rank's cursor segment is kept."""
    files = ts.write_files(tmp_path, [[[1]] * 12])
    rg = RowGroups(files)
    base = {"files": ["shard_00000.parquet"], "next": [1, 0, 0]}
    s0 = {**base, "taken": 2, "shard": [0, 2], "leftover": [[1, 0, 2, 0, 1]]}
    s1 = {**base, "taken": 4, "shard": [1, 2], "leftover": [[1, 0, 7, 0, 1]]}
    run = merge_exact_states([s1, s0], rg)                    # any order
    assert run["next"] == [1, 0, 8]
    assert run["leftover"] == [[1, 0, 2, 0, 1], [1, 0, 4, 0, 0], [1, 0, 6, 0, 0], [1, 0, 7, 0, 1]]


def test_no_rank_in_the_main_stream_keeps_the_front(tmp_path):
    files = ts.write_files(tmp_path, [[[1]] * 12])
    base = {"files": ["shard_00000.parquet"], "next": [1, 0, 10]}
    states = [{**base, "taken": 0, "shard": [0, 2], "leftover": [[1, 0, 2, 0, 0]]},
              {**base, "taken": 0, "shard": [1, 2], "leftover": [[1, 0, 4, 0, 1]]}]
    run = merge_exact_states(states, RowGroups(files))
    assert run["next"] == [1, 0, 10] and run["leftover"] == [[1, 0, 2, 0, 0], [1, 0, 4, 0, 1]]


def test_the_front_moves_on_across_files_and_epochs(tmp_path):
    files = ts.write_files(tmp_path, [[[1]] * 3, [[1]] * 2])
    rg = RowGroups(files)
    assert rg.successor((1, 0, 2)) == (1, 1, 0) and rg.successor((1, 1, 1)) == (2, 0, 0)
    # one rank, entered every row group of epoch 1 and the first of epoch 2
    state = {"files": [], "next": [1, 0, 0], "taken": 6, "shard": [0, 1], "leftover": []}
    assert merge_exact_states([state], rg)["next"] == [2, 0, 1]


def test_merge_needs_every_rank_of_one_run(tmp_path):
    files = ts.write_files(tmp_path, [[[1]] * 4])
    one = {"files": [], "next": [1, 0, 0], "taken": 1, "shard": [0, 2], "leftover": []}
    with pytest.raises(ValueError, match="every rank"):
        merge_exact_states([one], RowGroups(files))
    other = {**one, "shard": [1, 2], "next": [1, 0, 1]}
    with pytest.raises(ValueError, match="different fronts"):
        merge_exact_states([one, other], RowGroups(files))


# ------------------------------------------------------------------ restoring

def test_a_state_from_before_the_change(tmp_path):
    files = ts.write_files(tmp_path, [[[3]] * 3, [[3]] * 2])
    names = [os.path.basename(f) for f in files]
    src = ts.source(files, 4, rank=0, world_size=1)
    src.set_state({"files": names, "pq_idx": 0, "rg_idx": None, "epoch": 1})     # not yet begun: the start
    assert src.get_state()["next"] == [1, 0, 0]
    src.set_state({"files": names, "pq_idx": 0, "rg_idx": 2, "epoch": 1})        # in the file's last row group
    assert src.get_state()["next"] == [1, 1, 0] and src.get_state()["leftover"] == []
    assert _windows(src, 1, 4)[0][1][0][:3] == [1, 1, 0]


def test_one_ranks_state_is_taken_only_when_complete_by_itself(tmp_path):
    files = ts.write_files(tmp_path, [[[3]] * 6])
    src = ts.source(files, 4, rank=0, world_size=2)
    state = _windows(src, 1, 4)[0][2]
    assert state["shard"] == [0, 2] and state["taken"] == 1
    with pytest.raises(ValueError, match="only that rank's part"):
        ts.source(files, 4, rank=0, world_size=1).set_state(state)
    ts.source(files, 4).set_state({**state, "taken": 0, "leftover": []})          # nothing of its own: accepted
    alone = ts.source(files, 4, rank=0, world_size=1)
    one_rank = _windows(alone, 1, 4)[0][2]
    again = ts.source(files, 4, rank=0, world_size=1)
    again.set_state(one_rank)                                                      # a one-rank run's: complete
    assert _windows(again, 1, 4)[0][0] == _windows(alone, 1, 4)[0][0]


def test_more_ranks_than_row_groups_raises_instead_of_hanging(tmp_path):
    files = ts.write_files(tmp_path, [[[3]]])
    src = ts.source(files, 4, rank=1, world_size=2)

    def expired(signum, frame):
        raise TimeoutError("the reading loops forever")
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(20)
    try:
        with pytest.raises(ValueError, match="no row group"):
            next(iter(src))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_the_state_reads_back_from_json(tmp_path):
    files = ts.write_files(tmp_path, [[[5, 2]] * 4])
    src = ts.source(files, 4, rank=1, world_size=2)
    for _, _, state in _windows(src, 3, 4):
        assert json.loads(json.dumps(state)) == state


# ------------------------------------------------------------------ saves and resumes on changing rank counts

def _loader(files, T, rank, world_size, B=2):
    pads = {"idx": ts.CONTROL + 3, "targets": VocabLayout.IGNORE_INDEX, "token_types": 2, "target_types": 2}
    return MixedDataLoader({"text": (ts.source(files, T, rank=rank, world_size=world_size), 1.0)},
                           build_supervised_sample_fn=build_NTP_supervised_sample,
                           create_batcher_fn=partial(PaddingBatcher, shape=(B, T + 3), pad_values=pads))


def _texts(batch, T):
    """Each sample's text tokens out of a collated batch (doc_id k marks sample k; its idx is the prepared sample
    [bos, text_start, text..., text_end], the next-token shift having dropped eos)."""
    doc, idx = batch["doc_id"], batch["idx"]
    return [idx[doc == k].tolist()[2:2 + T] for k in range(len(batch["sample_meta"]))]


def _round(files, T, world_size, run, batches, trained, make=None, lengths=None):
    """One launch from `run`; every delivered window's text is checked against the documents its pieces name (by the
    stand-in tokenizer, independently of the reader), when `lengths` is given."""
    loaders = [(make or partial(_loader, files, T))(r, world_size) for r in range(world_size)]
    if run is not None:
        for loader in loaders:
            loader.set_state(json.loads(json.dumps(run)))
    snaps = []
    for r, loader in enumerate(loaders):
        it, snap = iter(loader), loader.get_state()
        for _ in range(batches[r]):
            b = next(it)
            for meta, text in zip(b["sample_meta"], _texts(b, T)):
                if "from" not in meta:
                    continue
                if lengths is not None:
                    assert text == _tokens_of(meta["from"], lengths), "a window's text is not what it says"
                for e, f, g, d, tok, n in meta["from"]:
                    trained += [(e, f, g, d, i) for i in range(tok, tok + n)]
            snap = b["state_dict"]
        snaps.append(json.loads(json.dumps(snap)))
    return loaders[0].merge_rank_snapshots(snaps)


def _check(lengths, run, trained):
    """No token position trained twice; every position in a row group before the front is trained or in a leftover
    segment, once; none at or after the front is trained."""
    st = run["source_states"]["text"]
    front = tuple(st["next"])
    assert len(set(trained)) == len(trained), "a token was trained twice"
    counts = collections.Counter()
    for e, f, g, d, tok in st["leftover"]:
        for dd in range(d, len(lengths[f][g])):
            n = len(ts.doc_tokens(lengths, f, g, dd))
            counts.update((e, f, g, dd, i) for i in range(tok if dd == d else 0, n))
    assert all(c == 1 for c in counts.values()), "leftover segments overlap"
    left = set(counts)
    assert not left & set(trained), "a trained token is in a leftover segment"
    before = set()
    for e in range(1, front[0] + 1):
        for f, groups in enumerate(lengths):
            for g, docs in enumerate(groups):
                if (e, f, g) < front:
                    for d in range(len(docs)):
                        before |= {(e, f, g, d, i) for i in range(len(ts.doc_tokens(lengths, f, g, d)))}
    assert set(trained) | left == before, (sorted(before - set(trained) - left)[:5],
                                           sorted((set(trained) | left) - before)[:5])


@pytest.mark.parametrize("seed", range(5))
def test_random_saves_and_resumes_on_changing_rank_counts(tmp_path, seed):
    rng = random.Random(seed)
    lengths = [[[rng.randint(0, 12) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(4, 7))]
               for _ in range(rng.randint(1, 3))]
    files, T = ts.write_files(tmp_path, lengths), rng.choice([3, 5, 8])
    trained, run = [], None
    for _ in range(7):
        world_size = rng.choice([1, 2, 3])
        steps = rng.randint(0 if run is not None else 1, 5)
        per = [steps + (rng.randint(0, 2) if rng.random() < 0.3 else 0) for _ in range(world_size)]
        run = _round(files, T, world_size, run, per, trained, lengths=lengths)
        _check(lengths, run, trained)
    assert trained


def test_a_source_iterated_again_without_a_restore_reads_on_from_its_cursor(tmp_path):
    """iter() again on the same sources (no set_state) continues each rank's reading from its own state (entered
    row groups and the cursor's segment), and the merge afterwards still keeps the invariant."""
    rng = random.Random(7)
    lengths = [[[rng.randint(0, 9) for _ in range(rng.randint(1, 3))] for _ in range(8)]]
    files, T = ts.write_files(tmp_path, lengths), 3
    trained, states = [], []
    for r in range(2):
        src = ts.source(files, T, rank=r, world_size=2)
        for k in (5 + r, 4):                                    # two iterations, one after the other
            for _, pieces, _ in _windows(src, k, T):
                for e, f, g, d, tok, n in pieces:
                    trained += [(e, f, g, d, i) for i in range(tok, tok + n)]
        states.append(src.get_state())
    assert any(st["taken"] > 1 for st in states)
    _check(lengths, {"source_states": {"text": merge_exact_states(states, RowGroups(files))}}, trained)


def test_iterated_again_inside_its_row_group_a_rank_keeps_what_it_entered(tmp_path):
    """The cursor in row group 1 (entered 2) when the first iteration stops; the second iteration starts inside that
    row group's rest, and its first window's state still says 2 row groups entered."""
    files = ts.write_files(tmp_path, [[[4], [19], [4]]])
    src = ts.source(files, 3, rank=0, world_size=1)
    before = _windows(src, 3, 3)[-1][2]                     # 5 tokens of rg 0, then 4 of rg 1
    assert before["taken"] == 2 and before["leftover"] == [[1, 0, 1, 0, 4]]
    after = _windows(src, 1, 3)[0][2]
    assert after["taken"] == 2 and after["leftover"] == [[1, 0, 1, 0, 7]]


class _Rows(DataSource):
    """A sampler-backed source mixed with the text: rows of 2..6 tokens, metas without "from"."""

    def __init__(self, n, rank, world_size):
        self.sampler = ResumableDistributedSampler(_Count(n), seed=3)
        self.sampler.rank, self.sampler.world_size = rank, world_size

    def __iter__(self):
        for j in self.sampler:
            yield {"tokens": torch.full((2 + j % 5,), j + 1),
                   "token_types": torch.zeros(2 + j % 5, dtype=torch.long), "meta": {"row": j}}

    def get_state(self):
        return self.sampler.state_dict()

    def set_state(self, state):
        self.sampler.load_state_dict(state)

    def merge_rank_states(self, states):
        return self.sampler.merge_rank_states(states)


class _Count:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


@pytest.mark.parametrize("seed", range(3))
def test_text_mixed_with_another_source_in_one_loader(tmp_path, seed):
    """PackingBatcher rows fit a text window and rows of the other source, so a window is often held back pending."""
    rng = random.Random(20 + seed)
    lengths = [[[rng.randint(0, 9) for _ in range(rng.randint(1, 3))] for _ in range(6)] for _ in range(2)]
    files, T = ts.write_files(tmp_path, lengths), 4
    pads = {"idx": ts.CONTROL + 3, "targets": VocabLayout.IGNORE_INDEX, "token_types": 2, "target_types": 2}

    def make(rank, world_size):
        return MixedDataLoader({"text": (ts.source(files, T, rank=rank, world_size=world_size), 1.0),
                                "rows": (_Rows(11, rank, world_size), 1.0)},
                               build_supervised_sample_fn=build_NTP_supervised_sample,
                               create_batcher_fn=partial(PackingBatcher, shape=(2, 2 * (T + 3)), pad_values=pads))
    trained, run = [], None
    for _ in range(6):
        world_size = rng.choice([1, 2, 3])
        steps = rng.randint(0 if run is not None else 1, 4)
        run = _round(files, T, world_size, run, [steps] * world_size, trained, make=make, lengths=lengths)
        _check(lengths, run, trained)
