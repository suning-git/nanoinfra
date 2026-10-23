"""Exact resume for a sampler-backed source: saved on N ranks, resumed on M, nothing skipped or repeated.

Every rank's MixedDataLoader snapshot is merged into the run's (MixedDataLoader.merge_rank_snapshots, the source's
merge_rank_states, ResumableDistributedSampler.merge_rank_states): the furthest rank's position, and the items before
it no rank has trained (leftover). Here N ranks are N loaders in one process, their sources told their rank; samples
of random length go through PackingBatcher, so the ranks consume different numbers of them per batch. The invariant,
checked after every save: the run state's leftover is exactly the untrained items before its position, and no item
at or after the position was trained. Items are (epoch, dataset index), never values.
"""

import json
import random
from functools import partial

import pytest
import torch

from core.data.batchers import PackingBatcher, PaddingBatcher
from core.data.data_source import DataSource
from core.data.dist_sampler import ResumableDistributedSampler
from core.data.mixed_dataloader import MixedDataLoader
from core.data.supervision import build_NTP_supervised_sample
from core.tokenization.vocab_layout import VocabLayout

IGNORE = VocabLayout.IGNORE_INDEX
PADS = {"idx": 0, "targets": IGNORE}


class _Count:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _Items(DataSource):
    """n items of random length 2..max_len raw tokens, drawn through the sampler; each sample's meta names it as
    [source tag, epoch, item], the epoch read off the sampler before the pull (a leftover item carries its own)."""

    def __init__(self, n, rank, world_size, seed=0, max_len=9, tag="items"):
        self.n, self.tag = n, tag
        self.lengths = [random.Random(seed * 7919 + j).randint(2, max_len) for j in range(n)]
        self.sampler = ResumableDistributedSampler(_Count(n), seed=seed)
        self.sampler.rank, self.sampler.world_size = rank, world_size

    def __iter__(self):
        s, it = self.sampler, iter(self.sampler)
        while True:
            epoch = s.leftover[0][0] if s.leftover else (s.epoch * self.n + s.index + s.rank) // self.n
            j = next(it)
            yield {"tokens": torch.full((self.lengths[j],), j + 1), "meta": {"item": [self.tag, epoch, j]}}

    def get_state(self):
        return self.sampler.state_dict()

    def set_state(self, state):
        self.sampler.load_state_dict(state)

    def merge_rank_states(self, states):
        return self.sampler.merge_rank_states(states)


def _loader(n, rank, world_size, batcher=PackingBatcher, shape=(2, 12), seed=0):
    return MixedDataLoader({"items": (_Items(n, rank, world_size, seed), 1.0)},
                           build_supervised_sample_fn=build_NTP_supervised_sample,
                           create_batcher_fn=partial(batcher, shape=shape, pad_values=PADS))


def _through_json(x):
    return json.loads(json.dumps(x))


def _position(n, seed, epoch, item, perms):
    if epoch not in perms:
        perms[epoch] = ResumableDistributedSampler(_Count(n), seed=seed)._perm(epoch)
    return epoch * n + perms[epoch].index(item)


def _check(n, seed, run_state, delivered, name="items"):
    """The invariant, for source `name`: no item delivered twice; everything before the position is delivered or in
    the leftover, once; nothing at or after it is delivered."""
    perms = {}
    delivered = [(e, j) for tag, e, j in delivered if tag == name]
    st = run_state["source_states"][name]
    front = st["epoch"] * n + st["index"]
    left = [tuple(x) for x in st["leftover"]]
    assert len(set(delivered)) == len(delivered), "an item was trained twice"
    assert len(set(left)) == len(left), "an item is in the leftover twice"
    assert not set(delivered) & set(left), "a trained item is in the leftover"
    covered = {_position(n, seed, e, j, perms) for e, j in delivered + left}
    assert covered == set(range(front)), (sorted(set(range(front)) - covered)[:10],
                                          sorted(covered - set(range(front)))[:10])


def _round(n, world_size, run_state, batches, delivered, seed=0, batcher=PackingBatcher, shape=(2, 12), make=None):
    """One launch on world_size ranks from run_state (None = the start): every rank takes `batches` batches (a list:
    per rank), then the snapshots are merged as checkpoint saving does. Returns the new run state. `make(rank,
    world_size)` builds a rank's loader (default: one source of n items)."""
    make = make or (lambda r, w: _loader(n, r, w, batcher, shape, seed))
    loaders = [make(r, world_size) for r in range(world_size)]
    if run_state is not None:
        for loader in loaders:
            loader.set_state(_through_json(run_state))
    snaps = []
    for r, loader in enumerate(loaders):
        it = iter(loader)
        snap = loader.get_state()
        for _ in range(batches[r]):
            b = next(it)
            delivered += [tuple(m["item"]) for m in b["sample_meta"]]
            snap = b["state_dict"]
        snaps.append(_through_json(snap))
    return loaders[0].merge_rank_snapshots(snaps)


@pytest.mark.parametrize("seed", range(6))
def test_random_saves_and_resumes_on_changing_rank_counts(seed):
    rng = random.Random(seed)
    n = rng.choice([7, 13, 29, 40])
    delivered, run = [], None
    for _ in range(8):
        world_size = rng.choice([1, 2, 3, 4])
        steps = rng.randint(0 if run is not None else 1, 6)
        uneven = [steps + (rng.randint(0, 2) if rng.random() < 0.3 else 0) for _ in range(world_size)]
        run = _round(n, world_size, run, uneven, delivered, seed=seed)
        _check(n, seed, run, delivered)
    assert delivered                                         # something was trained, across epochs for small n


def test_the_example_two_ranks_one_ahead():
    """From the start, rank 0 trains a0 a2 a4 and rank 1 a1 a3 a5 a7 a9 (one batch each, three and five rows): the
    position is rank 1's, 10, and rank 0's a6 and a8 before it are the leftover."""
    n = 40
    rank0, rank1 = _loader(n, 0, 2, PaddingBatcher, (3, 12)), _loader(n, 1, 2, PaddingBatcher, (5, 12))
    snaps = [next(iter(rank0))["state_dict"], next(iter(rank1))["state_dict"]]
    merged = rank0.merge_rank_snapshots(snaps)["source_states"]["items"]
    perm = ResumableDistributedSampler(_Count(n), seed=0)._perm(0)
    assert (merged["epoch"], merged["index"]) == (0, 10)
    assert merged["leftover"] == [[0, perm[6]], [0, perm[8]]]


def test_a_resume_on_four_ranks_from_two_hands_out_the_rest_of_the_epoch():
    """10 items, two ranks delivered positions 0..3: four ranks then hand out 4..9 before any item of epoch 1 (the
    sampler before 2026-10-22 dropped 8 and 9, the remainder of (10 - 4) by 4)."""
    n, delivered = 10, []
    run = _round(n, 2, None, [2, 2], delivered, batcher=PaddingBatcher, shape=(1, 12))
    four = [ResumableDistributedSampler(_Count(n)) for _ in range(4)]
    for r, s in enumerate(four):
        s.rank, s.world_size = r, 4
        s.load_state_dict(run["source_states"]["items"])
    its = [iter(s) for s in four]
    got = [next(it) for _ in range(2) for it in its]
    perm = four[0]._perm(0)
    assert got[:6] == perm[4:10]


def test_saving_again_right_after_a_resume_changes_nothing():
    """Zero batches after a resume, on another rank count, every rank still holding its whole share of the leftover:
    the merge gives back the same position and the same leftover items."""
    n, delivered = 29, []
    run = _round(n, 3, None, [4, 2, 5], delivered)
    st = run["source_states"]["items"]
    assert st["leftover"]
    again = _round(n, 2, run, [0, 0], delivered)["source_states"]["items"]
    assert (again["epoch"], again["index"]) == (st["epoch"], st["index"])
    assert sorted(map(tuple, again["leftover"])) == sorted(map(tuple, st["leftover"]))
    _check(n, 0, {"source_states": {"items": again}}, delivered)


def test_ranks_still_handing_out_the_leftover_do_not_move_the_position():
    n, delivered = 40, []
    run = _round(n, 3, None, [6, 1, 3], delivered)
    st = run["source_states"]["items"]
    left = len(st["leftover"])
    assert left >= 4
    after = _round(n, 4, run, [1, 0, 0, 0], delivered, batcher=PaddingBatcher, shape=(1, 12))
    a = after["source_states"]["items"]
    assert (a["epoch"], a["index"]) == (st["epoch"], st["index"])           # nobody reached the stream
    assert len(a["leftover"]) == left - 1
    _check(n, 0, after, delivered)


def test_one_rank_in_an_old_epochs_leftover_while_another_reads_the_next_epoch():
    n, delivered = 7, []
    run = _round(n, 2, None, [1, 4], delivered)                 # rank 1 runs far ahead: leftover in epoch 0
    _check(n, 0, run, delivered)
    run = _round(n, 2, run, [1, 6], delivered)                  # rank 0 hands out epoch-0 leftover, rank 1 epoch 1+
    _check(n, 0, run, delivered)
    epochs = {e for _, e, _ in delivered}
    assert len(epochs) > 1


def test_the_merge_does_not_touch_the_live_sources():
    n = 20
    loaders = [_loader(n, r, 2) for r in range(2)]
    snaps = []
    for loader in loaders:
        it = iter(loader)
        for _ in range(2):
            snap = next(it)["state_dict"]
        snaps.append(snap)
    live = [loader.sources["items"].get_state() for loader in loaders]
    loaders[0].merge_rank_snapshots(snaps)
    assert [loader.sources["items"].get_state() for loader in loaders] == live


def test_after_a_restore_a_rank_reports_its_share_not_the_whole_leftover():
    """A rank's snapshot entry, before it delivers anything of the source, is the source's own state after the
    restore: its share. Reporting the whole list would repeat the other ranks' items at the next merge."""
    n, delivered = 29, []
    run = _round(n, 2, None, [3, 1], delivered)
    whole = run["source_states"]["items"]["leftover"]
    assert len(whole) >= 2
    loaders = [_loader(n, r, 3) for r in range(3)]
    for r, loader in enumerate(loaders):
        loader.set_state(_through_json(run))
        assert loader.get_state()["source_states"]["items"]["leftover"] == whole[r::3]


def test_merge_needs_every_rank():
    s = ResumableDistributedSampler(_Count(10))
    one = {**s.state_dict(), "shard": [0, 2]}
    with pytest.raises(ValueError, match="every rank"):
        s.merge_rank_states([one])
    with pytest.raises(ValueError, match="every rank"):
        s.merge_rank_states([one, one])


def test_merge_needs_states_of_one_resume():
    """Two ranks of one resume start from one position and advance by N each: positions 0 and 1 of a two-rank run
    cannot both be, and merging them would put gap items past the front."""
    s = ResumableDistributedSampler(_Count(10))
    a = {"seed": 0, "epoch": 0, "index": 0, "leftover": [], "shard": [0, 2]}
    b = {"seed": 0, "epoch": 0, "index": 1, "leftover": [], "shard": [1, 2]}
    with pytest.raises(ValueError, match="multiple"):
        s.merge_rank_states([a, b])


def test_a_ranks_own_state_with_leftover_is_refused_alone():
    """rank 1 of 2 holds [b, d] of the run's [a, b, c, d]; restored alone it would be split again and lose b."""
    s = ResumableDistributedSampler(_Count(10))
    s.rank, s.world_size = 1, 2
    s.load_state_dict({"seed": 0, "epoch": 0, "index": 6, "leftover": [[0, 1], [0, 2], [0, 3], [0, 4]]})
    mine = s.state_dict()
    assert mine["leftover"] == [[0, 2], [0, 4]] and mine["shard"] == [1, 2]
    fresh = ResumableDistributedSampler(_Count(10))
    with pytest.raises(ValueError, match="share"):
        fresh.load_state_dict(mine)
    fresh.load_state_dict({**mine, "leftover": []})          # nothing left: the old rank-0 resume, accepted
    one_rank = {**mine, "shard": [0, 1]}
    fresh.load_state_dict(one_rank)                          # a one-rank run's state is complete by itself
    assert list(fresh.leftover) == [[0, 2], [0, 4]]


def test_a_state_from_before_the_change_reads_as_one_with_no_leftover():
    s = ResumableDistributedSampler(_Count(10))
    s.load_state_dict({"seed": 0, "epoch": 2, "index": 10})  # written at an epoch's end
    assert (s.epoch, s.index, list(s.leftover)) == (3, 0, [])


def _two(rank, world_size, seed):
    """Two sources of different sizes, lengths and weights in one loader: each merges with its own sampler."""
    return MixedDataLoader({"a": (_Items(13, rank, world_size, seed, tag="a"), 2.0),
                            "b": (_Items(7, rank, world_size, seed + 1, max_len=5, tag="b"), 1.0)},
                           build_supervised_sample_fn=build_NTP_supervised_sample,
                           create_batcher_fn=partial(PackingBatcher, shape=(2, 12), pad_values=PADS))


@pytest.mark.parametrize("seed", range(4))
def test_two_sources_each_merged_by_its_own(seed):
    rng = random.Random(100 + seed)
    delivered, run = [], None
    for _ in range(8):
        world_size = rng.choice([1, 2, 3])
        steps = rng.randint(0 if run is not None else 1, 6)
        per = [steps + (rng.randint(0, 2) if rng.random() < 0.3 else 0) for _ in range(world_size)]
        run = _round(None, world_size, run, per, delivered, make=lambda r, w: _two(r, w, seed))
        _check(13, seed, run, delivered, "a")
        _check(7, seed + 1, run, delivered, "b")
    assert {t for t, _, _ in delivered} == {"a", "b"}


class _Plain(DataSource):
    """A sampler-backed source without merge_rank_states."""

    def __init__(self, n, rank, world_size):
        self.sampler = ResumableDistributedSampler(_Count(n), seed=5)
        self.sampler.rank, self.sampler.world_size = rank, world_size

    def __iter__(self):
        for j in self.sampler:
            yield {"tokens": torch.full((3,), j + 1)}

    def get_state(self):
        return self.sampler.state_dict()

    def set_state(self, state):
        self.sampler.load_state_dict(state)


def test_a_source_without_a_merge_keeps_rank_0s_state_and_says_so_once(capsys):
    def make(rank, world_size):
        return MixedDataLoader({"merged": (_Items(20, rank, world_size), 1.0), "plain": (_Plain(20, rank, world_size), 1.0)},
                               build_supervised_sample_fn=build_NTP_supervised_sample,
                               create_batcher_fn=partial(PaddingBatcher, shape=(2, 12), pad_values=PADS))
    loaders = [make(r, 2) for r in range(2)]
    snaps = []
    for r, loader in enumerate(loaders):
        it = iter(loader)
        for _ in range(r + 1):                                     # rank 1 one batch further
            snap = next(it)["state_dict"]
        snaps.append(snap)
    capsys.readouterr()
    for _ in range(2):
        run = loaders[0].merge_rank_snapshots(snaps)
        assert run["source_states"]["plain"] == snaps[0]["source_states"]["plain"]
        assert "shard" not in run["source_states"]["merged"]
    assert capsys.readouterr().out.count("has no merge_rank_states") == 1
    one = make(0, 1)
    one.merge_rank_snapshots([next(iter(one))["state_dict"]])
    assert "has no merge_rank_states" not in capsys.readouterr().out
