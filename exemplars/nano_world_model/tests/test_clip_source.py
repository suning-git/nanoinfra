"""ClipSource: its state, the epoch it tags each clip with, and resume on another number of ranks.

  * The state reads back from JSON unchanged: it goes into the checkpoint's meta.json, which nothing checks when it is
    written (core.data.DataSource.get_state).
  * meta["epoch"] is the epoch the clip is read in — what the preparation seeds the clip's noise from — also for a
    leftover clip handed out after a resume, and on every rank.
  * Two ranks read different numbers of clips; their states, merged, resume on one rank: every clip of the stream is
    then read exactly once, none skipped or repeated, and each with the epoch it would have had.

    pytest exemplars/nano_world_model/tests/test_clip_source.py
"""

import json
from types import SimpleNamespace

import numpy as np
import torch

from exemplars.nano_world_model.dataset import ClipSource

N = 7                                                       # clips in the split


class _Dataset:
    split = "train"
    cache_dir = SimpleNamespace(name="tiny")
    codes = np.arange(N * 3, dtype=np.int64).reshape(N, 3)
    actions = np.zeros((N, 2), dtype=np.int64)

    def __len__(self):
        return N


_ROWS = SimpleNamespace(assemble=lambda codes, actions: torch.from_numpy(np.asarray(codes)))


def _source(rank=0, world=1):
    src = ClipSource(_Dataset(), _ROWS, seed=3)
    src.sampler.rank, src.sampler.world_size = rank, world
    return src


def _truth(g, seed=3):
    """(row, epoch) at global position g of the sampler's endless sequence."""
    e, p = divmod(g, N)
    return int(torch.randperm(N, generator=torch.Generator().manual_seed(seed + e))[p]), e


def test_the_state_reads_back():
    src = _source()
    it = iter(src)
    for _ in range(N + 3):                                  # into the second epoch
        next(it)
    state = src.get_state()
    assert json.loads(json.dumps(state)) == state


def test_each_clip_carries_its_epoch_on_every_rank():
    for world in (1, 2, 3):
        for rank in range(world):
            it = iter(_source(rank, world))
            for j in range(2 * N + 1):
                clip = next(it)
                row, epoch = _truth(j * world + rank)
                assert (clip["meta"]["row"], clip["meta"]["epoch"]) == (row, epoch), (world, rank, j)
                assert torch.equal(clip["tokens"], torch.from_numpy(_Dataset.codes[row]))


def test_two_ranks_merged_resume_on_one_every_clip_once():
    a, b = _source(0, 2), _source(1, 2)
    ia, ib = iter(a), iter(b)
    read = [next(ia)["meta"] for _ in range(5)] + [next(ib)["meta"] for _ in range(2)]   # rank 1 is behind
    merged = a.merge_rank_states([a.get_state(), b.get_state()])
    assert json.loads(json.dumps(merged)) == merged
    one = _source(0, 1)
    one.set_state(merged)
    it = iter(one)
    read += [next(it)["meta"] for _ in range(3 * N)]
    seen = [(m["row"], m["epoch"]) for m in read]
    front = 10                                              # rank 0 read positions 0, 2, ..., 8: the front is 10
    stream = [_truth(g) for g in range(front)]              # everything before it, each exactly once
    assert sorted(seen[:front]) == sorted(stream)
    assert seen[front:front + N] == [_truth(g) for g in range(front, front + N)]   # then on from the front
