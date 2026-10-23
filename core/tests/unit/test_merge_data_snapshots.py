"""Checkpoint saving writes the run's data snapshot: every rank's gathered and merged (merge_data_snapshots).

Two processes on gloo save through save_checkpoint_if_needed with a stand-in loader whose snapshot names its rank:
meta.json holds the merge of both, not rank 0's alone. A merge that fails on its input fails on every rank, before
the checkpoint's own collectives, so neither process is left waiting (the test has a timeout). One process merges
its own snapshot without communication; a loader without merge_rank_snapshots has rank 0's written, as before.
"""

import json
import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.model.checkpoint_manager import merge_data_snapshots, save_checkpoint_if_needed


class _Loader:
    def __init__(self, fail=False):
        self.fail, self.seen = fail, None

    def merge_rank_snapshots(self, snapshots):
        if self.fail:
            raise ValueError("cannot merge these")
        self.seen = snapshots
        return {"source_states": {"s": sorted(x["source_states"]["s"] for x in snapshots)}}


def _worker(rank, init_method, out_dir, fail):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        config = {"checkpoint": {"save_dir": out_dir, "save_every": 1, "keep_last_n": -1}}
        try:
            save_checkpoint_if_needed(0, model, [opt], config, rank, 2,
                                      dataloader_state={"source_states": {"s": f"rank{rank}"}},
                                      dataloader=_Loader(fail), max_steps=1)
            outcome = "saved"
        except ValueError as e:
            outcome = f"raised: {e}"
        with open(f"{out_dir}/outcome{rank}.txt", "w") as f:
            f.write(outcome)
    finally:
        dist.destroy_process_group()


def _spawn(tmp_path, fail):
    ctx = mp.spawn(_worker, args=(f"file://{tmp_path / 'pg'}", str(tmp_path), fail), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    while not ctx.join(timeout=1):                  # True once every process has exited
        if time.monotonic() > deadline:
            for p in ctx.processes:
                p.kill()
            pytest.fail("a rank was left waiting")
    return [open(tmp_path / f"outcome{r}.txt").read() for r in range(2)]


def test_meta_holds_the_merge_of_every_ranks_snapshot(tmp_path):
    assert _spawn(tmp_path, fail=False) == ["saved", "saved"]
    meta = json.load(open(tmp_path / "step_000000" / "meta.json"))
    assert meta["dataloader_state"] == {"source_states": {"s": ["rank0", "rank1"]}}


def test_a_merge_that_fails_fails_on_every_rank(tmp_path):
    outcomes = _spawn(tmp_path, fail=True)
    assert all(o.startswith("raised: cannot merge") for o in outcomes), outcomes


def test_one_rank_merges_its_own_and_a_loader_without_a_merge_keeps_its_snapshot():
    loader = _Loader()
    assert merge_data_snapshots(loader, {"source_states": {"s": "a"}}, 1) == {"source_states": {"s": ["a"]}}
    assert loader.seen == [{"source_states": {"s": "a"}}]
    assert merge_data_snapshots(object(), {"x": 1}, 4) == {"x": 1}
    assert merge_data_snapshots(loader, None, 4) is None
