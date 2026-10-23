"""SupervisedCE shares its frozen batches across ranks: one GPU or many, the same reading.

Every rank freezes the same K batches; rank r of N evaluates batches r, r + N, ... and core's LossEvaluator adds the
nats and the target counts over the ranks. Every rank runs ceil(K / N) forwards (under FSDP each is a collective): a
rank left one short adds a copy of the first batch with every target IGNORE. The rank and the world size come from
the launcher's environment (core.utils.process_identity reads RANK / LOCAL_RANK / WORLD_SIZE, not the process group),
so the two-process test sets them in each worker. CPU only, a stand-in System.

    pytest projects/nano_multimodal/tests/test_eval_shard.py
"""

import contextlib
import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.model.heads import LMHead
from core.model.system import prepare_unpacked_AR_trunk_inputs
from core.tokenization.vocab_layout import VocabLayout
from projects.nano_multimodal.evaluator import SupervisedCE

IGNORE = VocabLayout.IGNORE_INDEX
V, H = 16, 8


class _System(torch.nn.Module):
    """Embedding trunk + core's LMHead. Records the batch number (idx[0, 0]) of every forward, and the token types
    it was given."""
    prepare_trunk_inputs = staticmethod(prepare_unpacked_AR_trunk_inputs)

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.head = LMHead(H, V)
        with torch.no_grad():
            self.head.lm_head.weight.normal_(0, 1.0)
        self.seen, self.types_seen = [], []

    def trunk(self, x, token_types=None):
        self.seen.append(int(x[0, 0]))
        self.types_seen.append(token_types)
        return self.emb(x)


def _batches(k):
    """k batches of 2 rows x 6; batch b starts with b, and holds a different number of supervised targets."""
    g = torch.Generator().manual_seed(1)
    out = []
    for b in range(k):
        idx = torch.randint(0, V, (2, 6), generator=g)
        idx[0, 0] = b
        targets = torch.randint(0, V, (2, 6), generator=g)
        targets[:, :b % 4] = IGNORE
        out.append({"idx": idx, "targets": targets, "token_types": torch.randint(0, 3, (2, 6), generator=g),
                    "layout_facts": {"one_sample_per_row_left_aligned": True}})
    return out


def _launcher(env, rank, world_size):
    for name, value in (("RANK", rank), ("LOCAL_RANK", rank), ("WORLD_SIZE", world_size)):
        env.setenv(name, str(value))


@pytest.mark.parametrize("k, n", [(0, 2), (1, 1), (5, 1), (1, 2), (3, 2), (5, 2), (4, 2), (5, 3), (2, 3)])
def test_each_rank_takes_its_shard_and_runs_ceil_k_over_n_forwards(k, n, monkeypatch):
    batches = _batches(k)
    targets = [b["targets"].clone() for b in batches]
    for r in range(n):
        _launcher(monkeypatch, r, n)
        ev = SupervisedCE(iter(batches), n_batches=k, metric="m")
        shard, steps = ev._eval.dataloader, ev._eval.eval_steps
        assert steps == -(-k // n) == len(shard)
        mine = list(range(r, k, n))
        assert [int(b["idx"][0, 0]) for b in shard[:len(mine)]] == mine
        for filler in shard[len(mine):]:                     # at most one: a copy of batch 0 with no target
            assert filler["idx"] is batches[0]["idx"] and bool((filler["targets"] == IGNORE).all())
        assert len(shard) - len(mine) <= 1
        assert ev.n_batches == k
    assert all(torch.equal(b["targets"], t) for b, t in zip(batches, targets))        # batch 0 itself untouched


def test_the_reading_is_the_nats_over_all_supervised_targets_over_their_count(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    system, batches = _System(), _batches(5)
    got = SupervisedCE(iter(batches), n_batches=5, metric="m").evaluate(system, contextlib.nullcontext())["m"]
    nats, count = 0.0, 0
    with torch.no_grad():
        for b in batches:
            keep = b["targets"] != IGNORE
            nats += float(torch.nn.functional.cross_entropy(system.head(system.emb(b["idx"]))[keep], b["targets"][keep],
                                                            reduction="sum"))
            count += int(keep.sum())
    assert math.isclose(got, nats / count, rel_tol=1e-6)


def test_the_trainers_autocast_is_entered_once_and_left_as_it_was(monkeypatch):
    """The Trainer hands every evaluate() the same autocast object. Entered twice (by this wrapper and inside
    LossEvaluator), it stays on after the outer exit, and what the Trainer runs next would run in bf16."""
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    ev, system = SupervisedCE(iter(_batches(3)), n_batches=3, metric="m"), _System()
    ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    for _ in range(2):
        ev.evaluate(system, ctx)
        assert not torch.is_autocast_enabled("cpu")


def test_the_forward_gets_the_token_types_and_the_preparation_its_layout_facts(monkeypatch):
    """The frozen batches keep what the forward and the preparation read: the token types reach the trunk (without
    them the type embedding would be left out and the reading silently change), and layout_facts reach
    prepare_unpacked_AR_trunk_inputs, which refuses a packed batch by them."""
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    batches, system = _batches(3), _System()
    ev = SupervisedCE(iter(batches), n_batches=3, metric="m")
    ev.evaluate(system, contextlib.nullcontext())
    assert all(torch.equal(t, b["token_types"]) for t, b in zip(system.types_seen, batches))
    packed = _batches(1)
    packed[0]["layout_facts"] = {"one_sample_per_row_left_aligned": False}
    with pytest.raises(ValueError, match="packed"):
        SupervisedCE(iter(packed), n_batches=1, metric="m").evaluate(_System(), contextlib.nullcontext())


def _reading(k):
    system = _System()
    ev = SupervisedCE(iter(_batches(k)), n_batches=k, metric="m")
    return [ev.evaluate(system, contextlib.nullcontext())["m"] for _ in range(2)], system.seen


def _worker(rank, init_method, out_dir, k):
    import os
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        torch.save(_reading(k), f"{out_dir}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("k", [5, 4, 1])
def test_two_ranks_read_what_one_rank_reads(k, tmp_path, monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    (one, _) = _reading(k)
    mp.spawn(_worker, args=(f"file://{tmp_path / 'pg'}", str(tmp_path), k), nprocs=2, join=True)
    for r in range(2):
        two, seen = torch.load(tmp_path / f"r{r}.pt")
        for value in two:                                   # evaluated twice: nothing carries over
            assert math.isclose(value, one[0], rel_tol=1e-12), (r, value, one[0])
        mine = list(range(r, k, 2))
        per_eval = mine + [0] * (-(-k // 2) - len(mine))    # the filler is batch 0's input
        assert seen == per_eval * 2, (r, seen)
