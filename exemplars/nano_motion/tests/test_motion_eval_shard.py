"""nano_motion's SupervisedCEEvaluator: every supervised token counts once, and one GPU or many read the same.

The reading is the nats over all supervised targets divided by their count (core's mean_ce), not a mean of batch
means. Every rank assembles the same K batches; rank r of N evaluates batches r, r + N, ... and core's LossEvaluator
adds over the ranks; every rank runs ceil(K / N) forwards (a rank left one short adds a copy of the first batch with
every target IGNORE). The rank and world size come from the launcher's environment (core.utils.process_identity), so
the two-process test sets them in each worker. CPU only: a stand-in source of raw samples and a stand-in System.

    pytest exemplars/nano_motion/tests/test_motion_eval_shard.py
"""

import contextlib
import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from core.data.supervision import build_NTP_supervised_sample
from core.model.heads import LMHead
from core.model.system import prepare_packed_AR_trunk_inputs, prepare_unpacked_AR_trunk_inputs
from core.tokenization.vocab_layout import VocabLayout
from exemplars.nano_motion.train_t2m import SupervisedCEEvaluator

IGNORE = VocabLayout.IGNORE_INDEX
V, H, T, ROWS = 16, 8, 8, 2
EOS = 15


class _Resolver:
    def resolve(self, name):
        return EOS


class _Layout:
    def classify_token_types(self, ids):
        return torch.zeros_like(ids)


class _System(torch.nn.Module):
    """Embedding trunk + core's LMHead. Records the batch number (idx[0, 0]) of every forward, and what else the
    preparation handed the trunk."""
    prepare_trunk_inputs = staticmethod(prepare_unpacked_AR_trunk_inputs)

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.head = LMHead(H, V)
        with torch.no_grad():
            self.head.lm_head.weight.normal_(0, 1.0)
        self.seen, self.kwargs_seen = [], []

    def trunk(self, x, **kwargs):
        self.seen.append(int(x[0, 0]))
        self.kwargs_seen.append(kwargs)
        return self.emb(x)


class _PackedSystem(_System):
    prepare_trunk_inputs = staticmethod(prepare_packed_AR_trunk_inputs)


def _samples(k):
    """Raw samples for k batches of ROWS: in batch b every sample starts with b; lengths and supervised spans vary,
    so the batches hold different numbers of supervised targets (as T2M pairs do)."""
    g = torch.Generator().manual_seed(1)
    out = []
    for b in range(k):
        for row in range(ROWS):
            n = 4 + (b + row) % (T - 2)                          # raw tokens, <= T + 1
            tokens = torch.randint(0, EOS, (n,), generator=g)
            tokens[0] = b
            mask = torch.zeros(n, dtype=torch.bool)
            mask[1 + (b % 2):] = True
            out.append({"tokens": tokens, "token_types": torch.randint(0, 2, (n,), generator=g),
                        "supervision_mask": mask, "meta": {"window": len(out)}})
    return out


def _evaluator(k):
    ecfg = {"n_batches": k, "batch": ROWS, "interval_steps": 10}
    return SupervisedCEEvaluator(ecfg, iter(_samples(k)), _Layout(), _Resolver(), T, "cpu")


def _launcher(env, rank, world_size):
    for name, value in (("RANK", rank), ("LOCAL_RANK", rank), ("WORLD_SIZE", world_size)):
        env.setenv(name, str(value))


def _no_launcher(env):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.delenv(name, raising=False)


def test_every_supervised_token_counts_once(monkeypatch):
    """The reading is the pooled mean over the supervised targets, which differs here from the mean of batch means."""
    _no_launcher(monkeypatch)
    ev, system = _evaluator(3), _System()
    got = ev.evaluate(system, contextlib.nullcontext())["val/motion_ce"]
    sums, counts = [], []
    with torch.no_grad():
        for b in ev._eval.dataloader:
            keep = b["targets"] != IGNORE
            ce = F.cross_entropy(system.head(system.emb(b["idx"]))[keep], b["targets"][keep], reduction="sum")
            sums.append(float(ce))
            counts.append(int(keep.sum()))
    assert len(set(counts)) > 1
    assert math.isclose(got, sum(sums) / sum(counts), rel_tol=1e-6)
    assert not math.isclose(got, sum(s / c for s, c in zip(sums, counts)) / len(sums), rel_tol=1e-3)


def _round_and_round(samples):
    """The val source as the real ones are: endless, each pass the same samples in a new order."""
    g = torch.Generator().manual_seed(2)
    while True:
        for i in torch.randperm(len(samples), generator=g).tolist():
            yield samples[i]


@pytest.mark.parametrize("n_samples", [5, 6, 7])
def test_a_val_set_smaller_than_the_batches_is_evaluated_once_whole(n_samples, monkeypatch):
    """Asked for 4 batches of 2 from a source of 5-7 samples that goes round again, the evaluator takes one pass:
    every sample once (LAFAN1's 159 windows against 20 batches of 8 counted one twice before), the last batch short
    when the count is odd."""
    _no_launcher(monkeypatch)
    samples = _samples(4)[:n_samples]
    ecfg = {"n_batches": 4, "batch": ROWS, "interval_steps": 10}
    ev = SupervisedCEEvaluator(ecfg, _round_and_round(samples), _Layout(), _Resolver(), T, "cpu")
    batches = ev._eval.dataloader
    assert [int(b["idx"].shape[0]) for b in batches] == [ROWS] * (n_samples // ROWS) + [1] * (n_samples % ROWS)
    taken = sorted(tuple(row.tolist()) for b in batches for row in b["idx"])
    expected = sorted(tuple(build_NTP_supervised_sample(s)["data"]["idx"].tolist()
                            + [EOS] * (T - (len(s["tokens"]) - 1))) for s in samples)
    assert taken == expected
    assert ev.n_supervised == sum(int(s["supervision_mask"][1:].sum()) for s in samples)
    assert ev.n_samples == n_samples and ev.whole_set


def test_a_val_set_larger_than_the_batches_gives_the_first_k_batches(monkeypatch):
    _no_launcher(monkeypatch)
    ecfg = {"n_batches": 3, "batch": ROWS, "interval_steps": 10}
    ev = SupervisedCEEvaluator(ecfg, iter(_samples(4)), _Layout(), _Resolver(), T, "cpu")
    assert [int(b["idx"][0, 0]) for b in ev._eval.dataloader] == [0, 1, 2]
    assert (ev.n_samples, ev.n_batches, ev.whole_set) == (3 * ROWS, 3, False)


@pytest.mark.parametrize("k, n", [(0, 2), (1, 1), (3, 1), (1, 2), (3, 2), (4, 2), (5, 3)])
def test_each_rank_takes_its_shard_and_runs_ceil_k_over_n_forwards(k, n, monkeypatch):
    for r in range(n):
        _launcher(monkeypatch, r, n)
        ev = _evaluator(k)
        shard, steps = ev._eval.dataloader, ev._eval.eval_steps
        assert steps == -(-k // n) == len(shard)
        mine = list(range(r, k, n))
        assert [int(b["idx"][0, 0]) for b in shard[:len(mine)]] == mine
        for filler in shard[len(mine):]:                     # at most one: batch 0's input, no target
            assert int(filler["idx"][0, 0]) == 0 and bool((filler["targets"] == IGNORE).all())
        assert len(shard) - len(mine) <= 1
        assert all(bool((b["targets"] != IGNORE).any()) for b in shard[:len(mine)])   # real batches keep theirs


def test_the_trainers_autocast_is_entered_once_and_left_as_it_was(monkeypatch):
    """The Trainer hands every evaluate() the same autocast object. Entered twice (by this wrapper and inside
    LossEvaluator), it stays on after the outer exit, and what the Trainer runs next would run in bf16."""
    _no_launcher(monkeypatch)
    ev, system = _evaluator(3), _System()
    ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    for _ in range(2):
        ev.evaluate(system, ctx)
        assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("system_cls", [_System, _PackedSystem])
def test_the_preparation_gets_every_field_it_reads(system_cls, monkeypatch):
    """The batches keep what either preparation reads (training with batching=packing prepares packed): the token
    types reach the trunk, and the packed preparation's doc_pos arrives as position_ids, so doc_id and doc_pos were
    kept too."""
    _no_launcher(monkeypatch)
    ev, system = _evaluator(3), system_cls()
    ev.evaluate(system, contextlib.nullcontext())
    for kw, b in zip(system.kwargs_seen, ev._eval.dataloader):
        assert torch.equal(kw["token_types"], b["token_types"])
        if system_cls is _PackedSystem:
            assert torch.equal(kw["position_ids"], b["doc_pos"]) and "attention" in kw


def _reading(k, n_samples=None):
    """Two evaluations of k batches; with n_samples, of a source of that many samples going round again, asked for
    k batches of ROWS (one pass: the last batch short when n_samples is odd)."""
    system = _System()
    if n_samples is None:
        ev = _evaluator(k)
    else:
        ecfg = {"n_batches": k, "batch": ROWS, "interval_steps": 10}
        ev = SupervisedCEEvaluator(ecfg, _round_and_round(_samples(k)[:n_samples]), _Layout(), _Resolver(), T, "cpu")
    return [ev.evaluate(system, contextlib.nullcontext())["val/motion_ce"] for _ in range(2)], system.seen


def _worker(rank, init_method, out_dir, k, n_samples=None):
    import os
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        torch.save(_reading(k, n_samples), f"{out_dir}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("k", [5, 1])
def test_two_ranks_read_what_one_rank_reads(k, tmp_path, monkeypatch):
    _no_launcher(monkeypatch)
    (one, _) = _reading(k)
    mp.spawn(_worker, args=(f"file://{tmp_path / 'pg'}", str(tmp_path), k), nprocs=2, join=True)
    for r in range(2):
        two, seen = torch.load(tmp_path / f"r{r}.pt")
        for value in two:                                   # evaluated twice: nothing carries over
            assert math.isclose(value, one[0], rel_tol=1e-12), (r, value, one[0])
        mine = list(range(r, k, 2))
        assert seen == (mine + [0] * (-(-k // 2) - len(mine))) * 2, (r, seen)


def test_two_ranks_read_what_one_rank_reads_with_a_short_last_batch(tmp_path, monkeypatch):
    """7 samples asked as 4 batches of 2: batches of 2, 2, 2 and 1 row; rank 1 runs the short one. Each rank's
    forwards hold different numbers of rows, and the reading is still one rank's."""
    _no_launcher(monkeypatch)
    (one, _) = _reading(4, n_samples=7)
    mp.spawn(_worker, args=(f"file://{tmp_path / 'pg'}", str(tmp_path), 4, 7), nprocs=2, join=True)
    for r in range(2):
        two, _ = torch.load(tmp_path / f"r{r}.pt")
        for value in two:
            assert math.isclose(value, one[0], rel_tol=1e-12), (r, value, one[0])
