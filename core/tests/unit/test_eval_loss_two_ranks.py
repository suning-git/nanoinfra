"""evaluate_loss across two ranks (CPU, gloo) equals one rank holding both ranks' batches.

The metrics' rows are laid out by sorted name, so one all_reduce lines them up whatever each rank's batches hold and
whatever order each rank's dict lists them in: rank 1 gets the same metrics in reverse order (a dict built by
iterating a set of strings comes out in a different order per process, the hash seed differing). The fixture puts
data for some metrics on ONE rank only: targets of type 1 only on rank 1, byte-bearing targets only on rank 0, the
project metric's samples only on rank 1. The rank without data contributes (0, 0) to that row and the reading is the
other rank's ratio.
"""

import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.evaluation.eval_loss import evaluate_loss
from core.evaluation.metrics import bits_per_byte, mean_ce, type_ce
from core.model.heads import LMHead
from core.model.system import prepare_unpacked_AR_trunk_inputs
from core.tokenization.vocab_layout import VocabLayout

IGNORE = VocabLayout.IGNORE_INDEX
V, H = 16, 8
CONTROL = [12, 13, 14, 15]            # zero-byte ids


class _TinySystem(torch.nn.Module):
    prepare_trunk_inputs = staticmethod(prepare_unpacked_AR_trunk_inputs)

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.head = LMHead(H, V)
        with torch.no_grad():
            self.head.lm_head.weight.normal_(0, 1.0)

    def trunk(self, x, token_types=None):
        return self.emb(x)


def _flagged():
    """A project metric over the samples whose meta says flagged (a per-group breakdown has this form)."""
    def metric(batch, ce):
        flags = torch.tensor([m["flagged"] for m in batch["sample_meta"]])
        sel = flags[:, None].expand_as(ce) & (batch["targets"] != IGNORE)     # one sample per row here
        return ce[sel].sum(), sel.sum()
    return metric


def _metrics():
    table = torch.randint(1, 4, (V,), generator=torch.Generator().manual_seed(2))
    table[CONTROL] = 0
    return {"total": mean_ce(), "type0": type_ce(0), "type1": type_ce(1), "bpb": bits_per_byte(table),
            "flagged": _flagged()}


def _rank_batches(rank):
    g = torch.Generator().manual_seed(10 + rank)
    out = []
    for _ in range(2):
        idx = torch.randint(0, V, (2, 6), generator=g)
        if rank == 0:
            targets = torch.randint(0, 12, (2, 6), generator=g)                   # byte-bearing
            target_types = torch.zeros(2, 6, dtype=torch.long)                    # type 0 only
        else:
            targets = torch.tensor(CONTROL)[torch.randint(0, 4, (2, 6), generator=g)]   # no bytes
            target_types = torch.randint(0, 2, (2, 6), generator=g)               # types 0 and 1
        targets[0, 0] = IGNORE
        out.append({"idx": idx, "targets": targets, "target_types": target_types,
                    "sample_meta": [{"flagged": rank == 1}, {"flagged": False}]})
    return out


def _worker(rank, init_method, out_dir):
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        batches, metrics = _rank_batches(rank), _metrics()
        if rank == 1:
            metrics = dict(reversed(metrics.items()))
        torch.save(evaluate_loss(_TinySystem(), batches, len(batches), metrics), f"{out_dir}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_ranks_equal_one_rank_with_both_ranks_batches(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'pg'}", str(tmp_path)), nprocs=2, join=True)
    r0, r1 = torch.load(tmp_path / "r0.pt"), torch.load(tmp_path / "r1.pt")
    both = _rank_batches(0) + _rank_batches(1)
    want = evaluate_loss(_TinySystem(), both, len(both), _metrics())
    assert r0 == {name: r1[name] for name in r0}
    assert list(r0) == list(want)
    for name, value in want.items():
        assert math.isfinite(value)
        assert math.isclose(r0[name], value, rel_tol=1e-12), name
