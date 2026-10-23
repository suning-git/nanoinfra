"""The val NELBO across ranks: one GPU or many, the same reading.

Every rank holds the same plan of batches; rank r of N evaluates batches r, r + N, ... and core adds each level's CE
sum and masked count over the ranks (a rank left a batch short runs an all-IGNORE copy of the first). The rank and
world size come from the launcher's environment (core.utils.process_identity), so each worker sets them. On the CPU:
the tiny assembly with a real GPT, so the batches go through the System's own prepare_trunk_inputs (core's
evaluate_loss). The reference is computed independently: each level's CE summed over every masked cell of every val
clip, over their number, averaged over the levels.

    pytest exemplars/nano_world_model/tests/test_wm_eval_shard.py
"""

import contextlib
import math

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.data.layout import collate_samples, padding_layout
from core.data.supervision import IGNORE

from exemplars.nano_world_model import spec
from exemplars.nano_world_model.block_diffusion import BlockDiffusionSystem
from exemplars.nano_world_model.dataset import VideoRowDataset
from exemplars.nano_world_model.evaluator import NELBOEvaluator, level_sample
from exemplars.nano_world_model.rope3d import install_rope3d
from exemplars.nano_world_model.tests.tiny import N_VIDEO, tiny_assembly, tiny_gpt

N_VAL = 13                                               # rows in the val split; the evaluator spreads over them


class _Val:
    """A val split: VideoRowDataset's codes, actions and spread, without a cache on disk."""
    split, spread = "val", VideoRowDataset.spread

    def __init__(self, rows):
        g = np.random.default_rng(2)
        c = rows.contract
        self.codes = g.integers(0, N_VIDEO, (N_VAL, c["code_len"]))
        self.actions = g.integers(0, spec.N_ACTIONS, (N_VAL, c["n_action_tokens"]))

    def __len__(self):
        return N_VAL


def _system(layout, rows):
    trunk, head = tiny_gpt(layout, rows)
    install_rope3d(trunk, rows)
    return BlockDiffusionSystem(trunk, head, rows, 0.2, 1.0).eval()


def _reading(n_rows, batch, t_grid):
    layout, resolver, rows, pads, mask_id = tiny_assembly()
    ev = NELBOEvaluator(_Val(rows), rows, layout, mask_id, pads, n_rows=n_rows, t_grid=t_grid, batch=batch,
                        device="cpu")
    return ev.evaluate(_system(layout, rows), contextlib.nullcontext()), ev._eval.eval_steps


@torch.no_grad()
def _reference(n_rows, t_grid):
    layout, resolver, rows, pads, mask_id = tiny_assembly()
    val, system = _Val(rows), _system(layout, rows)
    sel = val.spread(n_rows)
    clips = rows.assemble(val.codes[sel], val.actions[sel])
    per_t = {}
    for t in t_grid:
        total, count = 0.0, 0
        for clip, row in zip(clips, sel):
            sample = level_sample(clip, t, row, rows, layout, mask_id)
            shape = (1, 2 * rows.row_len)
            b = collate_samples([sample], padding_layout([shape[1]], shape), shape=shape, pad_values=pads)
            idx, kw = system.prepare_trunk_inputs(b)
            logits = system.head(system.trunk(idx, **kw))[0].float()
            tgt = b["targets"][0]
            ce = torch.nn.functional.cross_entropy(logits, tgt, ignore_index=IGNORE, reduction="sum")
            total, count = total + ce.item(), count + int((tgt != IGNORE).sum())
        per_t[t] = total / count
    return sum(per_t.values()) / len(per_t), per_t


def _worker(rank, init_method, out_dir, n_rows, batch, t_grid):
    import os
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        torch.save(_reading(n_rows, batch, t_grid), f"{out_dir}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


# 10 clips in batches of 4, 4, 2 at two levels: 6 batches, 3 a rank. 3 clips at three levels: 3 batches, a filler.
@pytest.mark.parametrize("n_rows, batch, t_grid", [(10, 4, (0.3, 0.7)), (3, 4, (0.3, 0.5, 0.7))])
def test_two_ranks_read_what_one_rank_reads(n_rows, batch, t_grid, tmp_path, monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    (one, steps), (mean, per_t) = _reading(n_rows, batch, t_grid), _reference(n_rows, t_grid)
    assert steps == len(t_grid) * -(-n_rows // batch)
    assert math.isclose(one["val/nelbo"], mean, rel_tol=1e-5), (one["val/nelbo"], mean)
    assert all(math.isclose(one[f"val/nelbo_t{t}"], per_t[t], rel_tol=1e-5) for t in t_grid)
    mp.spawn(_worker, args=(f"file://{tmp_path / 'pg'}", str(tmp_path), n_rows, batch, t_grid), nprocs=2, join=True)
    for r in range(2):
        two, steps2 = torch.load(tmp_path / f"r{r}.pt")
        assert steps2 == -(-steps // 2)
        assert all(math.isclose(two[k], one[k], rel_tol=1e-12) for k in one), (r, two, one)
