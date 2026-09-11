"""metrics.jsonl gets the same events core sends to wandb (and needs no wandb).

Real Trainer.train() with a tiny system (on the GPU — the Trainer binds
torch.cuda; skipped without one). Both kinds of event (periodic training metrics,
evaluator results) land as separate lines; a resumed run appends rather than
truncates; other ranks create no file.
"""

import json
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from core.model.gpt import GPTConfig
from core.model.system import LMSystem
from core.training.optim import build_optimizers
from projects.nano_multimodal.metrics import TrainerWithJsonl, _jsonable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="Trainer binds torch.cuda; no GPU here")

V, D, T, B = 16, 4, 8, 2
TINY = GPTConfig(sequence_len=T, vocab_size=V, n_layer=1, n_head=1, n_kv_head=1,
                 n_embd=D, n_token_types=1)


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TINY
        self.wte = torch.nn.Embedding(V, D)
        self.lin = torch.nn.Linear(D, D)

    def forward(self, idx, token_types=None):
        return self.lin(self.wte(idx))

    def estimate_flops(self):
        return 6 * D * D


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(D, V)

    def loss(self, hidden, targets):
        return F.cross_entropy(self.proj(hidden).flatten(0, 1), targets.flatten())


def _batches():
    g = torch.Generator().manual_seed(0)
    while True:
        idx = torch.randint(0, V, (B, T), generator=g)
        yield {"idx": idx, "targets": idx.roll(-1, dims=1)}


class _Eval:
    eval_at, interval_steps = None, 1        # only read by the info printout

    def should_eval(self, step):
        return step == 1

    def evaluate(self, system, autocast_ctx):
        return {"val/text_ce": 1.5}


CONFIG = {
    "max_steps": 3, "sequence_len": T, "device_batch_size": B, "total_batch_size": B * T,
    "optimizer": {"type": "adamw", "lr_max": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95],
                  "embedding_lr": 0.2, "unembedding_lr": 0.004, "max_grad_norm": 1.0,
                  "scheduler": {"type": "linear", "warmup_steps": 0, "warmdown_ratio": 0.2}},
    "logging": {"log_every": 1, "wandb_log_every": 1},
    "checkpoint": {"enabled": False},
    "wandb": {"enabled": False},
}


def _run(path, config=CONFIG, rank=0):
    torch.manual_seed(0)
    system = LMSystem(_Trunk(), _Head())
    opts = build_optimizers(system, config["optimizer"], 1)
    TrainerWithJsonl(system=system, optimizers=opts, dataloader=_batches(), config=config,
                     rank=rank, world_size=1, evaluators=[_Eval()], metrics_path=path).train()


def _rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_both_event_kinds_land_and_a_resume_appends(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)       # no wandb anywhere
    path = tmp_path / "metrics.jsonl"
    _run(path)
    rows = _rows(path)
    assert [r["step"] for r in rows if "train/loss" in r] == [0, 1, 2]
    assert [r["step"] for r in rows if "val/text_ce" in r] == [1, 2]     # scheduled + forced final
    assert len(rows) == 5
    _run(path)
    assert len(_rows(path)) == 10, "a second run must append, not truncate"


def test_other_ranks_create_no_file(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    path = tmp_path / "metrics.jsonl"
    _run(path, rank=1)
    assert not path.exists()


def test_wandb_on_writes_the_file_and_forwards_the_same_events(tmp_path, monkeypatch):
    forwarded = []
    run = types.SimpleNamespace(log=lambda d: forwarded.append(dict(d)), finish=lambda: None)
    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: run
    monkeypatch.setitem(sys.modules, "wandb", fake)
    path = tmp_path / "metrics.jsonl"
    _run(path, {**CONFIG, "wandb": {"enabled": True}})
    assert _rows(path) == [json.loads(json.dumps({k: _jsonable(v) for k, v in e.items()}))
                           for e in forwarded]
