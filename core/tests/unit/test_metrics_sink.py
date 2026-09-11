"""The metrics sink seam (queue #4).

Trainer.train() creates ONE sink on rank 0 through `_init_metrics()` and sends both
the periodic training metrics and every evaluator's results through its
`.log(dict)`; `.finish()` runs at a normal end. The default sink is wandb; a
subclass returning its own object needs wandb neither installed nor initialised,
and the other ranks get the no-op without the override being consulted.

Real `Trainer.train()` with a tiny system — on the GPU: the Trainer binds
torch.cuda.synchronize and asks the device its name, so these skip without one.
`wandb` is swapped in `sys.modules` so each test proves what is (not) imported:
`None` there makes `import wandb` raise ImportError.
"""

import subprocess
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from core.model.gpt import GPTConfig
from core.model.system import LMSystem
from core.training.optim import build_optimizers
from core.training.trainer import Trainer
from core.utils import DummyWandb

REPO = str(__import__('pathlib').Path(__file__).resolve().parents[3])

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="Trainer binds torch.cuda; no GPU here")

V, D, T, B = 16, 4, 8, 2
TINY = GPTConfig(sequence_len=T, vocab_size=V, n_layer=1, n_head=1, n_kv_head=1,
                 n_embd=D, n_token_types=1)


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TINY
        self.wte = torch.nn.Embedding(V, D)     # -> embedding group
        self.lin = torch.nn.Linear(D, D)        # -> matrix group

    def forward(self, idx, token_types=None):
        return self.lin(self.wte(idx))

    def estimate_flops(self):
        return 6 * D * D                        # only feeds the MFU printout


class _Head(torch.nn.Module):                   # -> unembedding group
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
        return step % 2 == 0

    def evaluate(self, system, autocast_ctx):
        return {"val/ce": 1.5}


CONFIG = {
    "max_steps": 3, "sequence_len": T, "device_batch_size": B, "total_batch_size": B * T,
    "optimizer": {"type": "adamw", "lr_max": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95],
                  "embedding_lr": 0.2, "unembedding_lr": 0.004, "max_grad_norm": 1.0,
                  "scheduler": {"type": "linear", "warmup_steps": 0, "warmdown_ratio": 0.2}},
    "logging": {"log_every": 1, "wandb_log_every": 1},
    "checkpoint": {"enabled": False},
    "wandb": {"enabled": True, "project": "p", "name": "n"},
}


class _Sink:
    def __init__(self):
        self.events, self.finished = [], 0

    def log(self, data):
        self.events.append(dict(data))

    def finish(self):
        self.finished += 1


def _trainer(cls=Trainer, rank=0, config=CONFIG):
    torch.manual_seed(0)
    system = LMSystem(_Trunk(), _Head())
    opts = build_optimizers(system, config["optimizer"], 1)
    return cls(system=system, optimizers=opts, dataloader=_batches(), config=config,
               rank=rank, world_size=1, evaluators=[_Eval()])


def _check_events(sink):
    train = [e for e in sink.events if "train/loss" in e]
    evals = [e for e in sink.events if "val/ce" in e]
    assert [e["step"] for e in train] == [0, 1, 2]
    assert [e["step"] for e in evals] == [0, 2]
    assert all("val/ce" not in e for e in train) and all("train/loss" not in e for e in evals), \
        "training and eval metrics are separate events, never merged into one row"
    assert sink.finished == 1


# ---- counter-examples --------------------------------------------------------------

def test_other_ranks_get_the_noop_and_never_call_the_override():
    """Read together with test_an_override_receives_every_event_without_wandb: this
    one alone would also pass if the Trainer stopped calling _init_metrics at all."""
    class T(Trainer):
        def _init_metrics(self):
            raise AssertionError("rank 1 must not create a sink")

    _trainer(T, rank=1).train()


def test_the_old_name_is_gone():
    """An override of the old name would be silently ignored, not called."""
    assert not hasattr(Trainer, "_init_wandb")


def test_importing_the_trainer_module_does_not_import_wandb():
    """Out of process: in this process pytest may have imported wandb already."""
    subprocess.run([sys.executable, "-c",
                    "import sys, core.training.trainer; assert 'wandb' not in sys.modules"],
                   check=True, cwd=REPO)


# ---- the seam ----------------------------------------------------------------------

def test_an_override_receives_every_event_without_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    sink = _Sink()

    class T(Trainer):
        def _init_metrics(self):
            return sink

    _trainer(T).train()
    _check_events(sink)


def test_default_disabled_or_missing_wandb_is_the_noop(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", None)
    off = {**CONFIG, "wandb": {"enabled": False}}
    assert isinstance(_trainer(config=off)._init_metrics(), DummyWandb)
    assert isinstance(_trainer()._init_metrics(), DummyWandb)      # enabled, not installed
    assert "wandb is not installed" in capsys.readouterr().out
    _trainer().train()      # no assertion: finishing the run without wandb is the point


def test_default_is_wandb_when_installed(monkeypatch):
    run, inits = _Sink(), []
    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: (inits.append(kw), run)[1]
    monkeypatch.setitem(sys.modules, "wandb", fake)

    _trainer().train()
    _check_events(run)
    (kw,) = inits
    assert (kw["project"], kw["name"]) == ("p", "n")
    assert kw["config"]["max_steps"] == 3 and "effective_initial_lr" in kw["config"]
