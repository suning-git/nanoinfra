"""LR schedules: build_lr_schedule reads optimizer.scheduler and checks every rule once, when the run's max_steps is
known; the schedule it returns only computes. The Trainer builds it at construction, after resolving max_steps -1,
before a checkpoint is loaded or a step is trained, and sets every group's lr from it before each step. CPU only.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from core.model.gpt import GPTConfig
from core.model.system import LMSystem
from core.training.lr_schedulers import CosineSchedule, LinearSchedule, build_lr_schedule


def test_linear_warms_up_holds_and_warms_down():
    s = build_lr_schedule({"type": "linear", "warmup_steps": 2, "warmdown_ratio": 0.5, "final_lr_frac": 0.1},
                          max_steps=8)
    assert s == LinearSchedule(max_steps=8, warmup_steps=2, warmdown_ratio=0.5, final_lr_frac=0.1)
    assert s.warmdown_steps == 4
    assert [s(i) for i in range(4)] == [0.5, 1.0, 1.0, 1.0]
    assert s(4) == 1.0 and s(5) == pytest.approx(0.75 + 0.25 * 0.1) and s(7) == pytest.approx(0.25 + 0.75 * 0.1)


def test_the_warmdown_length_is_rounded_not_truncated():
    """(3/47) * 47 is 2.9999999999999996 in floating point: three warmdown steps, as a recipe that writes the ratio as
    steps / max_steps expects, not two. Python's round: halves go to the even neighbour (2.5 -> 2, 3.5 -> 4)."""
    s = build_lr_schedule({"warmdown_ratio": 3 / 47}, max_steps=47)
    assert s.warmdown_steps == 3 and s(44) == 1.0 and s(45) < 1.0
    assert build_lr_schedule({"warmdown_ratio": 0.5}, max_steps=5).warmdown_steps == 2
    assert build_lr_schedule({"warmdown_ratio": 0.5}, max_steps=7).warmdown_steps == 4


@pytest.mark.parametrize("ratio", [0.0, 0.04])     # 0.04 x 10 rounds to no warmdown step
def test_no_warmdown_holds_the_peak_to_the_last_step(ratio):
    s = build_lr_schedule({"warmdown_ratio": ratio, "final_lr_frac": 0.1}, max_steps=10)
    assert s.warmdown_steps == 0 and [s(i) for i in range(10)] == [1.0] * 10


def test_cosine_warms_up_then_anneals_toward_final_lr_frac():
    s = build_lr_schedule({"type": "cosine", "warmup_steps": 1, "final_lr_frac": 0.0}, max_steps=5)
    assert s == CosineSchedule(max_steps=5, warmup_steps=1, final_lr_frac=0.0)
    assert s(0) == 1.0 and s(1) == 1.0 and s(3) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))
    s = build_lr_schedule({"type": "cosine", "final_lr_frac": 0.2}, max_steps=4)
    assert s(0) == 1.0 and s(2) == pytest.approx(0.2 + 0.8 * 0.5) and s(4) == pytest.approx(0.2)


def test_the_defaults_live_in_the_builder():
    assert build_lr_schedule({}, max_steps=10) == LinearSchedule(10, 0, 0.2, 0.0)


@pytest.mark.parametrize("config, max_steps, message", [
    ({"type": "step"}, 10, "Unknown scheduler type: 'step'"),
    ({"warmup_ratio": 0.1}, 10, "warmup_ratio was removed"),
    ({}, 0, "max_steps must be positive"),
    ({"final_lr_frac": 1.5}, 10, "final_lr_frac must be in"),
    ({"warmdown_ratio": 1.0}, 10, "warmdown_ratio must be in"),
    ({"warmup_steps": 9, "warmdown_ratio": 0.2}, 10, r"warmup_steps \(9\) \+ warmdown \(2\) exceed max_steps \(10\)"),
    ({"warmup_steps": -1}, 10, r"warmup_steps must be >= 0, got -1"),
    ({"type": "cosine", "warmup_steps": 10}, 10, r"warmup_steps must be in \[0, max_steps\)"),
])
def test_every_rule_is_checked_when_the_schedule_is_built(config, max_steps, message):
    with pytest.raises(ValueError, match=message):
        build_lr_schedule(config, max_steps)


# ------------------------------------------------- in the Trainer

V, D, T, B = 16, 4, 8, 2


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = GPTConfig(sequence_len=T, vocab_size=V, n_layer=1, n_head=1, n_kv_head=1, n_embd=D)
        self.wte, self.lin = torch.nn.Embedding(V, D), torch.nn.Linear(D, D)

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


def _build_trainer(scheduler, trainer_cls=None):
    """max_steps -1: 164 parameters x ratio 1 = 164 tokens = 10 steps of 16."""
    from core.training.optim import build_optimizers
    from core.training.trainer import Trainer
    torch.manual_seed(0)
    opt = {"type": "adamw", "lr_max": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "embedding_lr": 0.2,
           "unembedding_lr": 0.004, "max_grad_norm": 1.0, "scheduler": scheduler}
    system = LMSystem(_Trunk(), _Head(), prepare_trunk_inputs_fn=lambda batch: (batch["idx"], {}))
    config = {"max_steps": -1, "target_param_data_ratio": 1, "sequence_len": T, "device_batch_size": B,
              "total_batch_size": B * T, "optimizer": opt, "logging": {"log_every": 100}, "wandb": {"enabled": False},
              "checkpoint": {"enabled": False}}
    return (trainer_cls or Trainer)(system=system, optimizers=build_optimizers(system, opt), dataloader=_batches(),
                                    config=config, rank=0, world_size=1, evaluators=[])


def test_the_trainer_checks_the_schedule_against_the_max_steps_it_resolved(cpu_cuda_stubs, monkeypatch):
    """A warmup of 9 with a warmdown of 2 does not fit the resolved 10 steps: refused when the Trainer is built, before
    a checkpoint is loaded, not at the first step."""
    def never_loaded(*a, **k):
        raise AssertionError("a refused schedule must stop the Trainer before any checkpoint is loaded")
    monkeypatch.setattr("core.model.checkpoint_manager.load_checkpoint_if_needed", never_loaded)
    with pytest.raises(ValueError, match=r"warmup_steps \(9\) \+ warmdown \(2\) exceed max_steps \(10\)"):
        _build_trainer({"type": "linear", "warmup_steps": 9, "warmdown_ratio": 0.2})


def test_every_step_trains_with_the_schedules_lr(cpu_cuda_stubs):
    """The real Trainer.train(): before each step's update, every group's lr is its initial_lr times the schedule at
    that step (not the next one, not after the update)."""
    from core.training.trainer import Trainer
    seen = []

    class Recording(Trainer):
        def _training_step(self, data_iter, step):
            seen.append([(g["lr"], g["initial_lr"]) for opt in self.optimizers for g in opt.param_groups])
            return super()._training_step(data_iter, step)

    scheduler = {"type": "linear", "warmup_steps": 2, "warmdown_ratio": 0.5, "final_lr_frac": 0.1}
    _build_trainer(scheduler, Recording).train()
    schedule = build_lr_schedule(scheduler, max_steps=10)
    assert len(seen) == 10
    for step, groups in enumerate(seen):
        assert all(lr == initial * schedule(step) for lr, initial in groups), step


def test_an_orchestrators_schedule_replaces_the_builder(cpu_cuda_stubs, monkeypatch):
    """H12: replacing lr_schedulers.build_lr_schedule before the Trainer is built gives the Trainer that schedule, with
    the max_steps it resolved."""
    calls = []

    def build_constant(config, max_steps):
        calls.append((config, max_steps))
        return lambda step: 0.5
    monkeypatch.setattr("core.training.lr_schedulers.build_lr_schedule", build_constant)
    trainer = _build_trainer({"type": "my_own"})
    assert calls == [({"type": "my_own"}, 10)] and trainer.lr_schedule(3) == 0.5
