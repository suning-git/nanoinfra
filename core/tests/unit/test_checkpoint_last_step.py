"""The last step is always saved, also when the Trainer computes the run length (`max_steps: -1`), and when
`num_iterations` stops a launch early.

save_checkpoint_if_needed used to read config['max_steps'] for its always-save-the-last-step rule. With -1 (the
Trainer resolves the length by the Chinchilla rule) that rule never fired: an auto-length run never wrote its final
model, and with `save_every: -1` (save only at the end) it wrote nothing at all. Found 2026-10-06 on text runs (the
text exemplar's shipped checkpoint stops at step 20000 of 20640 for the same reason).
"""

import pytest
import torch
import torch.nn.functional as F

from core.model import checkpoint_manager as cm
from core.model.gpt import GPTConfig
from core.model.system import LMSystem
from core.training.optim import build_optimizers

V, D, T, B = 16, 4, 8, 2
TINY = GPTConfig(sequence_len=T, vocab_size=V, n_layer=1, n_head=1, n_kv_head=1,
                 n_embd=D, n_token_types=1)
OPT = {"type": "adamw", "lr_max": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95],
       "embedding_lr": 0.2, "unembedding_lr": 0.004, "max_grad_norm": 1.0,
       "scheduler": {"type": "linear", "warmup_steps": 0, "warmdown_ratio": 0.2}}


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TINY
        self.wte = torch.nn.Embedding(V, D)     # -> embedding group
        self.lin = torch.nn.Linear(D, D)        # -> matrix group

    def forward(self, idx, token_types=None):
        return self.lin(self.wte(idx))

    def estimate_flops(self):
        return 6 * D * D


class _Head(torch.nn.Module):                   # -> unembedding group
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(D, V)

    def loss(self, hidden, targets):
        return F.cross_entropy(self.proj(hidden).flatten(0, 1), targets.flatten())


def _system_and_optimizers():
    torch.manual_seed(0)
    # how a batch becomes trunk inputs, chosen when the System is built (LMSystem.loss goes through it)
    system = LMSystem(_Trunk(), _Head(), prepare_trunk_inputs_fn=lambda batch: (batch["idx"], {}))
    return system, build_optimizers(system, OPT)


def _saved_steps(save_dir):
    return sorted(int(p.name.split("_")[1]) for p in save_dir.glob("step_*"))


def _auto_length_config(save_dir):
    return {"max_steps": -1, "checkpoint": {"save_dir": str(save_dir), "save_every": -1, "keep_last_n": 2}}


def test_the_resolved_length_decides_the_last_step(tmp_path):
    system, opts = _system_and_optimizers()
    for step in range(5):
        cm.save_checkpoint_if_needed(step, system, opts, _auto_length_config(tmp_path), 0, 1, max_steps=5)
    assert _saved_steps(tmp_path) == [4]


@pytest.mark.parametrize("ratio", [1, 1.0])
def test_an_auto_length_run_saves_its_last_step(tmp_path, cpu_cuda_stubs, ratio):
    """The real Trainer.train() on the CPU: max_steps -1, resolved from the parameter count (164 params x ratio 1 =
    164 tokens = 10 steps of 16), save only at the end, on the CPU (cpu_cuda_stubs). A float ratio must give an
    integer length too (10.0 used to break the step printout)."""
    from core.training.trainer import Trainer

    def batches():
        g = torch.Generator().manual_seed(0)
        while True:
            idx = torch.randint(0, V, (B, T), generator=g)
            yield {"idx": idx, "targets": idx.roll(-1, dims=1)}

    system, opts = _system_and_optimizers()
    config = {**_auto_length_config(tmp_path), "target_param_data_ratio": ratio, "sequence_len": T,
              "device_batch_size": B, "total_batch_size": B * T, "optimizer": OPT,
              "logging": {"log_every": 100}, "wandb": {"enabled": False}}
    config["checkpoint"]["enabled"] = True
    trainer = Trainer(system=system, optimizers=opts, dataloader=batches(), config=config,
                      rank=0, world_size=1, evaluators=[])
    trainer.train()
    assert trainer.max_steps == 10 and isinstance(trainer.max_steps, int) and _saved_steps(tmp_path) == [9]


def test_a_launch_stopped_by_num_iterations_saves_and_evaluates_its_last_step(tmp_path, cpu_cuda_stubs, capsys):
    """num_iterations stops a launch that many steps after where it started, inside a longer schedule (max_steps 10).
    Its stop step is the launch's last: saved whatever save_every says, evaluated whatever the cadence says. It used to
    stop with neither, losing the steps since the last regular save (2026-10-22). A resumed launch counts from its
    resume step."""
    from core.evaluation.evaluator import Evaluator
    from core.training.trainer import Trainer

    class Constant(Evaluator):
        interval_steps = 100

        def evaluate(self, model, autocast_ctx):
            return {"val/x": 0.0}

    def batches():
        g = torch.Generator().manual_seed(0)
        while True:
            idx = torch.randint(0, V, (B, T), generator=g)
            yield {"idx": idx, "targets": idx.roll(-1, dims=1)}

    def launch(**checkpoint):
        system, opts = _system_and_optimizers()
        config = {"max_steps": 10, "num_iterations": 4, "sequence_len": T, "device_batch_size": B,
                  "total_batch_size": B * T, "optimizer": OPT, "logging": {"log_every": 100},
                  "wandb": {"enabled": False},
                  "checkpoint": {"enabled": True, "save_dir": str(tmp_path), "save_every": 5, "keep_last_n": -1,
                                 **checkpoint}}
        Trainer(system=system, optimizers=opts, dataloader=batches(), config=config, rank=0, world_size=1,
                evaluators=[Constant()]).train()
        return [line.split(" |")[0] for line in capsys.readouterr().out.splitlines() if "| val/x" in line]

    assert launch() == ["Step 00000", "Step 00003"] and _saved_steps(tmp_path) == [0, 3]
    assert launch(resume_from=str(tmp_path / "step_000003")) == ["Step 00007"]
    assert _saved_steps(tmp_path) == [0, 3, 5, 7]
    assert launch(resume_from=str(tmp_path / "step_000007")) == ["Step 00009"]    # 4 more would pass max_steps
    assert _saved_steps(tmp_path) == [0, 3, 5, 7, 9]
