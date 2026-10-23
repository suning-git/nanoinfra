"""core/training/precision.py: the one training execution convention, and what the Trainer records of it.

Pinned on CPU:
  * the convention's pieces (bf16 autocast on CUDA, FSDP gathering and reducing in bf16), each Trainer getting an
    autocast context of its own;
  * a checkpoint the Trainer saves records the convention's name and the float32 matmul precision setting (the
    entry's, read at the save, not assumed), whether the entry set it through torch.set_float32_matmul_precision or
    through PyTorch's per-backend setting (under which the legacy getter raises).
"""

import json

import pytest
import torch
import torch.nn.functional as F

from core.model.gpt import GPTConfig
from core.model.system import LMSystem
from core.training import precision
from core.training.optim import build_optimizers

V, D, T, B = 16, 4, 8, 2
TINY = GPTConfig(sequence_len=T, vocab_size=V, n_layer=1, n_head=1, n_kv_head=1, n_embd=D, n_token_types=1)
OPT = {"type": "adamw", "lr_max": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "embedding_lr": 0.2,
       "unembedding_lr": 0.004, "max_grad_norm": 1.0,
       "scheduler": {"type": "linear", "warmup_steps": 0, "warmdown_ratio": 0.2}}


def test_the_convention():
    a, b = precision.training_autocast(), precision.training_autocast()
    assert a is not b
    assert (a.device, a.fast_dtype) == ("cuda", torch.bfloat16)
    mp = precision.fsdp_mixed_precision()
    assert (mp.param_dtype, mp.reduce_dtype) == (torch.bfloat16, torch.bfloat16)


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


def _train_and_read_meta(tmp_path):
    """The real Trainer.train() on the CPU for 3 steps, saving at the end (the caller asks for cpu_cuda_stubs).
    Returns the saved trainer_state."""
    from core.training.trainer import Trainer

    def batches():
        g = torch.Generator().manual_seed(0)
        while True:
            idx = torch.randint(0, V, (B, T), generator=g)
            yield {"idx": idx, "targets": idx.roll(-1, dims=1)}

    torch.manual_seed(0)
    system = LMSystem(_Trunk(), _Head(), prepare_trunk_inputs_fn=lambda batch: (batch["idx"], {}))
    config = {"max_steps": 3, "sequence_len": T, "device_batch_size": B, "total_batch_size": B * T,
              "optimizer": OPT, "logging": {"log_every": 100}, "wandb": {"enabled": False},
              "checkpoint": {"enabled": True, "save_dir": str(tmp_path), "save_every": -1, "keep_last_n": 2}}
    Trainer(system=system, optimizers=build_optimizers(system, OPT), dataloader=batches(), config=config,
            rank=0, world_size=1, evaluators=[]).train()
    return json.loads((tmp_path / "step_000002" / "meta.json").read_text())["trainer_state"]


@pytest.fixture
def matmul_precision_restored():
    # restored through the legacy setter, which writes the legacy and the per-backend state alike: restoring one
    # of them alone can leave the two inconsistent, and the legacy getter then raises for every later test
    before = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(before)


def test_a_trainer_checkpoint_records_how_the_run_computed(tmp_path, cpu_cuda_stubs, matmul_precision_restored):
    torch.set_float32_matmul_precision("highest")           # an entry's choice, whatever it is
    state = _train_and_read_meta(tmp_path)
    assert state["training_precision"] == precision.NAME == "bf16_autocast_v1"
    assert state["cuda_matmul_fp32_precision"] == "ieee"


def test_the_per_backend_setting_is_recorded_not_refused(tmp_path, cpu_cuda_stubs, matmul_precision_restored):
    torch.backends.cuda.matmul.fp32_precision = "tf32"      # PyTorch's newer API; the legacy getter raises after it
    assert _train_and_read_meta(tmp_path)["cuda_matmul_fp32_precision"] == "tf32"
