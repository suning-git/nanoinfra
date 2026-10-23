"""Small runtime utilities used by core: who this process is (process_identity, print0), the process-wide PyTorch
settings an entry applies (set_torch_global_defaults), and the base directory for artifacts.

Importing this module imports nothing beyond the standard library; set_torch_global_defaults and
ProcessIdentity.device import torch when they are called."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessIdentity:
    """Which process this is: what process_identity() answers."""
    distributed: bool    # launched by torchrun (RANK is set); a one-process torchrun is distributed too
    rank: int
    local_rank: int
    world_size: int

    @property
    def device(self):
        """The GPU this process uses: cuda:local_rank under torchrun; otherwise "cuda", i.e. whatever device is
        current when it is used (not a resolved index). A description only: it does not set the current device."""
        import torch

        return torch.device("cuda", self.local_rank) if self.distributed else torch.device("cuda")


_LAUNCHER_VARS = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


def process_identity() -> ProcessIdentity:
    """Which process this is, read from the launcher's environment as it is now — no CUDA, no process group — so
    it gives the same answer before torch.distributed is initialized as after, and is safe to call anywhere.

    torchrun sets RANK, LOCAL_RANK and WORLD_SIZE: then this is a distributed process (a one-process torchrun
    too: core builds a process group and places the model for it as for many). None of them set is one plain
    process: rank 0 of 1. Some but not all set is refused: a hand-made environment like RANK=1 alone would
    otherwise read as "rank 1 of 1" and shard data wrongly without an error.

    It answers WHO this process is, not whether there is a process group to talk to: code that runs a collective
    asks torch.distributed (dist.is_initialized()). A process group built by hand (mp.spawn with an explicit rank
    and world size) is invisible here unless the three variables are set the way torchrun sets them.
    """
    present = [name for name in _LAUNCHER_VARS if name in os.environ]
    if not present:
        return ProcessIdentity(distributed=False, rank=0, local_rank=0, world_size=1)
    if len(present) < len(_LAUNCHER_VARS):
        missing = [name for name in _LAUNCHER_VARS if name not in present]
        raise RuntimeError(f"partial launcher environment: {present} set but {missing} not "
                           "(torchrun sets all three; set none for a single process)")
    return ProcessIdentity(distributed=True, rank=int(os.environ["RANK"]),
                           local_rank=int(os.environ["LOCAL_RANK"]), world_size=int(os.environ["WORLD_SIZE"]))


def print0(*args: Any, **kwargs: Any) -> None:
    """Print only on rank 0."""
    if process_identity().rank == 0:
        print(*args, **kwargs)


def set_torch_global_defaults() -> None:
    """Set the process-wide PyTorch settings the training entries run with. An entry calls it first thing, before
    anything uses CUDA; core itself never does (an import or an assembly must not change process-wide settings).
    Two settings, nothing else (seeds and deterministic mode are each run's own):

    * The CUDA caching allocator uses expandable segments: a block of cached GPU memory can grow instead of a new
      block being reserved beside it, which helps against fragmentation and memory reserved beyond what is used;
      results are unchanged. Left exactly as it is if the user configured the allocator: PYTORCH_ALLOC_CONF or its
      older name PYTORCH_CUDA_ALLOC_CONF set, even to an empty value. It writes PYTORCH_ALLOC_CONF, the name torch
      documents as the primary one. The option is read when CUDA initializes (measured on torch 2.12), so a later
      setting would be ignored without a word, and that is refused here. Only this option: others are read at other times (the allocator
      backend when torch is imported), so set those yourself in the environment that starts the process.
    * Float32 matmuls may use TF32: torch.set_float32_matmul_precision("high"), what core set inside every
      build_system before 0.3.0 (PyTorch's own default is "highest"). An entry that wants another precision sets it
      after this call.
    """
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        loaded = sys.modules.get("torch")
        if loaded is not None and loaded.cuda.is_initialized():
            raise RuntimeError("set_torch_global_defaults: CUDA is already initialized, so the allocator would "
                               "silently ignore expandable segments; call it at the top of the entry, before "
                               "anything uses CUDA, or configure the allocator in the environment that starts the "
                               "process (PYTORCH_ALLOC_CONF)")
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    torch.set_float32_matmul_precision("high")


def get_base_dir() -> str:
    """Base directory for runtime artifacts. Defaults to ./outputs."""
    return str(Path(os.environ.get("NANOINFRA_BASE_DIR", "./outputs")).expanduser())


class DummyWandb:
    """No-op object with the subset of the wandb Run API used by Trainer."""

    def __getattr__(self, name: str):
        def noop(*args: Any, **kwargs: Any):
            return None
        return noop

    def log(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self, *args: Any, **kwargs: Any) -> None:
        return None
