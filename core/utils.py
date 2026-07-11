"""Small runtime utilities used by core."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def get_dist_info() -> tuple[int, int, int, int]:
    """Return (rank, local_rank, world_size, device_count)."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    try:
        import torch

        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0
    return rank, local_rank, world_size, device_count


def print0(*args: Any, **kwargs: Any) -> None:
    """Print only on rank 0."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


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
