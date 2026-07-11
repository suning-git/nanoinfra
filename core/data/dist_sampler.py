"""
ResumableDistributedSampler: distributed sharding + checkpoint resume.

A general-purpose sampler that combines DistributedSampler's data sharding
with deterministic checkpoint resume. Works with any map-style Dataset.

Key properties:
- State is 3 integers: (seed, epoch, index) -- hardware-independent
- Permutation is deterministic from (seed + epoch) -- no RNG state saving
- Resume with different world_size is naturally supported
- Infinite iteration for step-based training loops

Usage:
    sampler = ResumableDistributedSampler(dataset, seed=42)
    loader = StatefulDataLoader(dataset, sampler=sampler, batch_size=64)

    # Checkpoint
    state = sampler.state_dict()   # {"seed": 42, "epoch": 3, "index": 4096}

    # Resume (works even with different GPU count)
    sampler.load_state_dict(state)
"""

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Dict, Any, Iterator


class ResumableDistributedSampler(Sampler[int]):
    """
    Distributed sampler with checkpoint resume support.

    Combines DistributedSampler's sharding with deterministic resume.
    State is hardware-independent: (seed, epoch, index) -- no rank/world_size
    stored, enabling resume with different GPU counts.

    Args:
        dataset: Map-style dataset (must implement __len__)
        seed: Random seed for reproducibility (default: 0)

    Sharding mechanism:
        Each epoch generates a deterministic auxiliary permutation from (seed + epoch).
        The permutation is consumed in chunks of world_size: at each step, rank r
        yields perm[index + r], then index advances by world_size.

    State dict:
        {"seed": int, "epoch": int, "index": int}
        - seed: random seed (verified on load)
        - epoch: current epoch number
        - index: auxiliary position in the permutation (not shard-local)
    """

    def __init__(self, dataset, seed: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.index = 0  # Global position in permutation

        # Auto-detect distributed settings
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def _generate_permutation(self, epoch: int) -> list:
        """Generate deterministic permutation for an epoch.

        Same (seed, epoch) always produces the same permutation on all ranks.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        return torch.randperm(len(self.dataset), generator=g).tolist()

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices, sharded for this rank, cycling through epochs."""
        while True:
            perm = self._generate_permutation(self.epoch)
            n = len(perm)

            # Truncate remaining samples to be divisible by world_size
            remaining = n - self.index
            usable = remaining - (remaining % self.world_size)
            end = self.index + usable

            # Yield this rank's samples from the auxiliary permutation
            # NOTE: Update self.index BEFORE yield, because in a generator
            # code after yield runs on the NEXT next() call, which would
            # make state_dict() return a stale index.
            idx = self.index
            while idx < end:
                sample = perm[idx + self.rank]
                idx += self.world_size
                self.index = idx
                yield sample

            # Next epoch
            self.epoch += 1
            self.index = 0

    def __len__(self) -> int:
        """Number of samples remaining for this rank in current epoch."""
        n = len(self.dataset)
        remaining = n - self.index
        usable = remaining - (remaining % self.world_size)
        return usable // self.world_size

    def __repr__(self) -> str:
        n = len(self.dataset)
        pct = 100 * self.index / n if n > 0 else 0
        return f"Sampler(seed={self.seed}, epoch={self.epoch}, index={self.index}/{n} ({pct:.1f}%), rank={self.rank}/{self.world_size})"

    def state_dict(self) -> Dict[str, Any]:
        """Get sampler state for checkpointing.

        Returns 3 integers -- no hardware info (rank/world_size) stored.
        """
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "index": self.index,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore from checkpoint.

        Validates seed match. Supports resume with different world_size --
        remaining data is re-sharded among the current number of GPUs.
        """
        if state_dict["seed"] != self.seed:
            raise ValueError(
                f"Seed mismatch: sampler has seed={self.seed}, "
                f"checkpoint has seed={state_dict['seed']}"
            )
        self.epoch = state_dict["epoch"]
        self.index = state_dict["index"]
