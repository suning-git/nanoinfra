"""
ResumableDistributedSampler: distributed sharding + checkpoint resume.

A general-purpose sampler for any map-style Dataset: an endless stream of indices, sharded over the ranks, with a
small hardware-independent state that a run saved on one number of GPUs resumes from on another.

Key properties:
- One endless sequence: the epochs' permutations laid end to end; global position g is item perm_e[p] with
  (e, p) = divmod(g, len(dataset)). Every item of every epoch is handed out exactly once over the ranks; nothing is
  dropped at an epoch's end, so a few ranks may already read the next epoch while others finish this one.
- Permutation is deterministic from (seed + epoch) -- no RNG state saving.
- State is (seed, epoch, index) = global position epoch * n + index, plus `leftover`: items before that position
  that no rank has trained yet (see merge_rank_states). Hardware-independent.

Usage:
    sampler = ResumableDistributedSampler(dataset, seed=42)
    loader = StatefulDataLoader(dataset, sampler=sampler, batch_size=64)

    # Checkpoint: each rank's state, merged into the run's (one rank: just its own)
    run_state = sampler.merge_rank_states(states_of_every_rank)

    # Resume (works with a different GPU count)
    sampler.load_state_dict(run_state)
"""

from collections import deque
from typing import Any, Dict, Iterator, List

import torch
from torch.utils.data import Sampler

from core.utils import process_identity


class ResumableDistributedSampler(Sampler[int]):
    """
    Distributed sampler with checkpoint resume support.

    Args:
        dataset: Map-style dataset (must implement __len__)
        seed: Random seed for reproducibility (default: 0)

    Sharding: rank r of N at global position g yields the item at g + r, then g advances by N. The ranks of a run
    start from the same position and each advances N per item it takes, so their positions agree only while they have
    taken the same number (with packing they drift apart, which merge_rank_states accounts for). Before that stream,
    a rank yields its share of the leftover items the run's state carries (rank r of N takes leftover[r::N]).

    State dict, as state_dict() gives it (this rank's):
        {"seed", "epoch", "index", "leftover": [[epoch, item], ...], "shard": [rank, world_size]}
        - epoch, index: this rank's global position epoch * n + index (0 <= index < n)
        - leftover: this rank's leftover items not yet yielded, each [the epoch it belongs to, the dataset index]
        - shard: marks the state as one rank's; merge_rank_states turns every rank's into the run's, which has no
          shard and the whole leftover list
    """

    def __init__(self, dataset, seed: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.index = 0  # position in this epoch's permutation; global position = epoch * n + index
        self.leftover = deque()                  # this rank's leftover items, [epoch, item], yielded first
        self._perms = {}

        # Which share this rank reads: by default the launcher's process identity, which is known before the
        # process group exists (a source may be built first). A source that shards otherwise — every rank reading
        # the whole validation set — sets rank and world_size after construction, before load_state_dict.
        identity = process_identity()
        self.rank, self.world_size = identity.rank, identity.world_size

    def _perm(self, epoch: int) -> list:
        """The epoch's permutation, deterministic from (seed, epoch): the same on all ranks. Two are kept: near an
        epoch's end the ranks read from it and the next."""
        if epoch not in self._perms:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            self._perms = {e: p for e, p in self._perms.items() if e >= epoch - 1}
            self._perms[epoch] = torch.randperm(len(self.dataset), generator=g).tolist()
        return self._perms[epoch]

    def __iter__(self) -> Iterator[int]:
        """This rank's leftover items, then its share of the endless sequence."""
        n = len(self.dataset)
        # The state moves BEFORE each yield: code after a yield runs at the NEXT next(), so state_dict() taken right
        # after a pull would otherwise be stale.
        while self.leftover:
            _, item = self.leftover.popleft()
            yield item
        while True:
            e, p = divmod(self.epoch * n + self.index + self.rank, n)
            item = self._perm(e)[p]
            self.epoch, self.index = divmod(self.epoch * n + self.index + self.world_size, n)
            yield item

    def __repr__(self) -> str:
        n = len(self.dataset)
        pct = 100 * self.index / n if n > 0 else 0
        left = f", leftover={len(self.leftover)}" if self.leftover else ""
        return (f"Sampler(seed={self.seed}, epoch={self.epoch}, index={self.index}/{n} ({pct:.1f}%){left}, "
                f"rank={self.rank}/{self.world_size})")

    def state_dict(self) -> Dict[str, Any]:
        """This rank's state: its position, its leftover items not yet yielded, and which rank of how many it is
        (the run's state is merge_rank_states of every rank's)."""
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "index": self.index,
            "leftover": [list(x) for x in self.leftover],
            "shard": [self.rank, self.world_size],
        }

    def merge_rank_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The run's state from every rank's (each from state_dict(), after its last DELIVERED item): no shard, the
        whole leftover list. Pure: reads nothing but the states and this sampler's dataset length and seed.

        The front is the furthest rank's position: every item from it on is untrained on every rank. A rank behind it
        has not yet read its items between its own position and the front, g + r for its positions g up to the
        front; they join the leftover, with the leftover items the ranks have not yet yielded. Every rank has the same
        base position after a resume and advances by N per item, so a rank's distance to the front is a multiple of N.
        """
        n = len(self.dataset)
        world_size = len(states)
        if sorted(tuple(s["shard"]) for s in states) != [(r, world_size) for r in range(world_size)]:
            raise ValueError(f"merge_rank_states needs the state of every rank, ranks 0..{world_size - 1} of "
                             f"{world_size} once each (got shards {[s.get('shard') for s in states]}): a missing "
                             f"rank's items would be lost")
        positions = {s["shard"][0]: s["epoch"] * n + s["index"] for s in states}
        front = max(positions.values())
        leftover = [list(x) for s in sorted(states, key=lambda s: s["shard"][0]) for x in s["leftover"]]
        for rank, g in sorted(positions.items()):
            if (front - g) % world_size:
                raise ValueError(f"rank {rank} is at position {g}, the front at {front}: not a multiple of "
                                 f"{world_size} apart, so the states do not come from one resume of this run")
            for h in range(g + rank, front + rank, world_size):
                e, p = divmod(h, n)
                leftover.append([e, self._perm(e)[p]])
        epoch, index = divmod(front, n)
        return {"seed": states[0]["seed"], "epoch": epoch, "index": index, "leftover": leftover}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore from the run's state (merge_rank_states), under the current rank and world_size, which must be
        set first: every rank takes the position, the leftover items are split, rank r of N taking leftover[r::N].
        Seed must match. A state without leftover (written before 2026-10-22, or by a loader that never merges) has
        none.

        One rank's state (state_dict(), it carries `shard`): a one-rank run's is complete and taken. A multi-rank
        one with leftover items is refused: they are only that rank's share, and splitting them again would lose
        the others silently. A multi-rank one without leftover is taken, for loaders that save rank 0's state and
        never merge (nano_world_model's loader before 2026-10): exact only when every rank was at that position with
        nothing left over, e.g. every rank taking one batch of rows a step, as before 2026-10-22.
        """
        if state_dict["seed"] != self.seed:
            raise ValueError(
                f"Seed mismatch: sampler has seed={self.seed}, "
                f"checkpoint has seed={state_dict['seed']}"
            )
        n = len(self.dataset)
        if not 0 <= state_dict["index"] <= n:
            raise ValueError(
                f"Sampler position {state_dict['index']} is beyond this dataset's {n} samples: "
                f"the dataset is not the one the checkpoint was taken on")
        leftover = state_dict.get("leftover", [])
        shard = state_dict.get("shard")
        if shard is not None and shard[1] > 1 and leftover:
            raise ValueError(
                f"this is rank {shard[0]}'s state of a {shard[1]}-rank run, and it holds leftover items: only that "
                f"rank's share of them. Restore the run's state, which merge_rank_states makes from every rank's "
                f"(MixedDataLoader.merge_rank_snapshots; core's checkpoint saving does it)")
        # an index of n (a state written at an epoch's end by the sampler before 2026-10-22) is the next epoch's start
        self.epoch, self.index = divmod(state_dict["epoch"] * n + state_dict["index"], n)
        self.leftover = deque(list(x) for x in leftover[self.rank::self.world_size])
