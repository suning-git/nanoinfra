"""
dataset.py — the data leg: the fixed-stride cache, as a source for core's data pipeline.

Two objects, one job each.

`VideoRowDataset` is a plain map-style dataset over the fixed-stride cache built by
build_cache.py. `dataset[i]` is a memmap slice, so random access is a page fault
rather than a shard decompression — which is the entire reason the cache exists.

`ClipSource` is what core's data pipeline reads (core/data/): a
DataSource yielding one raw clip at a time — the clip's assembled row and which clip
it is — endlessly, in the order of core's ResumableDistributedSampler. What the clip
becomes for training (two copies, noise, targets) is not decided here: the loader hands
every clip to the objective's preparation (block_diffusion.TwoStreamPreparation)
before it is batched.

The sampler's position is {seed, epoch, index}, global, with no rank and no world size
in it. At every save core gathers every rank's state and the source merges them
(merge_rank_states): the furthest rank's position plus the clips behind it that no
rank has trained yet. So a run checkpointed on 2 GPUs resumes on 1, or on 4, and every
clip is trained once, none skipped or repeated. (The research loader this exemplar
started from stored its per-rank SHARD LIST in the checkpoint — hardware-dependent
state, which a resumed rank 1 could not read, so under DDP the position restore was
skipped entirely. The loader was the problem, not the parallelism.)

No worker processes: a clip is a memmap slice plus numpy fancy-indexing into a
precomputed row template. Workers would add nondeterminism and a second copy of the
resume problem for no throughput.
"""

import json

import numpy as np

from core.data.data_source import DataSource
from core.data.dist_sampler import ResumableDistributedSampler

from exemplars.nano_world_model import spec


class VideoRowDataset:
    """Map-style access to one split of a fixed-stride cache.

    Returns raw cache rows — (codes, actions) as numpy — not model input. Turning
    those into token rows is RowLayout's job.
    """

    def __init__(self, cache_dir, split="train"):
        self.cache_dir = cache_dir
        self.split = split
        self.meta = json.loads((cache_dir / "meta.json").read_text())
        contract = spec.meta_contract(self.meta)
        n = self.meta["rows"][split]
        self.contract = contract
        self.codes = np.memmap(cache_dir / f"{split}_codes.u16",
                               dtype=self.meta["code_dtype"], mode="r",
                               shape=(n, contract["code_len"]))
        self.actions = np.memmap(cache_dir / f"{split}_actions.u8",
                                 dtype=self.meta["action_dtype"], mode="r",
                                 shape=(n, contract["n_action_tokens"]))

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        return self.codes[i], self.actions[i]

    def spread(self, n):
        """n row indices spread evenly over the split (all of them when it holds fewer) — for the frozen val set.
        Not the first n: the split is the encoded shards laid end to end, and its head can be a single recording."""
        n = min(n, len(self))
        return (np.arange(n) * len(self)) // n

    def __repr__(self):
        return (f"VideoRowDataset({self.split}: {len(self)} rows x "
                f"{self.contract['code_len']} codes, {self.cache_dir.name})")


class ClipSource(DataSource):
    """One split of a cache as an endless stream of raw clips {tokens [row_len] int64 (CPU), meta {row, epoch}}.

    `tokens` is the clip's assembled row (RowLayout.assemble); meta says which clip it is — its row in the split and
    the epoch it is read in — which is what the preparation seeds the clip's noise from, so the same clip in the same
    epoch is noised the same way on any rank and after a resume, and differently in the next epoch."""

    def __init__(self, dataset, rows, seed=0):
        self.dataset, self.rows = dataset, rows
        self.sampler = ResumableDistributedSampler(dataset, seed=seed)

    def __iter__(self):
        s, n = self.sampler, len(self.dataset)
        it = iter(s)                                  # endless: the sampler cycles epochs
        while True:
            # The epoch of the clip the pull returns. A leftover clip (handed out first after a resume) carries its own:
            # it is the one the pull takes off the front of the leftover. Otherwise this rank's next clip is at global
            # position epoch * n + index + rank, read before the pull (the sampler moves its position as it yields).
            n_left, lead = len(s.leftover), (s.leftover[0][0] if s.leftover else None)
            main = (s.epoch * n + s.index + s.rank) // n
            i = next(it)
            epoch = lead if len(s.leftover) < n_left else main
            row = self.rows.assemble(self.dataset.codes[i][None], self.dataset.actions[i][None])[0]
            yield {"tokens": row, "meta": {"row": int(i), "epoch": int(epoch)}}

    def get_state(self):
        return {"sampler": self.sampler.state_dict()}

    def set_state(self, state):
        self.sampler.load_state_dict(state["sampler"])

    def merge_rank_states(self, states):
        # the sampler merges: the furthest rank's position, plus the clips behind it that some rank has not trained
        return {"sampler": self.sampler.merge_rank_states([s["sampler"] for s in states])}

    def __repr__(self):
        return f"ClipSource({self.dataset.split} of {self.dataset.cache_dir.name}, {self.sampler!r})"
