"""
dataset.py — the data leg: memmap rows -> assembled batches, resumably.

Two objects, one job each.

`VideoRowDataset` is a plain map-style dataset over the fixed-stride cache built by
build_cache.py. `dataset[i]` is a memmap slice, so random access is a page fault
rather than a shard decompression — which is the entire reason the cache exists.

`VideoRowLoader` is what core's Trainer consumes. Its contract with the Trainer is
small and duck-typed (see core/training/trainer.py): be an INFINITE iterable of
batch dicts, and expose `set_state()` so a resume can restore the position. The
position itself comes from core's ResumableDistributedSampler, whose state is
{seed, epoch, index} — three integers with no rank and no world_size in them. That
is what makes two properties true at once:

    * resume is exact to the sample, on every rank, from rank 0's saved state alone
      (only rank 0 writes meta.json, so a per-rank state could never be restored);
    * a run checkpointed on 2 GPUs resumes on 1, or on 4.

The research loader this replaces stored its per-rank SHARD LIST in the checkpoint.
That is hardware-dependent state, so a resumed rank 1 read rank 0's list and tripped
an assertion; the workaround was to skip the position restore entirely under DDP.
The loader was the problem, not the parallelism.

No worker processes: a batch is a memmap slice plus numpy fancy-indexing into a
precomputed row template (~50us for 4 rows). Workers would add nondeterminism and
a second copy of the resume problem for no throughput.
"""

import json

import numpy as np
import torch

from core.data.dist_sampler import ResumableDistributedSampler


class VideoRowDataset:
    """Map-style access to one split of a fixed-stride cache.

    Returns raw cache rows — (codes, actions) as numpy — not model input. Turning
    those into token rows is RowLayout's job, and it happens per batch.
    """

    def __init__(self, cache_dir, split="train"):
        self.cache_dir = cache_dir
        self.split = split
        self.meta = json.loads((cache_dir / "meta.json").read_text())
        geom = self.meta["geometry"]
        n = self.meta["rows"][split]
        self.geometry = geom
        self.codes = np.memmap(cache_dir / f"{split}_codes.u16",
                               dtype=self.meta["code_dtype"], mode="r",
                               shape=(n, geom["code_len"]))
        self.actions = np.memmap(cache_dir / f"{split}_actions.u8",
                                 dtype=self.meta["action_dtype"], mode="r",
                                 shape=(n, geom["n_action_tokens"]))

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        return self.codes[i], self.actions[i]

    def take(self, n):
        """The first n rows as one (codes, actions) pair — for the frozen val set,
        which is evaluated as a fixed block rather than sampled."""
        n = min(n, len(self))
        return np.asarray(self.codes[:n]), np.asarray(self.actions[:n])

    def __repr__(self):
        return (f"VideoRowDataset({self.split}: {len(self)} rows x "
                f"{self.geometry['code_len']} codes, {self.cache_dir.name})")


class VideoRowLoader:
    """Infinite, resumable, rank-sharded batches of assembled token rows.

    Yields {"idx": [B, row_len] int64 on device, "state_dict": sampler state}.
    The Trainer stores the state that rode with the last batch of a step, so a
    checkpoint at step N resumes at exactly the row after step N's last row.
    """

    # A large odd multiplier and a Mersenne-prime modulus: a cheap, stable mix with
    # no dependence on PYTHONHASHSEED (python's hash() is salted per process, which
    # would make "the same rows" mean different noise on different ranks).
    _MIX, _MOD = 1000003, (1 << 61) - 1

    def __init__(self, dataset, rows, batch_size, seed=0, device="cuda"):
        self.dataset = dataset
        self.rows = rows
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.sampler = ResumableDistributedSampler(dataset, seed=seed)

    def state_dict(self):
        return {"sampler": self.sampler.state_dict()}

    def set_state(self, state):
        self.sampler.load_state_dict(state["sampler"])

    def _noise_seed(self, sel):
        """Seed the diffusion masks from WHICH ROWS are in the micro-batch, not from
        the step or the rank. Three consequences, all wanted:
          * resume replays the same noise for the same rows;
          * different ranks hold different rows, so their masks are independent;
          * a 2-GPU step and a 1-GPU 2-accumulation step over the SAME micro-batches
            draw the SAME noise — which is what makes the DDP equivalence gate tight
            enough to be worth running.
        """
        h = self.seed
        for i in sel:
            h = (h * self._MIX + int(i) + 1) % self._MOD
        return h

    def __iter__(self):
        it = iter(self.sampler)          # infinite: cycles epochs internally
        while True:
            sel = [next(it) for _ in range(self.batch_size)]
            codes = np.stack([self.dataset.codes[i] for i in sel])
            actions = np.stack([self.dataset.actions[i] for i in sel])
            idx = self.rows.assemble(codes, actions).to(self.device, non_blocking=True)
            yield {"idx": idx,
                   "rows": torch.tensor(sel, dtype=torch.long),
                   "noise_seed": self._noise_seed(sel),
                   "state_dict": self.state_dict()}

    def __repr__(self):
        return f"VideoRowLoader(batch={self.batch_size}, {self.sampler!r})"
