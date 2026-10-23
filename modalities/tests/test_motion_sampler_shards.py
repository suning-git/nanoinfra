"""The motion sources' samplers shard as the source is told, not as the process happens to be launched.

ResumableDistributedSampler takes its default rank and world size from the launcher's environment
(core.utils.process_identity). The motion sources set both explicitly (`_index_sampler`): the training source
with the process's own, the validation source with rank 0 of 1, so that every process holds the whole validation
set (nano_motion's SupervisedCEEvaluator freezes the same batches on every rank and evaluates its shard of them).
Pinned with a launcher environment of rank 1 of 2 and no process group.
"""

from modalities.motion.data.sources import _index_sampler


def _epoch(sampler, n):
    it = iter(sampler)
    return [next(it) for _ in range(n)]


def test_the_explicit_share_wins_over_the_launchers(monkeypatch):
    for name, value in (("RANK", 1), ("LOCAL_RANK", 1), ("WORLD_SIZE", 2)):
        monkeypatch.setenv(name, str(value))
    whole = _index_sampler(10, seed=0, rank=0, world_size=1)
    assert (whole.rank, whole.world_size) == (0, 1) and sorted(_epoch(whole, 10)) == list(range(10))
    mine, other = _index_sampler(10, seed=0, rank=1, world_size=2), _index_sampler(10, seed=0, rank=0, world_size=2)
    assert sorted(_epoch(mine, 5) + _epoch(other, 5)) == list(range(10))
