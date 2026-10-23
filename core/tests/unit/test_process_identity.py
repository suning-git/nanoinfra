"""core.utils.process_identity: which process this is, from the launcher's environment alone.

Pinned on CPU, with the three torchrun variables set by hand:
  * none of them is one plain process (rank 0 of 1, not distributed); all three is a distributed process, a
    one-process torchrun included (core still builds a process group and places the model for it);
  * some but not all is refused — RANK=1 alone would otherwise read as "rank 1 of 1" and shard data wrongly;
  * it reads the environment as it is now (no cache), without importing torch (test_process_settings checks that);
  * print0 prints on rank 0 only;
  * ResumableDistributedSampler takes its default share from it, with no process group: a sampler built before
    build_system gets its rank's share, not rank 0's of 1.
"""

import pytest

from core.data.dist_sampler import ResumableDistributedSampler
from core.utils import ProcessIdentity, print0, process_identity

VARS = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


@pytest.fixture
def launcher_env(monkeypatch):
    """Clears the three variables, then sets the ones given (as strings)."""
    def set_env(**values):
        for name in VARS:
            monkeypatch.delenv(name, raising=False)
        for name, value in values.items():
            monkeypatch.setenv(name, str(value))
    return set_env


def test_no_launcher_variables_is_one_plain_process(launcher_env):
    launcher_env()
    assert process_identity() == ProcessIdentity(distributed=False, rank=0, local_rank=0, world_size=1)


def test_torchrun_variables_are_read(launcher_env):
    launcher_env(RANK=5, LOCAL_RANK=1, WORLD_SIZE=8)
    assert process_identity() == ProcessIdentity(distributed=True, rank=5, local_rank=1, world_size=8)


def test_one_process_torchrun_is_distributed(launcher_env):
    launcher_env(RANK=0, LOCAL_RANK=0, WORLD_SIZE=1)
    identity = process_identity()
    assert identity.distributed and identity.world_size == 1


@pytest.mark.parametrize("present", [("RANK",), ("LOCAL_RANK",), ("WORLD_SIZE",), ("RANK", "WORLD_SIZE"),
                                     ("RANK", "LOCAL_RANK"), ("LOCAL_RANK", "WORLD_SIZE")])
def test_partial_launcher_environment_is_refused(launcher_env, present):
    launcher_env(**{name: 1 for name in present})
    with pytest.raises(RuntimeError, match="partial launcher environment"):
        process_identity()
    with pytest.raises(RuntimeError, match="partial launcher environment"):
        print0("never printed")


def test_reads_the_environment_as_it_is_now(launcher_env):
    launcher_env(RANK=1, LOCAL_RANK=1, WORLD_SIZE=2)
    assert process_identity().rank == 1
    launcher_env()
    assert process_identity().rank == 0


def test_device_describes_the_process_gpu(launcher_env):
    launcher_env(RANK=3, LOCAL_RANK=1, WORLD_SIZE=4)
    assert str(process_identity().device) == "cuda:1"
    launcher_env()
    device = process_identity().device
    assert device.type == "cuda" and device.index is None       # whatever device is current when used


def test_print0_prints_on_rank_0_only(launcher_env, capsys):
    launcher_env(RANK=0, LOCAL_RANK=0, WORLD_SIZE=2)
    print0("from rank 0")
    launcher_env(RANK=1, LOCAL_RANK=1, WORLD_SIZE=2)
    print0("from rank 1")
    launcher_env()
    print0("from a plain process")
    assert capsys.readouterr().out.splitlines() == ["from rank 0", "from a plain process"]


class _Range:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def test_sampler_takes_its_share_from_the_launcher_without_a_process_group(launcher_env):
    import torch.distributed as dist
    assert not dist.is_initialized()
    launcher_env(RANK=1, LOCAL_RANK=1, WORLD_SIZE=2)
    s1 = ResumableDistributedSampler(_Range(10), seed=3)
    launcher_env(RANK=0, LOCAL_RANK=0, WORLD_SIZE=2)
    s0 = ResumableDistributedSampler(_Range(10), seed=3)
    assert (s0.rank, s0.world_size, s1.rank, s1.world_size) == (0, 2, 1, 2)
    i0, i1 = iter(s0), iter(s1)
    first = [next(i0) for _ in range(5)], [next(i1) for _ in range(5)]
    assert sorted(first[0] + first[1]) == list(range(10))        # the two shares make one epoch, no overlap
