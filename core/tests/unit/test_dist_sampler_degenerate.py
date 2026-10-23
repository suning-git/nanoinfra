"""ResumableDistributedSampler on a dataset no larger than the number of ranks.

Until 2026-10-22 each epoch handed out len(dataset) rounded down to a multiple of world_size; with fewer samples than
ranks that was zero, every epoch yielded nothing and iteration never returned (a training job hanging at its first
batch), so the sampler refused such a dataset. The epochs now run end to end as one sequence and nothing is dropped,
so a rank simply reads into the next epoch. Pinned here on CPU (world size set on the instance, no process group):
  * fewer samples than ranks works: every rank gets samples, and each epoch's samples go out exactly once;
  * an empty dataset raises at the first next() instead of hanging;
  * as many samples as ranks works, one sample per rank per epoch, epoch after epoch;
  * a restored position beyond the dataset (it shrank since the checkpoint) raises instead of silently skipping.
"""

import signal

import pytest

from core.data.dist_sampler import ResumableDistributedSampler


class _Range:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _sampler(n, world_size, rank=0):
    s = ResumableDistributedSampler(_Range(n), seed=3)
    s.world_size, s.rank = world_size, rank
    return s


def _first(sampler, timeout=20):
    """next(iter(sampler)) under an alarm, so that code without the check fails this test instead of hanging the whole
    suite (no pytest-timeout here): returns the item, or the exception it raised."""
    def expired(signum, frame):
        raise TimeoutError(f"next() did not return within {timeout} s: the sampler loops forever")
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(timeout)
    try:
        return next(iter(sampler))
    except TimeoutError as e:
        pytest.fail(str(e))
    except Exception as e:              # handed back to the test
        return e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.mark.parametrize("n,world_size", [(1, 2), (3, 4), (2, 5)])
def test_fewer_samples_than_ranks_works_and_hands_each_epoch_out_once(n, world_size):
    its = [iter(_sampler(n, world_size, rank=r)) for r in range(world_size)]
    rounds = 3 * n                                              # 3 * world_size epochs' worth of positions
    got = [next(it) for _ in range(rounds) for it in its]       # global order: position by position
    for e in range(rounds * world_size // n):
        assert sorted(got[e * n:(e + 1) * n]) == list(range(n)), (e, got)


@pytest.mark.parametrize("world_size", [1, 2])
def test_an_empty_dataset_raises_instead_of_hanging(world_size):
    assert isinstance(_first(_sampler(0, world_size)), Exception)


def test_as_many_samples_as_ranks_cycles():
    it0, it1 = iter(_sampler(2, 2, rank=0)), iter(_sampler(2, 2, rank=1))
    got = [(next(it0), next(it1)) for _ in range(4)]          # four epochs, one sample per rank each
    assert all(sorted(pair) == [0, 1] for pair in got)


def test_restored_position_beyond_the_dataset_raises():
    s = _sampler(10, 1)
    with pytest.raises(ValueError, match="beyond"):
        s.load_state_dict({"seed": 3, "epoch": 0, "index": 11})
    s.load_state_dict({"seed": 3, "epoch": 0, "index": 10})    # end of an epoch is a valid position
    assert next(iter(s)) in range(10)
