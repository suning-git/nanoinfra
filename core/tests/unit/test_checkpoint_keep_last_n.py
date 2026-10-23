"""checkpoint.keep_last_n: a positive integer, or -1 to keep all. 0 used to delete the checkpoint just written at every
save (the cleanup keeps the newest 0); a negative other than -1 deleted the oldest few. Both are refused before
anything is deleted.
"""
import pytest

from core.model.checkpoint_manager import cleanup_old_checkpoints


def _dirs(tmp_path, steps):
    for s in steps:
        (tmp_path / f"step_{s:06d}").mkdir()


def _left(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir())


@pytest.mark.parametrize("keep", [0, -2])
def test_a_keep_count_with_no_meaning_is_refused_and_nothing_is_deleted(tmp_path, keep):
    _dirs(tmp_path, [0, 5, 10])
    with pytest.raises(ValueError, match="keep_last_n must be a positive integer, or -1"):
        cleanup_old_checkpoints(str(tmp_path), keep, current_step=10)
    assert _left(tmp_path) == ["step_000000", "step_000005", "step_000010"]


def test_minus_one_keeps_all_and_n_keeps_the_newest_n(tmp_path):
    _dirs(tmp_path, [0, 5, 10])
    cleanup_old_checkpoints(str(tmp_path), -1, current_step=10)
    assert len(_left(tmp_path)) == 3
    cleanup_old_checkpoints(str(tmp_path), 2, current_step=10)
    assert _left(tmp_path) == ["step_000005", "step_000010"]
