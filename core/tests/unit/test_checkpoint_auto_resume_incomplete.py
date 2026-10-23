"""`resume_from: auto` must survive a save that was interrupted.

Why this file exists. save_checkpoint_dcp writes every rank's DCP shards and then, from rank 0, meta.json. Every
load opens meta.json first. A run preempted or killed mid-save therefore leaves a step_* directory without
meta.json. find_latest_checkpoint used to take the highest step_* name regardless, so the relaunch, e.g. a requeued
preempt-queue job with resume_from=auto, picked that directory and died with FileNotFoundError on meta.json. The
directory had to be removed by hand.

Pinned here, with a real DCP save on CPU (no mocks):
  * an incomplete newest directory is skipped, with a warning, and left in place; the newest complete one is used,
    by find_latest_checkpoint and by resolve_checkpoint_path (auto, and a parent directory given explicitly);
  * step_* directories none of which is complete raise. "No checkpoint, start fresh" would silently restart a run
    that had progressed;
  * no step_* directory at all still means a fresh start.
"""

import pytest
import torch

from core.model import checkpoint_manager as cm
from core.model.gpt import GPTConfig
from core.model.system import LMSystem


class _Trunk(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w = torch.nn.Parameter(torch.zeros(4))


TINY = GPTConfig(sequence_len=64, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, n_token_types=3)


def _complete(dirpath, step):
    system = LMSystem(_Trunk(TINY), torch.nn.Linear(4, 4))
    opt = torch.optim.AdamW(system.parameters(), lr=1e-3)
    cm.save_checkpoint_dcp(str(dirpath / f"step_{step:06d}"), system, [opt], meta_data={"step": step})


def _interrupted(dirpath, step):
    d = dirpath / f"step_{step:06d}"
    d.mkdir()
    (d / "__0_0.distcp").write_bytes(b"partial shard")      # shards begun, meta.json never written
    return d


def test_an_interrupted_newest_save_is_skipped_and_kept(tmp_path, capsys):
    _complete(tmp_path, 5)
    bad = _interrupted(tmp_path, 10)
    assert cm.find_latest_checkpoint(str(tmp_path)) == str(tmp_path / "step_000005")
    assert "incomplete checkpoint" in capsys.readouterr().out
    assert bad.exists() and (bad / "__0_0.distcp").exists()          # never deleted
    assert cm.resolve_checkpoint_path("auto", save_dir=str(tmp_path)) == str(tmp_path / "step_000005")
    assert cm.resolve_checkpoint_path(str(tmp_path)) == str(tmp_path / "step_000005")
    assert cm.load_metadata(cm.resolve_checkpoint_path("auto", save_dir=str(tmp_path)))["step"] == 5


def test_only_interrupted_saves_raise_instead_of_starting_fresh(tmp_path):
    _interrupted(tmp_path, 3)
    with pytest.raises(FileNotFoundError, match="none is complete"):
        cm.resolve_checkpoint_path("auto", save_dir=str(tmp_path))


def test_no_step_directory_is_still_a_fresh_start(tmp_path):
    (tmp_path / "notes").mkdir()
    assert cm.find_latest_checkpoint(str(tmp_path)) is None
    assert cm.resolve_checkpoint_path("auto", save_dir=str(tmp_path)) is None


def test_complete_checkpoints_pick_the_highest_step(tmp_path):
    _complete(tmp_path, 4)
    _complete(tmp_path, 12)
    assert cm.find_latest_checkpoint(str(tmp_path)) == str(tmp_path / "step_000012")
