"""A resume that cannot restore the data position says so.

Model, optimizer and step resumed while the data silently started over whenever the checkpoint carried no
dataloader_state or the loader had no set_state. Real DCP save and resume on the CPU.
"""

import torch

from core.model import checkpoint_manager as cm
from core.model.gpt import GPTConfig
from core.model.system import LMSystem

TINY = GPTConfig(sequence_len=8, vocab_size=16, n_layer=1, n_head=1, n_kv_head=1, n_embd=4, n_token_types=1)


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TINY
        self.w = torch.nn.Parameter(torch.ones(4))


class _Loader:
    def __init__(self):
        self.restored = None

    def set_state(self, state):
        self.restored = state


def _resume(tmp_path, dataloader_state, loader):
    system = LMSystem(_Trunk(), torch.nn.Linear(4, 4))
    opt = torch.optim.AdamW(system.parameters(), lr=1e-3)
    cm.save_checkpoint_dcp(str(tmp_path / "ckpt"), system, [opt], meta_data={"step": 3},
                           dataloader_state=dataloader_state)
    fresh = LMSystem(_Trunk(), torch.nn.Linear(4, 4))
    return cm.load_checkpoint_if_needed({"resume_from": str(tmp_path / "ckpt")}, fresh,
                                        [torch.optim.AdamW(fresh.parameters(), lr=1e-3)], loader, 0, 1)


def test_no_saved_data_position_warns(tmp_path, capsys):
    assert _resume(tmp_path, None, _Loader())[0] == 4
    assert "WARNING: the data starts over" in capsys.readouterr().out


def test_a_loader_that_cannot_restore_warns(tmp_path, capsys):
    _resume(tmp_path, {"epoch": 1}, object())
    assert "has no set_state" in capsys.readouterr().out


def test_a_restored_position_does_not_warn(tmp_path, capsys):
    loader = _Loader()
    _resume(tmp_path, {"epoch": 1}, loader)
    assert loader.restored == {"epoch": 1} and "WARNING" not in capsys.readouterr().out
