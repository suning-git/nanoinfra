"""Two loaders, one intent each: `load_checkpoint_dcp` = weights + optimizer (a resume),
`load_model_only` = weights (an init). No switch between them.

Why this file exists (queue #10, 2026-09-10). `load_checkpoint_dcp` used to carry a
`load_optimizer` flag whose default, False, selected a path that had never run: it
handed `optimizers=None` to torch's get_state_dict, which rejects None. The first
caller to try it (nano_t2v serve_compare, 2026-07-25) crashed and wrote its own
weights-only loader instead. The fix is not to repair that branch but to remove the
switch: the weights-only loader already existed under its own name.

Real DCP save/load on CPU in a single process — no mocks — so the tests pin what
the loaders actually do to the tensors and the optimizer.
"""

import inspect

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

    def forward(self, x):
        return x * self.w


TINY = GPTConfig(sequence_len=64, vocab_size=128, n_layer=2,
                 n_head=2, n_kv_head=2, n_embd=32, n_token_types=3)


def _system():
    return LMSystem(_Trunk(TINY), torch.nn.Linear(4, 4))


def _group(opt):
    return {k: v for k, v in opt.param_groups[0].items() if k != "params"}


@pytest.fixture
def saved(tmp_path):
    """A checkpoint written by the real saver: non-trivial weights, three optimizer
    steps taken (so moments exist and the per-parameter step count is 3, which a
    freshly initialised optimizer cannot fake), step=7 in meta."""
    torch.manual_seed(0)
    system = _system()
    with torch.no_grad():
        for p in system.parameters():
            p.copy_(torch.randn_like(p))
    opt = torch.optim.AdamW(system.parameters(), lr=3e-4, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=False)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    for _ in range(3):
        system.head(system.trunk(torch.randn(2, 4))).sum().backward()
        opt.step()
        opt.zero_grad()
    cm.save_checkpoint_dcp(str(tmp_path), system, [opt], meta_data={"step": 7})
    return tmp_path, system, opt


def _fresh():
    system = _system()
    with torch.no_grad():                    # perturb, so "untouched" is observable
        for p in system.parameters():
            p.fill_(7.0)
    opt = torch.optim.AdamW(system.parameters(), lr=6e-4, weight_decay=0.0,
                            betas=(0.8, 0.9), fused=False)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return system, opt


def _params_equal(a, b):
    return all(torch.equal(pa, pb)
               for pa, pb in zip(a.parameters(), b.parameters(), strict=True))


def _state_by_name(system, opt):
    return {name: opt.state[p] for name, p in system.named_parameters()}


# ---- counter-examples: red on the old loader -------------------------------------

def test_full_loader_has_no_switch():
    params = inspect.signature(cm.load_checkpoint_dcp).parameters
    assert "load_optimizer" not in params, "the weights-only switch is gone; use load_model_only"
    assert params["optimizers"].default is inspect.Parameter.empty, "optimizers is required"


def test_full_loader_refuses_no_optimizers(saved):
    ckpt, _, _ = saved
    system, _ = _fresh()
    for bad in ([], None, (o for o in [])):
        with pytest.raises(ValueError, match="load_model_only"):
            cm.load_checkpoint_dcp(str(ckpt), system, bad)
    # a generator would be consumed by get_state_dict and leave the optimizer
    # silently unrestored — refused as well
    _, opt = _fresh()
    with pytest.raises(ValueError, match="load_model_only"):
        cm.load_checkpoint_dcp(str(ckpt), system, (o for o in [opt]))
    # and it refused BEFORE touching the model
    assert torch.equal(system.trunk.w, torch.full((4,), 7.0))


# ---- the two intents -------------------------------------------------------------

def test_load_model_only_restores_weights_and_leaves_optimizer_alone(saved):
    ckpt, src, _ = saved
    system, opt = _fresh()
    before = _group(opt)
    cm.load_model_only(str(ckpt), system)
    assert _params_equal(system, src)
    assert opt.state == {}, "an init must not import optimizer state"
    assert _group(opt) == before, "an init must not touch optimizer hyperparameters"


def test_full_load_restores_weights_state_groups_and_meta(saved):
    ckpt, src, src_opt = saved
    system, opt = _fresh()
    meta = cm.load_checkpoint_dcp(str(ckpt), system, [opt])
    assert meta["step"] == 7
    assert _params_equal(system, src)
    live = _group(opt)
    for key in ("lr", "initial_lr", "weight_decay", "betas"):
        assert live[key] == _group(src_opt)[key], key
    saved, restored = _state_by_name(src, src_opt), _state_by_name(system, opt)
    assert saved.keys() == restored.keys()
    for name in saved:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(saved[name][key], restored[name][key]), (name, key)
        assert float(restored[name]["step"]) == 3.0
