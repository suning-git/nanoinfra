"""Trunk compilation is the orchestrator's decision; core provides two mechanisms.

Decided 2026-09-10: build_system / load_system do NOT compile the
trunk. After assembly the orchestrator calls ONE of
  - compile_system_trunk(system, **kw): torch.compile the trunk as a whole and
    register the wrapper with the System (consumed by LMSystem._run_trunk);
  - compile_blocks(trunk, dynamic=...): torch.compile each block's forward in place.

These run on CPU: torch.compile is lazy, so creating a wrapper compiles nothing.
The counter-examples come first — repeating a mechanism, or per-block THEN whole, is
refused rather than silently wrapped twice; the reverse order (whole, then per-block)
is not detectable and is pinned as such — then the positive mechanics: the registered
wrapper is what `loss()` actually runs, and neither mechanism disturbs the registry.
"""

import inspect

import pytest
import torch
import torch.nn as nn

from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.system import LMSystem
from core.training.model_setup import (build_system, compile_blocks, compile_system_trunk,
                                       load_system)

TINY = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2,
                 n_head=2, n_kv_head=2, n_embd=16, n_token_types=2)


class _StubTrunk(nn.Module):
    """The slice of the trunk contract these tests touch: `blocks` + forward -> hidden.
    A stub so the forward runs on CPU without the real attention kernels."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(TINY.vocab_size, TINY.n_embd)
        self.blocks = nn.ModuleList([nn.Linear(TINY.n_embd, TINY.n_embd) for _ in range(2)])
        self.calls = 0

    def forward(self, idx, token_types=None):
        self.calls += 1
        h = self.wte(idx)
        for block in self.blocks:
            h = block(h)
        return h


def _stub_system():
    return LMSystem(_StubTrunk(), LMHead(TINY.n_embd, TINY.vocab_size))


def _gpt_system():
    return LMSystem(GPT(TINY), LMHead(TINY.n_embd, TINY.vocab_size))


# ---- the API deletion itself ------------------------------------------------------

def test_assembly_has_no_use_compile_parameter():
    """build_system / load_system decide placement and head, not trunk compilation."""
    assert "use_compile" not in inspect.signature(build_system).parameters
    assert "use_compile" not in inspect.signature(load_system).parameters


# ---- counter-examples: one mechanism, once ----------------------------------------

def test_whole_trunk_twice_is_refused():
    system = _stub_system()
    compile_system_trunk(system)
    with pytest.raises(RuntimeError, match="already registered"):
        compile_system_trunk(system)


def test_whole_after_per_block_is_refused():
    system = _stub_system()
    compile_blocks(system.trunk)
    with pytest.raises(RuntimeError, match="already compiled per block"):
        compile_system_trunk(system)


def test_whole_then_per_block_is_not_detected():
    """Pins the asymmetry, so nobody reads the guard as covering both orders:
    compile_blocks takes the trunk, not the System, and cannot see the wrapper."""
    system = _stub_system()
    compile_system_trunk(system)
    compile_blocks(system.trunk)                      # no error: both are now live
    assert system._compiled_trunk is not None
    assert all(getattr(b.forward, "_torchdynamo_orig_callable", None) is not None
               for b in system.trunk.blocks)


def test_per_block_twice_is_refused():
    trunk = _StubTrunk()
    compile_blocks(trunk)
    with pytest.raises(RuntimeError, match="already compiled"):
        compile_blocks(trunk)


def test_setter_refuses_overwrite_but_none_clears():
    system = _stub_system()
    system.set_compiled_trunk(lambda idx, token_types=None: None)
    with pytest.raises(RuntimeError):
        system.set_compiled_trunk(lambda idx, token_types=None: None)
    system.set_compiled_trunk(None)                   # deliberate switch: clear first
    system.set_compiled_trunk(lambda idx, token_types=None: None)


# ---- positive mechanics -----------------------------------------------------------

def test_compile_kwargs_reach_torch_compile_unchanged(monkeypatch):
    seen = {}
    monkeypatch.setattr(torch, "compile",
                        lambda fn, **kw: seen.update(fn=fn, **kw) or "WRAPPER")
    system = _stub_system()
    compile_system_trunk(system, dynamic=False, mode="max-autotune")
    assert seen["fn"] is system.trunk
    assert {k: v for k, v in seen.items() if k != "fn"} == {"dynamic": False,
                                                          "mode": "max-autotune"}
    assert system._compiled_trunk == "WRAPPER"


def test_registered_wrapper_is_what_loss_runs():
    """The setter is only worth anything if `loss()` goes through what it registered."""
    system = _stub_system()
    ran = []

    def wrapper(idx, token_types=None):
        ran.append(idx.shape)
        return system.trunk(idx, token_types=token_types)

    batch = {"idx": torch.randint(0, TINY.vocab_size, (2, 8)),
             "targets": torch.randint(0, TINY.vocab_size, (2, 8))}
    system.set_compiled_trunk(wrapper)
    loss = system.loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert ran == [(2, 8)]
    system.set_compiled_trunk(None)
    system.loss(batch)
    assert ran == [(2, 8)]                            # eager path again: wrapper not called


def test_compile_blocks_keeps_module_identity_and_keys():
    trunk = GPT(TINY)
    before_keys = list(trunk.state_dict().keys())
    before_blocks = [b for b in trunk.blocks]
    compile_blocks(trunk)
    assert list(trunk.state_dict().keys()) == before_keys
    assert all(a is b for a, b in zip(trunk.blocks, before_blocks))
    assert all(getattr(b.forward, "_torchdynamo_orig_callable", None) is not None
               for b in trunk.blocks)


def test_compile_system_trunk_keeps_registry():
    system = _gpt_system()
    trunk = system.trunk
    before_keys = list(system.state_dict().keys())
    n_params = sum(1 for _ in system.parameters())
    compile_system_trunk(system, dynamic=True)
    assert system.trunk is trunk                      # the registered module is untouched
    assert list(system.state_dict().keys()) == before_keys
    assert sum(1 for _ in system.parameters()) == n_params
    assert system._compiled_trunk is not None


def test_each_mechanism_logs_its_action(capsys):
    compile_blocks(_StubTrunk(), dynamic=False)
    compile_system_trunk(_stub_system(), dynamic=True)
    out = capsys.readouterr().out
    assert "trunk compile: per-block, dynamic=False" in out
    assert "trunk compile: whole, {'dynamic': True}" in out
