"""Trunk compilation is the orchestrator's decision; core provides two mechanisms.

Decided 2026-09-10: build_system / load_system do NOT compile the
trunk. After assembly the orchestrator calls ONE of
  - compile_system_trunk(system, **kw): compile the trunk as a whole, in place
    (nn.Module.compile on the registered trunk);
  - compile_blocks(trunk, dynamic=...): torch.compile each block's forward in place.

These run on CPU: torch.compile is lazy, so compiling sets up a callable and compiles nothing.
The counter-examples come first — repeating a mechanism, or mixing the two in either
order, is refused rather than silently compiling a compiled callable — then the positive
mechanics: `system.trunk` stays the registered module, training and teacher-forcing
evaluation both run its compiled call, and a checkpoint saves and loads unchanged.
"""

import contextlib
import inspect

import pytest
import torch
import torch.nn as nn

from core.evaluation.evaluator import LossEvaluator
from core.evaluation.metrics import mean_ce
from core.model import checkpoint_manager as cm
from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.system import LMSystem, prepare_unpacked_AR_trunk_inputs
from core.training.model_setup import (build_system, compile_blocks, compile_system_trunk,
                                       load_system)
from core.training.optim import build_optimizers

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
    return LMSystem(_StubTrunk(), LMHead(TINY.n_embd, TINY.vocab_size),
                    prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)


def _gpt_system():
    return LMSystem(GPT(TINY), LMHead(TINY.n_embd, TINY.vocab_size),
                    prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)


# ---- the API deletion itself ------------------------------------------------------

def test_assembly_has_no_use_compile_parameter():
    """build_system / load_system decide placement and head, not trunk compilation."""
    assert "use_compile" not in inspect.signature(build_system).parameters
    assert "use_compile" not in inspect.signature(load_system).parameters


# ---- counter-examples: one mechanism, once ----------------------------------------

def test_whole_trunk_twice_is_refused():
    system = _stub_system()
    compile_system_trunk(system)
    with pytest.raises(RuntimeError, match="already compiled as a whole"):
        compile_system_trunk(system)


def test_whole_after_per_block_is_refused():
    system = _stub_system()
    compile_blocks(system.trunk)
    with pytest.raises(RuntimeError, match="already compiled per block"):
        compile_system_trunk(system)


def test_per_block_after_whole_is_refused():
    """The whole-trunk compile is in place on the trunk itself, so compile_blocks (which takes the trunk) sees it."""
    system = _stub_system()
    compile_system_trunk(system)
    with pytest.raises(RuntimeError, match="already compiled as a whole"):
        compile_blocks(system.trunk)


def test_per_block_twice_is_refused():
    trunk = _StubTrunk()
    compile_blocks(trunk)
    with pytest.raises(RuntimeError, match="already compiled"):
        compile_blocks(trunk)


# ---- positive mechanics -----------------------------------------------------------

def test_compile_kwargs_reach_torch_compile_unchanged(monkeypatch):
    seen = {}
    monkeypatch.setattr(torch, "compile",
                        lambda fn, **kw: seen.update(fn=fn, **kw) or "COMPILED")
    system = _stub_system()
    trunk = system.trunk
    compile_system_trunk(system, dynamic=False, mode="max-autotune")
    assert seen["fn"].__self__ is trunk                # the registered trunk's own call, compiled in place
    assert {k: v for k, v in seen.items() if k != "fn"} == {"dynamic": False,
                                                          "mode": "max-autotune"}
    assert system.trunk is trunk and trunk._compiled_call_impl == "COMPILED"


def test_training_and_evaluation_run_the_compiled_trunk(monkeypatch):
    """Both call `system.trunk`, so a whole-trunk compile reaches both. (Before H14 the compiled trunk was a separate
    object that only loss() used; the evaluator ran the registered module uncompiled.)"""
    ran = []

    def fake_compile(fn, **kw):
        def compiled(*a, **k):
            ran.append("compiled")
            return fn(*a, **k)
        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    system = _stub_system()
    compile_system_trunk(system)
    batch = {"idx": torch.randint(0, TINY.vocab_size, (2, 8)),
             "targets": torch.randint(0, TINY.vocab_size, (2, 8))}
    loss = system.loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    LossEvaluator(dataloader=[batch], eval_steps=1, metrics={"val/ce": mean_ce()}).evaluate(
        system, contextlib.nullcontext())
    assert ran == ["compiled", "compiled"]


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
    before_params = [id(p) for p in system.parameters()]
    compile_system_trunk(system, dynamic=True)
    assert system.trunk is trunk and isinstance(system.trunk, GPT)   # the same module, compiled in place
    assert trunk._compiled_call_impl is not None
    assert list(system.state_dict().keys()) == before_keys
    assert [id(p) for p in system.parameters()] == before_params


def test_checkpoint_round_trip_after_a_whole_trunk_compile(tmp_path):
    """torch.distributed.checkpoint resolves every key through getattr (trunk -> transformer -> ...), so a compiled
    trunk must still answer to `system.trunk` as the registered module. A separate torch.compile wrapper returned
    under that name made saving and loading raise (Expected '_orig_mod')."""
    system = _gpt_system()
    opts = build_optimizers(system, {"type": "adamw"})
    compile_system_trunk(system, dynamic=True)
    cm.save_checkpoint_dcp(str(tmp_path / "step_000001"), system, opts, meta_data={"step": 1})
    fresh = _gpt_system()
    fresh_opts = build_optimizers(fresh, {"type": "adamw"})
    compile_system_trunk(fresh, dynamic=True)
    cm.load_checkpoint_dcp(str(tmp_path / "step_000001"), fresh, fresh_opts)
    saved, loaded = system.state_dict(), fresh.state_dict()
    assert list(saved) == list(loaded) and all(torch.equal(saved[k], loaded[k]) for k in saved)


def test_force_eager_runs_a_compiled_trunk_without_compiling():
    """The documented debugging bypass (compile_system_trunk's docstring): under set_stance("force_eager") the
    in-place compiled trunk runs eagerly and nothing is compiled; a normal call afterwards compiles."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    system = _stub_system()
    compile_system_trunk(system, backend=backend)
    idx = torch.randint(0, TINY.vocab_size, (2, 8))
    with torch.compiler.set_stance("force_eager"):
        system.trunk(idx)
    assert graphs == []
    system.trunk(idx)
    n = len(graphs)
    assert n >= 1
    with torch.compiler.set_stance("force_eager"):                 # and after it has run compiled
        system.trunk(torch.randint(0, TINY.vocab_size, (3, 5)))    # a new shape would otherwise compile again
    assert len(graphs) == n


def test_compiled_trunk_matches_eager_through_train_eval_train():
    """Real dynamo (a capturing backend, no inductor), the Trainer's sequence: training steps, eval() + no_grad,
    training again. Outputs and gradients equal the eager trunk's, and going back to training reuses its graphs."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch.manual_seed(0)
    eager, compiled = _gpt_system(), _gpt_system()
    compiled.load_state_dict(eager.state_dict())
    compile_system_trunk(compiled, backend=backend)
    batch = {"idx": torch.randint(0, TINY.vocab_size, (2, 8)),
             "targets": torch.randint(0, TINY.vocab_size, (2, 8))}

    def train_step(system):
        system.train()
        system.zero_grad(set_to_none=True)
        loss = system.loss(batch)
        loss.backward()
        return loss.detach(), [None if p.grad is None else p.grad.clone() for p in system.parameters()]

    for system in (eager, compiled):
        system.trunk.register_forward_pre_hook(lambda m, a: None)     # a root hook, as FSDP installs one
    def same(ga, gb):
        return all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                   for a, b in zip(ga, gb))

    (le, ge), (lc, gc) = train_step(eager), train_step(compiled)
    assert torch.equal(le, lc) and same(ge, gc)
    for system in (eager, compiled):
        system.eval()
    with torch.no_grad():
        assert torch.equal(eager.trunk(batch["idx"]), compiled.trunk(batch["idx"]))
    n = len(graphs)
    (le, ge), (lc, gc) = train_step(eager), train_step(compiled)
    assert torch.equal(le, lc) and same(ge, gc)
    assert len(graphs) == n                                             # training's graph reused, no recompile


def test_each_mechanism_logs_its_action(capsys):
    compile_blocks(_StubTrunk(), dynamic=False)
    compile_system_trunk(_stub_system(), dynamic=True)
    out = capsys.readouterr().out
    assert "trunk compile: per-block, dynamic=False" in out
    assert "trunk compile: whole, {'dynamic': True}" in out
