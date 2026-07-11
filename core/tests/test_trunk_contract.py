"""The trunk contract.

WHICH model is the orchestrator's decision; core assembles WHATEVER satisfies
the contract: __init__(config), init_weights(), forward -> hidden, `blocks`
(FSDP grouping units), estimate_flops(), class attr `Config`. GPT is the
reference implementation — pin that it actually satisfies it, and that the
system-level FLOPs decomposition matches the historical formula (MFU numbers
must not drift on a refactor).
"""

import pytest
import torch

from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.system import LMSystem

TINY = GPTConfig(sequence_len=64, vocab_size=128, n_layer=2,
                 n_head=2, n_kv_head=2, n_embd=32, n_token_types=3)


def test_gpt_declares_its_config_class():
    assert GPT.Config is GPTConfig


def test_gpt_exposes_blocks():
    trunk = GPT(TINY)
    # `blocks` is the assembly-facing name; transformer.h is private layout.
    assert list(trunk.blocks) == list(trunk.transformer.h)
    assert len(trunk.blocks) == TINY.n_layer


def test_system_flops_match_historical_formula():
    trunk, head = GPT(TINY), LMHead(TINY.n_embd, TINY.vocab_size)
    system = LMSystem(trunk, head)
    # Pre-seam system-level formula: 6*(all params - wte) + attention term.
    nparams = sum(p.numel() for p in system.parameters())
    nparams_wte = trunk.transformer.wte.weight.numel()
    l, h = TINY.n_layer, TINY.n_head
    q, t = TINY.n_embd // TINY.n_head, TINY.sequence_len
    historical = 6 * (nparams - nparams_wte) + 12 * l * h * q * t
    assert system.estimate_flops() == historical


def test_old_name_wrapper_passes_gpt(monkeypatch):
    # setup_model_for_training keeps the GPT-implied single-config signature
    # for older callers; it must delegate to build_system with GPT.
    from core.training import model_setup
    seen = {}
    monkeypatch.setattr(model_setup, "build_system",
                        lambda cls, cfg, **kw: seen.update(cls=cls, cfg=cfg) or "ok")
    assert model_setup.setup_model_for_training(TINY, use_compile=False) == "ok"
    assert seen["cls"] is GPT and seen["cfg"] is TINY
