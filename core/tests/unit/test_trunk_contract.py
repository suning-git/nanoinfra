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


def test_load_system_builds_bf16_storage_unless_told(monkeypatch, tmp_path):
    # Inference keeps loading into bf16 when the training default changes (2026-10): load_system passes its
    # own param_dtype, bf16 by default, rather than inheriting build_system's.
    import json
    import torch
    from dataclasses import asdict
    from core.training import model_setup
    from core.model import checkpoint_manager
    (tmp_path / "meta.json").write_text(json.dumps({"step": 0, "model_config": asdict(TINY), "model_arch": "GPT"}))
    seen = []

    class _System:
        def eval(self):
            return self

    monkeypatch.setattr(model_setup, "build_system", lambda cls, cfg, **kw: seen.append(kw) or
                        {"system": _System(), "rank": 0, "world_size": 1})
    monkeypatch.setattr(checkpoint_manager, "load_model_only", lambda *a, **kw: None)
    model_setup.load_system(str(tmp_path))
    model_setup.load_system(str(tmp_path), param_dtype=torch.float32)
    assert [kw["param_dtype"] for kw in seen] == [torch.bfloat16, torch.float32]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="build_system places the model on CUDA")
def test_build_system_stores_fp32_unless_told():
    # The training default (since 2026-10): parameters, and so gradients and AdamW moments, in fp32; bf16 on request.
    from core.training.model_setup import build_system
    default = build_system(GPT, TINY, parallel="ddp")["system"]
    assert {p.dtype for p in default.parameters()} == {torch.float32}
    bf16 = build_system(GPT, TINY, parallel="ddp", param_dtype=torch.bfloat16)["system"]
    assert {p.dtype for p in bf16.parameters()} == {torch.bfloat16}


def test_build_system_refuses_a_storage_it_does_not_offer():
    # Checked first, before any process group or model: with parameters only converted through bf16 into
    # param_dtype, float16 would otherwise go through as fp16 master weights, with no GradScaler anywhere.
    from core.training.model_setup import build_system
    with pytest.raises(ValueError, match="param_dtype must be torch.bfloat16 or torch.float32"):
        build_system(GPT, TINY, parallel="ddp", param_dtype=torch.float16)
