"""Checkpoint self-description: save records model_config, loaders recover
(gpt_config_from_meta) and validate (_validate_model_config) against it.

The anti-loss-of-capability suite: this design once died silently — the load
side survived into the repo but the save side was never wired, so validation
skipped every checkpoint as "old format" and nobody noticed. The save-side
recording is pinned here so it cannot die quietly again.
"""

import json
from dataclasses import replace

import pytest
import torch

from core.model.gpt import GPTConfig
from core.model.system import LMSystem
from core.model import checkpoint_manager as cm


class _Trunk(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w = torch.nn.Parameter(torch.zeros(2))


TINY = GPTConfig(sequence_len=64, vocab_size=128, n_layer=2,
                 n_head=2, n_kv_head=2, n_embd=32, n_token_types=3)


def _tiny_system(**overrides):
    return LMSystem(_Trunk(replace(TINY, **overrides)), torch.nn.Linear(2, 2))


def _save(tmp_path, monkeypatch, system):
    # dcp.save itself needs no exercising here — pin OUR meta.json contract.
    monkeypatch.setattr(cm.dcp, "save", lambda **kw: None)
    cm.save_checkpoint_dcp(str(tmp_path), system, [], meta_data={"step": 7})


def test_system_exposes_trunk_config():
    assert _tiny_system().config.n_embd == 32


def test_save_records_model_config(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, _tiny_system())
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["model_config"] == {
        "sequence_len": 64, "vocab_size": 128, "n_layer": 2,
        "n_head": 2, "n_kv_head": 2, "n_embd": 32, "n_token_types": 3}


def test_save_model_config_live_model_is_the_only_authority(tmp_path, monkeypatch):
    # A caller-supplied 'model_config' must NOT win over the model being saved
    # (silent caller-precedence is how the stale 4-field writer once shadowed
    # the full record). The live model's config is written unconditionally.
    monkeypatch.setattr(cm.dcp, "save", lambda **kw: None)
    cm.save_checkpoint_dcp(str(tmp_path), _tiny_system(), [],
                           meta_data={"step": 7, "model_config": {"n_layer": 999}})
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["model_config"]["n_layer"] == 2


def test_blueprint_roundtrip_and_seqlen_override(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, _tiny_system())
    assert cm.gpt_config_from_meta(str(tmp_path)) == TINY
    short = cm.gpt_config_from_meta(str(tmp_path), sequence_len=16)
    assert short.sequence_len == 16 and short.n_embd == 32


def test_old_checkpoint_returns_none(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"step": 3}))
    assert cm.gpt_config_from_meta(str(tmp_path)) is None


def test_validate_strict_on_weight_shaping(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, _tiny_system())
    # n_head mismatch would not even shape-error at DCP level (attention
    # weights are [dim, 3*dim] regardless) — the validator must catch it.
    with pytest.raises(ValueError, match="n_head"):
        cm._validate_model_config(str(tmp_path), _tiny_system(n_head=1))


def test_validate_lenient_on_sequence_len(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, _tiny_system())
    cm._validate_model_config(str(tmp_path), _tiny_system(sequence_len=16))  # no raise


# --- Architecture tag (the trunk_cls seam's checked default) -----------------
# build_system stamps `system.arch = trunk_cls.__name__` at ASSEMBLY time
# (type(trunk) at save time lies — fully_shard rewrites it to FSDP{ClassName});
# save records it; loaders audit the caller's class choice against it.

def test_save_records_model_arch(tmp_path, monkeypatch):
    system = _tiny_system()
    system.arch = "GPT"
    _save(tmp_path, monkeypatch, system)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["model_arch"] == "GPT"


def test_save_without_arch_writes_no_tag(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, _tiny_system())  # bare system, no .arch
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert "model_arch" not in meta


def test_validate_arch_mismatch_raises(tmp_path, monkeypatch):
    saved = _tiny_system()
    saved.arch = "GPT"
    _save(tmp_path, monkeypatch, saved)
    other = _tiny_system()
    other.arch = "MambaTrunk"
    with pytest.raises(ValueError, match="model_arch"):
        cm._validate_model_config(str(tmp_path), other)


def test_validate_arch_lenient_when_either_side_untagged(tmp_path, monkeypatch):
    # Old checkpoints (no tag) and bare models (no .arch) skip the audit —
    # you cannot audit against a fact that was never recorded.
    _save(tmp_path, monkeypatch, _tiny_system())          # untagged checkpoint
    tagged = _tiny_system()
    tagged.arch = "GPT"
    cm._validate_model_config(str(tmp_path), tagged)      # no raise


def test_config_from_meta_generic(tmp_path, monkeypatch):
    # The generic entry the trunk contract feeds (trunk_cls.Config);
    # gpt_config_from_meta is its GPTConfig specialization.
    _save(tmp_path, monkeypatch, _tiny_system())
    assert cm.config_from_meta(str(tmp_path), GPTConfig) == TINY
