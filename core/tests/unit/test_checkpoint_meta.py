"""Checkpoint self-description: save records model_config, loaders recover
(config_from_meta) and validate (_validate_model_config) against it.

The anti-loss-of-capability suite: this design once died silently — the load
side survived into the repo but the save side was never wired, so validation
skipped every checkpoint as "old format" and nobody noticed. The save-side
recording is pinned here so it cannot die quietly again.
"""

import json
import math
import sys
import types
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union

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
    assert cm.config_from_meta(str(tmp_path), GPTConfig) == TINY
    short = cm.config_from_meta(str(tmp_path), GPTConfig, sequence_len=16)
    assert short.sequence_len == 16 and short.n_embd == 32


def test_old_checkpoint_returns_none(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"step": 3}))
    assert cm.config_from_meta(str(tmp_path), GPTConfig) is None


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


# ---- what save_checkpoint_dcp writes, and replacing a step -------------------------------

def _save_into_step_dir(tmp_path, monkeypatch, meta_data, dataloader_state=None):
    monkeypatch.setattr(cm.dcp, "save", lambda **kw: None)
    target = tmp_path / "step_000007"
    cm.save_checkpoint_dcp(str(target), _tiny_system(), [], meta_data=meta_data, dataloader_state=dataloader_state)
    return target


def test_a_snapshot_in_json_types_is_saved_and_reads_back_equal(tmp_path, monkeypatch):
    snap = {"source_states": {"text": {"row_group": 5, "files": ["a", "b"], "frac": 0.5, "done": False,
                                       "next": None}}}
    target = _save_into_step_dir(tmp_path, monkeypatch, {"step": 7}, snap)
    assert json.loads((target / "meta.json").read_text())["dataloader_state"] == snap


@dataclass
class _TupleConfig:                     # a trunk config may default a sequence field to a tuple (NanoDSV4 does)
    n_embd: int = 8
    ratios: tuple = (0, 4, 32)


def test_a_tuple_field_of_the_model_config_is_saved_as_a_list_and_rebuilt_as_a_tuple(tmp_path, monkeypatch):
    monkeypatch.setattr(cm.dcp, "save", lambda **kw: None)
    cfg = _TupleConfig()
    cm.save_checkpoint_dcp(str(tmp_path), LMSystem(_Trunk(cfg), torch.nn.Linear(2, 2)), [], meta_data={"step": 7})
    assert json.loads((tmp_path / "meta.json").read_text())["model_config"]["ratios"] == [0, 4, 32]
    assert cm.config_from_meta(str(tmp_path), _TupleConfig) == cfg


@dataclass
class _AnnotatedTuples:                 # tuple fields told by their annotation alone
    plain: tuple
    typed: tuple[int, ...]
    n_embd: int = 8


def test_tuple_fields_without_a_tuple_default_are_rebuilt_as_tuples_too(tmp_path, monkeypatch):
    monkeypatch.setattr(cm.dcp, "save", lambda **kw: None)
    cfg = _AnnotatedTuples(plain=(1, 2), typed=(3, 4, 5))
    cm.save_checkpoint_dcp(str(tmp_path), LMSystem(_Trunk(cfg), torch.nn.Linear(2, 2)), [], meta_data={"step": 7})
    assert cm.config_from_meta(str(tmp_path), _AnnotatedTuples) == cfg


def _round_trip(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(cm.dcp, "save", lambda **kw: None)
    cm.save_checkpoint_dcp(str(tmp_path), LMSystem(_Trunk(cfg), torch.nn.Linear(2, 2)), [], meta_data={"step": 7})
    return cm.config_from_meta(str(tmp_path), type(cfg))


@dataclass
class _Nested:
    groups: tuple[list[int], ...]
    pairs: tuple[tuple[int, int], ...] = ((1, 2),)
    xs: list[int] = field(default_factory=lambda: [1, 2])
    loose: Any = None
    either: Union[tuple[int, ...], list[int], None] = None     # two sequence arms: nothing to go by
    bare_opt: Optional[tuple] = None
    n_embd: int = 8


def test_values_come_back_as_their_annotation_says_level_by_level(tmp_path, monkeypatch):
    """Lists the config holds stay lists (a list field, a list under Any, the list elements of a tuple field, a
    Union with two sequence arms keeps what JSON gave); tuples come back tuples at every level their annotation
    names."""
    cfg = _Nested(groups=([1, 2], [3]), loose=[5, 6], either=[1, 2], bare_opt=(3, (4, 5)))
    back = _round_trip(tmp_path, monkeypatch, cfg)
    assert back == cfg
    assert type(back.groups) is tuple and type(back.groups[0]) is list
    assert type(back.pairs) is tuple and type(back.pairs[0]) is tuple
    assert type(back.xs) is list and type(back.loose) is list and type(back.either) is list


@dataclass
class _Unnamed:                         # no element types named
    pairs: tuple = ((1, 2), (3, 4))
    loose: Any = ((5, 6),)
    n_embd: int = 8


def test_where_no_element_types_are_named_every_level_comes_back_a_tuple(tmp_path, monkeypatch):
    """A bare `tuple` annotation, or a field told only by its tuple default: every level comes back a tuple, as
    before the annotations were read."""
    for i, cfg in enumerate((_Unnamed(), _Unnamed(pairs=((7, 8),), loose=((9,),)))):
        assert _round_trip(tmp_path / str(i), monkeypatch, cfg) == cfg


_POSTPONED = """
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class PostponedTuples:
    ratios: tuple[int, ...]
    maybe: Optional[tuple[int, ...]]
    either: tuple[int, ...] | None = None
    n_embd: int = 8
"""


def test_postponed_and_optional_tuple_annotations_are_resolved(tmp_path, monkeypatch):
    """Under `from __future__ import annotations` a field's type is a string; it is resolved, so a tuple field
    without a tuple default still comes back a tuple, and so does an Optional one (None stays None)."""
    module = types.ModuleType("_postponed_tuple_config")
    monkeypatch.setitem(sys.modules, module.__name__, module)     # dataclass and get_type_hints look it up
    exec(_POSTPONED, module.__dict__)
    cfg = module.PostponedTuples(ratios=(0, 4, 32), maybe=(1,), either=(2, 3))
    assert _round_trip(tmp_path / "a", monkeypatch, cfg) == cfg
    cfg = module.PostponedTuples(ratios=(5,), maybe=None)
    assert _round_trip(tmp_path / "b", monkeypatch, cfg) == cfg


@dataclass
class _PartlyUnresolvable:
    typed: tuple[int, ...]
    thing: Optional["_NoSuchName"] = None      # makes get_type_hints fail for the whole class


_UNRESOLVABLE = """
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from nowhere import Missing


@dataclass
class Unresolvable:
    ratios: tuple = (1, 2)
    thing: Optional[Missing] = None
"""


def test_annotations_that_do_not_resolve_fall_back_to_each_fields_own(tmp_path, monkeypatch):
    """When some annotation does not resolve, every field keeps its own: a real tuple annotation still counts, and
    a postponed (string) one leaves the field to its tuple default."""
    cfg = _PartlyUnresolvable(typed=(3, 4))
    assert _round_trip(tmp_path / "a", monkeypatch, cfg) == cfg
    module = types.ModuleType("_unresolvable_tuple_config")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(_UNRESOLVABLE, module.__dict__)
    cfg = module.Unresolvable()
    back = _round_trip(tmp_path / "b", monkeypatch, cfg)
    assert back == cfg and type(back.ratios) is tuple


def test_a_non_finite_trainer_state_value_is_saved_and_reads_back(tmp_path, monkeypatch):
    # Python's json writes NaN / Infinity and reads them back as they were; a diverged loss does not stop a save
    trainer_state = {"smooth_train_loss": float("nan"), "total_training_time": float("inf")}
    target = _save_into_step_dir(tmp_path, monkeypatch, {"step": 7, "trainer_state": trainer_state})
    back = json.loads((target / "meta.json").read_text())["trainer_state"]
    assert math.isnan(back["smooth_train_loss"]) and back["total_training_time"] == float("inf")


def test_the_periodic_saver_replaces_a_step_whole(tmp_path, monkeypatch):
    """save_checkpoint_if_needed re-saving the last step (step 9 of 10) replaces the checkpoint that was there."""
    monkeypatch.setattr(cm.dcp, "save", lambda **kw: None)
    config = {"checkpoint": {"save_dir": str(tmp_path), "save_every": -1, "keep_last_n": 2}}
    step = tmp_path / "step_000009"

    def save(state):
        cm.save_checkpoint_if_needed(9, _tiny_system(), [], config, 0, 1, max_steps=10, dataloader_state=state)
    save({"source_states": {"text": {"pos": 1}}})
    (step / "__0_0.distcp").write_text("old shard")
    save({"source_states": {"text": {"pos": 2}}})
    assert not (step / "__0_0.distcp").exists()
    assert json.loads((step / "meta.json").read_text())["dataloader_state"] == {"source_states": {"text": {"pos": 2}}}


def test_every_rank_waits_for_rank_0s_deletion(tmp_path, monkeypatch):
    """With replace_existing every rank passes a barrier only after rank 0 removed the old directory."""
    monkeypatch.setattr(cm.dcp, "save", lambda **kw: None)
    step = tmp_path / "step_000007"
    barrier_saw = []
    monkeypatch.setattr(cm.torch.distributed, "barrier", lambda: barrier_saw.append(step.exists()))
    step.mkdir()
    (step / "__0_0.distcp").write_text("old shard")
    cm.save_checkpoint_dcp(str(step), _tiny_system(), [], {"step": 7}, rank=0, world_size=2, replace_existing=True)
    assert barrier_saw == [False] and not (step / "__0_0.distcp").exists()
