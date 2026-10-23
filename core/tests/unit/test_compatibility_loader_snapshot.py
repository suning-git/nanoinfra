"""core.compatibility.loader_snapshot: reading a list snapshot (written before the 2026-10 change) with the
name-keyed loader — accept_list_snapshot at load time, upgrade_list_snapshot once on meta.json. CPU only."""
import json
import subprocess
import sys
from functools import partial

import pytest
import torch

from core import compatibility
from core.data import MixedDataLoader, PaddingBatcher, build_NTP_supervised_sample
from core.data.supervision import IGNORE


class Indexed:
    def __init__(self, base):
        self.base, self.state = base, {"index": 0}

    def __iter__(self):
        while True:
            k = self.state["index"]
            self.state = {"index": k + 1}
            yield {"tokens": torch.full((5,), self.base + k, dtype=torch.long), "meta": {"k": k, "src": self.base}}

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


def loader(sources):
    return MixedDataLoader({n: (s, 1.0) for n, s in sources.items()},
                           build_supervised_sample_fn=build_NTP_supervised_sample,
                           create_batcher_fn=partial(PaddingBatcher, shape=(2, 8),
                                                     pad_values={"idx": 0, "targets": IGNORE}))


def test_one_source_needs_no_order_and_the_accum_is_dropped(capsys):
    ld = compatibility.accept_list_snapshot(loader({"text": Indexed(0)}))
    ld.set_state({"source_states": [{"index": 7}], "accum": [0.25]})
    assert [m["k"] for m in next(iter(ld))["sample_meta"]] == [7, 8]
    assert "0 -> 'text'; its 'accum' dropped" in capsys.readouterr().out


def test_several_sources_follow_the_given_order():
    ld = loader({"a": Indexed(0), "b": Indexed(1000)})
    assert compatibility.accept_list_snapshot(ld, ["b", "a"]) is ld          # the same loader, its set_state wrapped
    ld.set_state({"source_states": [{"index": 3}, {"index": 9}]})          # by position: b, then a
    got = {m["src"]: m["k"] for m in next(iter(ld))["sample_meta"]}
    assert got == {1000: 3, 0: 9}


def test_a_name_keyed_snapshot_still_restores_as_usual():
    ld = compatibility.accept_list_snapshot(loader({"a": Indexed(0)}))
    ld.set_state({"source_states": {"a": {"index": 4}}})
    assert [m["k"] for m in next(iter(ld))["sample_meta"]] == [4, 5]


def test_wrong_orders_are_refused():
    two = loader({"a": Indexed(0), "b": Indexed(1000)})
    with pytest.raises(ValueError, match="source_order is needed"):
        compatibility.accept_list_snapshot(two)
    with pytest.raises(ValueError, match="does not have: \\['c'\\]"):
        compatibility.accept_list_snapshot(two, ["a", "c"])
    with pytest.raises(ValueError, match="repeats \\['a'\\]"):
        compatibility.accept_list_snapshot(two, ["a", "a"])
    with pytest.raises(TypeError, match="not a string"):
        compatibility.accept_list_snapshot(loader({"t2m": Indexed(0)}), "t2m")
    with pytest.raises(TypeError, match="nonempty strings"):
        compatibility.accept_list_snapshot(two, ["a", 0])
    ld = compatibility.accept_list_snapshot(loader({"a": Indexed(0), "b": Indexed(1000)}), ["a", "b"])
    with pytest.raises(ValueError, match="has 1 source states but source_order names 2"):
        ld.set_state({"source_states": [{"index": 1}]})


def _write_meta(directory, snapshot):
    meta = {"step": 12, "dataloader_state": snapshot, "model_config": {"n_layer": 2}}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def test_upgrade_rewrites_meta_json_once_and_keeps_a_backup(tmp_path):
    original = _write_meta(tmp_path, {"source_states": [{"index": 3}, None], "accum": [0.1, -0.1]})
    compatibility.upgrade_list_snapshot(str(tmp_path), ["x", "y"])
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["dataloader_state"] == {"source_states": {"x": {"index": 3}, "y": None}}
    assert {k: v for k, v in meta.items() if k != "dataloader_state"} == {"step": 12, "model_config": {"n_layer": 2}}
    assert json.loads((tmp_path / "meta.json.bak").read_text()) == original
    # a name-keyed snapshot is left alone, and a second backup is never written over the first
    compatibility.upgrade_list_snapshot(str(tmp_path), ["x", "y"])
    assert json.loads((tmp_path / "meta.json.bak").read_text()) == original
    _write_meta(tmp_path, {"source_states": [{"index": 1}, None]})
    with pytest.raises(FileExistsError):
        compatibility.upgrade_list_snapshot(str(tmp_path), ["x", "y"])


def test_upgrade_runs_from_the_command_line(tmp_path):
    _write_meta(tmp_path, {"source_states": [{"index": 5}]})
    out = subprocess.run([sys.executable, "-m", "core.compatibility.loader_snapshot", str(tmp_path), "text"],
                         capture_output=True, text=True, check=True)
    assert json.loads((tmp_path / "meta.json").read_text())["dataloader_state"] == {
        "source_states": {"text": {"index": 5}}}
    assert "rewritten by name" in out.stdout and "Warning" not in out.stderr
