"""scaling.py keeps the curves of different head CE implementations apart (GPT round 38, 2026-10-22): `run` does not
skip a depth whose stored curve came from another head_ce or records none, and `fit` fits only curves of the current
HEAD_CE. Before, a study resumed after HEAD_CE moved from liger to compiled skipped every depth and `fit` labelled the
untouched liger curves compiled. No training here: run_curve and the plot are stand-ins."""
import argparse
import json

import pytest

from exemplars.text_pretrain import scaling


def _curve(depth, head_ce=None, dtype="fp32"):
    c = {"depth": depth, "N": scaling.n_nonembed(depth), "param_dtype": dtype,
         "trajectory": [{"step": 20, "tokens": 655360, "compute": 1.0, "val": 9.0}]}
    if head_ce is not None:
        c["head_ce"] = head_ce
    return c


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(scaling, "OUT", tmp_path)
    return tmp_path


def test_run_trains_the_depths_whose_curves_are_another_arms(out, monkeypatch):
    # the shard name of before (no head) and the current one holding curves of another arm / none recorded
    (out / "curves_2-3-4_fp32.json").write_text(json.dumps({"curves": [_curve(d, "liger") for d in (2, 3, 4)]}))
    (out / f"curves_2-3-4_fp32_{scaling.HEAD_CE}.json").write_text(
        json.dumps({"curves": [_curve(2, "liger"), _curve(3), _curve(4, scaling.HEAD_CE)]}))
    trained = []
    monkeypatch.setattr(scaling, "run_curve",
                        lambda d, param_dtype="fp32": trained.append(d) or _curve(d, scaling.HEAD_CE, param_dtype))
    scaling.cmd_run(argparse.Namespace(depths=[2, 3, 4], param_dtype="fp32"))
    assert trained == [2, 3]                                   # d4's curve is this arm's: resumed, not retrained
    stored = json.loads((out / f"curves_2-3-4_fp32_{scaling.HEAD_CE}.json").read_text())["curves"]
    assert sorted(c["depth"] for c in stored) == [2, 3, 4]
    assert all(c["head_ce"] == scaling.HEAD_CE for c in stored)


def test_fit_refuses_curves_of_another_arm_or_none_recorded(out):
    (out / "curves_2-3-4_fp32.json").write_text(
        json.dumps({"curves": [_curve(2, "liger"), _curve(3, "liger"), _curve(4)]}))
    with pytest.raises(SystemExit, match="only 0"):
        scaling.cmd_fit(argparse.Namespace(param_dtype="fp32"))


def test_fit_takes_this_arms_curves_only(out, monkeypatch):
    mine = [_curve(d, scaling.HEAD_CE) for d in (2, 3, 4)]
    (out / f"curves_2-3-4_fp32_{scaling.HEAD_CE}.json").write_text(json.dumps({"curves": mine}))
    (out / "curves_6-8_fp32.json").write_text(json.dumps({"curves": [_curve(6, "liger"), _curve(8)]}))
    monkeypatch.setattr(scaling.scaling_fit, "frontier_exponent", lambda curves: 0.5)
    monkeypatch.setattr(scaling, "_plot", lambda curves, a: None)
    monkeypatch.setattr(scaling, "_compare_to_example", lambda curves, a: None)
    scaling.cmd_fit(argparse.Namespace(param_dtype="fp32"))
    fitted = json.loads((out / "scaling.json").read_text())
    assert fitted["study"]["head_ce"] == scaling.HEAD_CE
    assert [c["depth"] for c in fitted["curves"]] == [2, 3, 4]
