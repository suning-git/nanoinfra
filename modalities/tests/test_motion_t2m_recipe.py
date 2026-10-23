"""The nano_motion exemplar's T2M recipe builds, and supervises the motion segments only.

Why this test exists. exemplars/nano_motion/configs/train_t2m.yaml shipped its t2m recipe as `supervise: tags` with a
two-entry `supervise_tags` against an eight-segment template (since 07-28). SequenceRecipe requires one tag per
segment, so T2M training stopped while building its data source with "supervise_tags length (2) != template length
(8)"; and `set("tags")` would have been a set of letters had it got past that. A student hit it in a course project
(public repo PR #7, 2026-08-10, with the one-line fix). The config now names the supervised segments directly
(SequenceRecipe level 2). The fix came in with the data-pipeline rewrite of 2026-10; this test pins it.
"""
from pathlib import Path

import yaml

from modalities.motion.data.sources import create_recipe

CFG = Path(__file__).resolve().parents[2] / "exemplars" / "nano_motion" / "configs" / "train_t2m.yaml"


def _recipes():
    return yaml.safe_load(CFG.read_text())["recipe"]


def test_t2m_recipe_builds_and_supervises_motion_only():
    r = create_recipe(_recipes()["t2m"])
    supervised = [seg for i, seg in enumerate(r.template) if r._is_supervised(i)]
    assert supervised == ["motion_tokens", "motion_end"]


def test_motion_recipe_supervises_everything():
    r = create_recipe(_recipes()["motion"])
    assert all(r._is_supervised(i) for i in range(len(r.template)))
