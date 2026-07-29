"""Guards for the train/eval ruler agreement in the text modality.

These encode the two checks that actually caught the 2026-07-26 bug, where
training assembled `bos + text_start + … + text_end + eos` while evaluation
packed raw text with only a bos — silently, for months.

Run: .venv/bin/python -m pytest modalities/tests/test_text_streams.py -q
(Needs the FineWeb shards on disk; skipped otherwise.)
"""

from pathlib import Path

import pytest
import torch

import modalities.text
from modalities.text import get_tokenizer
from modalities.text.datasets import resolve_split
from modalities.text.fineweb import token_data_loader
from modalities.text.streams import make_loader_factory, resolve_sources
from modalities.text.train_text import SOURCE_TYPES, assemble_vocab
from core.data.mixed_dataloader import MixedDataLoader


def _config():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    cd = Path(modalities.text.__file__).resolve().parent / "configs"
    with initialize_config_dir(config_dir=str(cd), version_base=None):
        cfg = compose(config_name="train_text",
                      overrides=["sequence_len=512", "device_batch_size=4", "max_steps=10"])
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.fixture(scope="module")
def env():
    cfg = _config()
    try:
        resolve_split(cfg["data"], "fineweb", "val")
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"FineWeb shards unavailable: {e}")
    tok = get_tokenizer()
    layout, resolver = assemble_vocab(tok)
    return cfg, {"text": tok, "layout": layout, "control_resolver": resolver}, tok


def _control_ids(ids, tok):
    lo = tok.get_vocab_size() - len(tok.get_special_tokens())
    flat = ids.flatten().cpu()
    return sorted(torch.unique(flat[flat >= lo]).tolist())


def test_splits_are_disjoint(env):
    """train and val must never overlap — the single-shard case used to alias them."""
    cfg, _, _ = env
    train = set(resolve_split(cfg["data"], "fineweb", "train"))
    val = set(resolve_split(cfg["data"], "fineweb", "val"))
    assert train and val
    assert not (train & val)


def test_undeclared_split_raises(env):
    """No inference: an undeclared split is an error, not a guess."""
    cfg, _, _ = env
    with pytest.raises(KeyError):
        resolve_split(cfg["data"], "fineweb", "test")


def test_train_and_eval_structure_signatures_match(env):
    """THE regression guard: same control-token structure on both sides."""
    cfg, toks, tok = env
    data_cfg = cfg["data"]
    srcs = resolve_sources(data_cfg, 512, device="cuda" if torch.cuda.is_available() else "cpu")
    train_batch = next(iter(MixedDataLoader(
        {"batch_size": 4, "data": {"sequence_len": 512, "sources": srcs}},
        toks, SOURCE_TYPES, None)))

    factory = make_loader_factory(data_cfg, toks, SOURCE_TYPES,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    val_batch = next(factory(files=resolve_split(data_cfg, "fineweb", "val"),
                             recipe=data_cfg["recipes"]["text_pretrain"],
                             recipe_name="text_pretrain", B=4, T=512))

    assert _control_ids(train_batch["idx"], tok) == _control_ids(val_batch["idx"], tok)


def test_raw_bridge_really_is_the_old_ruler(env):
    """The `recipe: null` bridge must reproduce the OLD packing (bos only).

    If this ever starts matching the training signature, the bridge metric has
    stopped being a bridge and the two series are no longer distinguishable.
    """
    cfg, _, tok = env
    raw = next(token_data_loader(B=4, T=512,
                                 files=resolve_split(cfg["data"], "fineweb", "val")))
    assert _control_ids(raw["idx"], tok) == [tok.encode_special("<|bos|>")]


def test_val_stream_is_reproducible(env):
    """Two fresh val loaders must yield byte-identical batches.

    Curves are read point-to-point; if evaluation resampled different data each
    time, wiggles would be data noise masquerading as model change.
    """
    cfg, toks, _ = env
    data_cfg = cfg["data"]
    factory = make_loader_factory(data_cfg, toks, SOURCE_TYPES,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(files=resolve_split(data_cfg, "fineweb", "val"),
              recipe=data_cfg["recipes"]["text_pretrain"],
              recipe_name="text_pretrain", B=4, T=512)
    a = next(factory(**kw))
    b = next(factory(**kw))
    assert torch.equal(a["idx"], b["idx"])
    assert torch.equal(a["targets"], b["targets"])
