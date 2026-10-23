"""
Stream assembly for the text modality — the ONE place that turns declarations
into loaders, for training and evaluation alike.

The point of this module is structural: a training source and its validation
counterpart are built from the same declaration (dataset + split + recipe) by
the same code, so they cannot silently disagree. Previously training went
through ``TextDataSource`` (recipe-assembled) while evaluation called
``token_data_loader`` directly (raw packing), and nothing connected the two.

Config shape::

    data:
      datasets: {fineweb: {splits: {val: {files: [...]}, train: {rest: true}}}}
      recipes:  {text_pretrain: {template: [bos, text_start, text_tokens, text_end, eos]}}
      sources:  [{type: text, dataset: fineweb, split: train, recipe: text_pretrain, ...}]

    evaluation:
      streams:
        - {name: text, dataset: fineweb, split: val, recipe: text_pretrain, metric: val/text_ce}

``recipe: null`` is legal on a stream and means "assemble nothing" (raw packing).
It is never the default and never silent — a caller that wants it declares it,
the way a project-side ``--raw-ruler`` migration bridge does.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List

import torch

from core.data.batchers import PaddingBatcher
from core.data.mixed_dataloader import MixedDataLoader
from core.data.supervision import IGNORE, build_NTP_supervised_sample

from modalities.text.datasets import describe, resolve_split
from modalities.text.evaluator import TextEvaluator


def _recipe(data_cfg: Dict[str, Any], name):
    """Look up a NAMED recipe. ``None`` is legal but must be written down."""
    if name is None:
        return None
    recipes = data_cfg.get('recipes') or {}
    if name not in recipes:
        raise KeyError(f"unknown recipe {name!r}; declared: {sorted(recipes)}")
    return recipes[name]


def resolve_sources(data_cfg, sequence_len, device='cpu') -> List[Dict[str, Any]]:
    """Training source declarations -> concrete source configs (files + recipe resolved). `sequence_len` is T, the
    positions per row; each source reads `sample_len` raw tokens per window, T + 1 unless it declares its own."""
    out = []
    for sc in data_cfg['sources']:
        sc = dict(sc)
        if 'sequence_len' in sc:
            raise KeyError("a text source declares `sequence_len`; it is now `sample_len` (raw tokens per window, "
                           "default sequence_len + 1)")
        sc.setdefault('sample_len', sequence_len + 1)
        sc.setdefault('device', device)
        if 'files' not in sc:
            sc['files'] = resolve_split(data_cfg, sc['dataset'], sc['split'])
        sc['recipe_name'] = sc.get('recipe')
        sc['recipe'] = _recipe(data_cfg, sc.get('recipe'))
        out.append(sc)
    return out


def ntp_pad_values(layout, control_resolver) -> Dict[str, int]:
    """Pad values for next-token-prediction samples of the assembled vocab: idx is padded with `eos` (a real
    token, as the motion sources always padded), its type with eos's type, targets with IGNORE. Text windows are
    always full, so they meet these only in a finite stream's tail batch; a pad position is never a target and
    never read by a real position (doc_id = -1 marks it)."""
    eos = control_resolver.resolve('eos')
    eos_type = int(layout.classify_token_types(torch.tensor([eos]))[0])
    return {'idx': eos, 'targets': IGNORE, 'token_types': eos_type, 'target_types': eos_type}


def make_text_loader(loader_config, tokenizers, source_types, *, batch_size, sequence_len, device='cuda'):
    """The text pipeline's MixedDataLoader: sources on the CPU (their configs say so), each raw window prepared by
    build_NTP_supervised_sample (sample_len raw tokens -> sample_len - 1 positions), one window per row
    (PaddingBatcher, batch_size rows of `sequence_len` = T positions), whole batches moved to `device`."""
    return MixedDataLoader.from_config(
        loader_config, tokenizers, source_types, build_supervised_sample_fn=build_NTP_supervised_sample,
        create_batcher_fn=partial(PaddingBatcher, shape=(batch_size, sequence_len),
                                  pad_values=ntp_pad_values(tokenizers['layout'], tokenizers['control_resolver'])),
        device=device)


def make_loader_factory(data_cfg, tokenizers, source_types, device='cuda',
                        buffer_batch_size=32):
    """Return ``callable(files, recipe, recipe_name, B, T) -> fresh batch iterator``.

    Evaluation streams use this so they are assembled by exactly the code that
    assembles training batches.
    """
    def factory(files, recipe, recipe_name, B, T):
        src = {
            'type': 'text', 'files': files, 'recipe': recipe,
            'recipe_name': recipe_name, 'split': 'val', 'weight': 1.0,
            'sample_len': T + 1, 'device': 'cpu',               # rows of T positions, as in training
            'buffer_batch_size': buffer_batch_size,
        }
        dl = make_text_loader({'data': {'sources': [src]}}, tokenizers, source_types, batch_size=B,
                              sequence_len=T, device=device)
        return iter(dl)
    return factory


def build_evaluators(config, tokenizers, source_types, device_batch_size,
                     sequence_len, device='cuda') -> List[TextEvaluator]:
    """Declared evaluation streams -> TextEvaluator list (empty if disabled)."""
    data_cfg = config['data']
    eval_cfg = dict(config.get('evaluation') or {})
    if not eval_cfg.get('enabled', True):
        return []
    streams = eval_cfg.get('streams')
    if not streams:
        raise KeyError(
            "evaluation.streams is not declared. Evaluation is a LIST of declared "
            "streams now (see modalities/text/streams.py); there is no implicit "
            "'the val set'."
        )
    factory = make_loader_factory(data_cfg, tokenizers, source_types, device=device)
    evaluators = []
    for st in streams:
        st = dict(st)
        if 'files' not in st:
            st['files'] = resolve_split(data_cfg, st['dataset'], st['split'])
        st['recipe_name'] = st.get('recipe') or 'raw'
        st['recipe'] = _recipe(data_cfg, st.get('recipe'))
        evaluators.append(TextEvaluator(st, eval_cfg, device_batch_size,
                                        sequence_len, loader_factory=factory))
    return evaluators


# FineWeb sample-10BT under this repo's tokenizer: the six documented shards hold
# 6,275,704 rows and ~4.36B tokens (README "Data"), i.e. ~695 tokens per row. It
# only has to turn a shard count into an epoch ratio, so a 10% error in it changes
# nothing about the judgement it supports.
TOKENS_PER_ROW = 695


def _si(n):
    """1.23B / 45.6M / 789K — a short run's budget must not print as 0.00B."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n/div:.2f}{suf}"
    return f"{n:.0f}"


def _estimate_tokens(path):
    """Rough token count for one FineWeb shard, from its footer only (no decode)."""
    if not str(path).endswith(".parquet"):
        return None
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows * TOKENS_PER_ROW
    except Exception:
        return None


def report_token_supply(config, sources, max_steps, printer=print) -> None:
    """Say how many epochs the run will actually take over the data on disk.

    Another ruler that did not speak. The val split is declared by name and train
    is `rest: true`, so the TRAIN set size is an emergent property of whatever
    else is in base_data/ — declaring the ruler does not declare how much corpus
    sits behind it. Download three shards instead of six and a single-epoch run
    quietly becomes ~1.9 epochs: nothing errors, the loader simply wraps, and
    every number afterwards is measured under conditions the docs do not
    describe. That is the expensive kind of wrong — silent and hard to attribute.

    Must be called AFTER the Trainer exists: with the default `max_steps: -1`
    the budget is Chinchilla-derived and only resolved in Trainer.__init__.

    Says nothing rather than guessing whenever it cannot be sure: a MIXTURE of
    sources (each drawn at its own weight, so a pooled ratio is wrong for every
    source in it), a corpus other than the one the per-row constant was measured
    on, or a shard whose footer will not read. The count is an estimate anyway —
    rows x a measured average — and labelled as one, because 1.9 vs 1.0 is the
    judgement being supported and it does not need three digits.
    """
    if len(sources) != 1:
        return
    # TOKENS_PER_ROW is calibrated on FineWeb. Another parquet corpus with a
    # different document length would be mis-sized by the same silent factor this
    # function exists to expose, so it only speaks for the corpus it was measured
    # on. Re-measure and widen the check deliberately, never by inheritance.
    if sources[0].get("dataset") != "fineweb":
        return
    supply = 0
    for path in (sources[0].get("files") or []):
        n = _estimate_tokens(path)
        if n is None:
            return
        supply += n
    if not supply:
        return
    demand = max_steps * config["total_batch_size"]
    epochs = demand / supply
    printer("\n--- token supply (estimated) ---")
    printer(f"  train tokens: need ~{_si(demand)}, have ~{_si(supply)} on disk "
            f"-> {epochs:.2f} epoch(s)")
    if epochs > 1.05:
        printer(f"  NOTE: this run repeats the training data {epochs:.2f}x. If you expected a "
                f"single epoch you are short of shards — see the \"Data\" section of "
                f"exemplars/text_pretrain/README.md, and its data/download_shards.py.")
    printer("--------------------------------\n")


def report(config, sources, evaluators, printer=print) -> None:
    """Print training and evaluation rulers SIDE BY SIDE.

    This bug survived for months because only the training side ever spoke in
    the log. Both sides speak now, and every split prints its fingerprint.
    """
    data_cfg = config['data']
    printer("\n--- data rulers (train vs eval) ---")
    for sc in sources:
        printer(f"  train source [{sc.get('dataset','?')}/{sc.get('split','?')}] "
                f"recipe={sc.get('recipe_name')}")
        if sc.get('dataset'):
            printer(f"    {describe(data_cfg, sc['dataset'], sc['split'], sc.get('files'))}")
    for ev in evaluators:
        printer(ev.describe())
    printer("-----------------------------------\n")
