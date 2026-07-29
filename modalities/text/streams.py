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
the way ``projects/frontier_arch``'s ``--raw-ruler`` migration bridge does.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.data.mixed_dataloader import MixedDataLoader

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


def resolve_sources(data_cfg, sequence_len, device='cuda') -> List[Dict[str, Any]]:
    """Training source declarations -> concrete source configs (files + recipe resolved)."""
    out = []
    for sc in data_cfg['sources']:
        sc = dict(sc)
        sc.setdefault('sequence_len', sequence_len)
        sc.setdefault('device', device)
        if 'files' not in sc:
            sc['files'] = resolve_split(data_cfg, sc['dataset'], sc['split'])
        sc['recipe_name'] = sc.get('recipe')
        sc['recipe'] = _recipe(data_cfg, sc.get('recipe'))
        out.append(sc)
    return out


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
            'sequence_len': T, 'device': device,
            'buffer_batch_size': buffer_batch_size,
        }
        dl = MixedDataLoader(
            loader_config={'batch_size': B, 'data': {'sequence_len': T, 'sources': [src]}},
            tokenizers=tokenizers, source_types=source_types, resume_state_dict=None,
        )
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
