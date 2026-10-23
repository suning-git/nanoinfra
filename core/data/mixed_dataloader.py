"""
MixedDataLoader: the complete loader entry — sources -> SampleMixer -> per-sample supervision -> batcher -> batches.

Sources and snapshots are keyed by source name; the loader's records come back inside the batch for a delivery
check. The parts are core components a project can also wire itself (SampleMixer, build_NTP_supervised_sample,
PaddingBatcher / PackingBatcher, layout + collate); this class is the ordinary way to use them together.

Two ways in, one pipeline:

    # direct: existing sources and parts, source name -> (source, sample weight)
    loader = MixedDataLoader({"t2m": (t2m_source, 1.0)},
                             build_supervised_sample_fn=build_NTP_supervised_sample,
                             create_batcher_fn=partial(PackingBatcher, shape=(B, T), pad_values=pad_values),
                             device="cuda")

    # from the standard config: builds the sources and the sample weights; a source is named by its `name`, and may
    # go without one only when it is the only source of its type (it is then named by the type)
    loader = MixedDataLoader.from_config(loader_config, tokenizers, source_types,
                                         build_supervised_sample_fn=build_NTP_supervised_sample,
                                         create_batcher_fn=partial(PaddingBatcher, shape=(B, T),
                                                                   pad_values=pad_values),
                                         device="cuda")

build_supervised_sample_fn and create_batcher_fn are required in both entries (omitting one is a TypeError).
build_supervised_sample_fn is build_NTP_supervised_sample, a project function, or None when the sources already
yield supervised {data, meta} samples. create_batcher_fn carries the batch shape (B rows of T positions, T the
model's row length) and the pad values: they are written there once and nowhere else.

Trainer interface (duck typing): iter(loader) -> batches {data fields [B, T, ...], doc_id, doc_pos, sample_meta,
layout_facts, state_dict}; get_state(); set_state(state). Data fields are moved to `device` after collate, one copy
per field (a CPU-native source is supervised and batched on the CPU first).

Mixing counts samples. from_config derives the sample weights from each source's `weight` in the config
(`auto` = by token budget, or an explicit share; compute_sample_mixing_weights has the rules). A project that mixes
sources of different mean lengths and wants a token ratio converts it to sample weights itself (weight_i ∝ token
share_i / mean sample length_i) and writes those weights into the direct entry's mapping.

Delivery: every pulled sample carries {"source", "pull", "state_after"} under the reserved top-level key
`loader_record` — pull counts from 1 per source, again after every iter(loader) and every restore. Collate gathers
the records into batch["loader_records"]; before a batch goes on, the loader takes the list out and checks that,
per source, the pulls delivered so far are exactly 1..n. Order inside a batch is free; a batcher that delivers a
later sample of a source while holding an earlier one back is refused, because a snapshot taken "after the later
one" would skip the held-back sample on resume. Every record is checked before any source's snapshot moves.

Resume: batch["state_dict"] is this rank's data snapshot {"source_states": {name: state}}, the source's own state
right after its last DELIVERED sample (the state_after of its largest delivered pull). A pending sample does not
advance its source's entry; a source not drawn since a restore keeps the state it was restored to (its get_state()
right after set_state: this rank's share of what the snapshot handed it); a source not yet delivered keeps its
starting state, taken from get_state() when the loader is built. So every entry is a real position, and a restore
puts each source there, also in a loader that has already read. The entry is None only for a source that keeps no
state; a restore leaves it where it stands. Every entry is a private copy, so later pulls or a source editing its
live state cannot change a snapshot already handed out.
The run's snapshot is merge_rank_snapshots of every rank's: each source merges its ranks' states
(merge_rank_states: the furthest rank's position and the items before it no rank has trained), so a resume, on any
number of ranks, neither skips nor repeats a sample of such a source. A source without merge_rank_states keeps rank
0's state, approximate as before 2026-10-22: a rank that consumed more or fewer of its samples than rank 0 replays or
skips the difference. The Trainer hands the snapshot of the last micro-batch of a completed update to the checkpoint
saving, which gathers every rank's and writes the merge (core.model.checkpoint_manager.merge_data_snapshots).
set_state takes the run's snapshot in this name-keyed format only: a list-format snapshot (nanoinfra before 0.3.0:
states matched by position, with or without `accum`) is refused, and core.compatibility.loader_snapshot converts it
on request.
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch

from core.data.data_source import DataSource
from core.data.sample_mixer import SampleMixer
from core.utils import print0

_NOT_TENSORS = ("sample_meta", "state_dict", "layout_facts")


def _describe_batcher(create_batcher_fn) -> str:
    """For the init log: the batcher and its shape when create_batcher_fn is a functools.partial, else its name."""
    target = getattr(create_batcher_fn, 'func', create_batcher_fn)
    shape = getattr(create_batcher_fn, 'keywords', {}).get('shape')
    name = getattr(target, '__name__', repr(target))
    return name if shape is None else f"{name} shape={tuple(shape)}"


def copy_state(x):
    """A private copy of a source state (nested dict/list/tuple of scalars; anything else deep-copied)."""
    if isinstance(x, dict):
        return {k: copy_state(v) for k, v in x.items()}
    if isinstance(x, list):
        return [copy_state(v) for v in x]
    if isinstance(x, tuple):
        return tuple(copy_state(v) for v in x)
    if x is None or isinstance(x, (int, float, str, bool)):
        return x
    return copy.deepcopy(x)


def _name_diff(a, b, a_label, b_label) -> str:
    return f"only in {a_label} {sorted(set(a) - set(b))}, only in {b_label} {sorted(set(b) - set(a))}"


class MixedDataLoader:
    def __init__(self, weighted_sources: Dict[str, Tuple[DataSource, float]], *,
                 build_supervised_sample_fn: Optional[Callable], create_batcher_fn: Callable,
                 device=None):
        self.mixer = SampleMixer(weighted_sources)      # checks the mapping; only the mixer reads the weights
        if any(not isinstance(name, str) for name in self.mixer.sources):
            raise ValueError("source names must be strings: they are the keys of the snapshot written to JSON")
        self.sources = self.mixer.sources               # the sources it draws from, for the snapshot and restore
        self.build_supervised_sample_fn, self.create_batcher_fn = build_supervised_sample_fn, create_batcher_fn
        self.device = None if device is None else torch.device(device)
        # each source's state after its last DELIVERED sample; until then its starting state
        self._delivered = {name: copy_state(source.get_state()) for name, source in self.sources.items()}
        self._reset_counts()
        self._current_state = None
        self._told_approximate = set()          # sources named once as merged approximately

    def _reset_counts(self):
        """Pull numbers start again from 1 (every iter(loader), every restore), and so do the delivery counts."""
        self.pulls = {name: 0 for name in self.sources}              # samples pulled from each source
        self._delivered_pulls = {name: 0 for name in self.sources}   # pulls 1..n of each source delivered so far

    @classmethod
    def from_config(cls, loader_config: Dict[str, Any], tokenizers: Dict[str, Any],
                    source_types: Dict[str, Type[DataSource]], *, build_supervised_sample_fn: Optional[Callable],
                    create_batcher_fn: Callable, device=None) -> "MixedDataLoader":
        """Sources from `loader_config["data"]["sources"]` (each built by source_types[type](config, tokenizers),
        the config without its `type`, `weight` and `name`), sample weights from their `weight` / budget. A source
        is named by its `name`; it may go without one only when it is the only source of its type, and is then
        named by the type. Missing and repeated names are refused here, before the list becomes a dict (a dict
        passed to the direct entry has already lost a repeated key). The batch shape and pad values are the
        orchestrator's, in create_batcher_fn."""
        if 'batch_size' in loader_config:
            raise ValueError("loader_config['batch_size'] is not read: the batch shape lives in create_batcher_fn")
        source_configs = loader_config['data']['sources']
        types = [sc['type'] for sc in source_configs]
        unnamed = sorted({t for sc, t in zip(source_configs, types) if 'name' not in sc and types.count(t) > 1})
        if unnamed:
            raise ValueError(f"data.sources has several sources of type {unnamed} and not all of them have a "
                             f"`name`: a source may go without one only when it is the only source of its type")
        names = [sc.get('name', sc['type']) for sc in source_configs]
        repeated = {n: [i for i, m in enumerate(names) if m == n] for n in names if names.count(n) > 1}
        if repeated:
            where = ", ".join(f"{n!r} at entries {at}" for n, at in sorted(repeated.items()))
            raise ValueError(f"data.sources repeats source names ({where}): give each source its own `name`")
        sources = {}
        for name, sc in zip(names, source_configs):
            if sc['type'] not in source_types:
                raise ValueError(f"Unknown data source type: {sc['type']}. Available: {list(source_types)}")
            sources[name] = source_types[sc['type']](
                {k: v for k, v in sc.items() if k not in ('type', 'weight', 'name')}, tokenizers)
        weight_list, total_budget = compute_sample_mixing_weights(source_configs, list(sources.values()), names)
        weights = dict(zip(names, weight_list))
        loader = cls({name: (sources[name], weights[name]) for name in names},
                     build_supervised_sample_fn=build_supervised_sample_fn, create_batcher_fn=create_batcher_fn,
                     device=device)
        print("MixedDataLoader initialized:")
        print(f"  batcher={_describe_batcher(create_batcher_fn)}, build_supervised_sample_fn="
              f"{getattr(build_supervised_sample_fn, '__name__', build_supervised_sample_fn)}")
        for name, sc in zip(names, source_configs):
            budget = sources[name].budget_tokens()
            budget_str = f", budget={budget:,.0f} tokens" if budget is not None else ""
            print(f"  source {name!r} type={sc['type']}, sample weight={weights[name]:.4f}{budget_str}")
        if total_budget is not None:
            print(f"  total_budget={total_budget:,.0f} tokens")
        return loader

    # ------------------------------------------------------------------ the stream
    def _sample_stream(self):
        for name, raw in self.mixer:
            self.pulls[name] += 1
            state_after = copy_state(self.sources[name].get_state())
            try:
                sample = raw if self.build_supervised_sample_fn is None else self.build_supervised_sample_fn(raw)
            except ValueError as e:                  # the builder knows the rule, the loader knows where the sample came from
                meta = raw.get("meta") if isinstance(raw, dict) else None
                where = f"pull {self.pulls[name]} since this iteration started or the loader was restored" + (
                    f", sample meta {meta!r}" if meta else "")
                raise ValueError(f"data source {name!r} ({where}): {e}") from e
            # a new outer dict per pull: the source or the supervised-sample function may hand out the same object
            # again (one cached sample drawn twice = two occurrences, two doc_ids); data and the project's meta stay
            # as they are — the loader's own record rides in a separate top-level key
            yield {**sample, 'loader_record': {'source': name, 'pull': self.pulls[name], 'state_after': state_after}}

    def __iter__(self):
        self._reset_counts()
        # The batcher is this iteration's and is held here only: its sample stream holds the loader, so the loader
        # holding the batcher made a reference cycle, freed only by a full collection of Python's cyclic collector,
        # rare in a training process. An iteration left before its end (the text evaluator builds a loader per
        # evaluation) then kept the loader, its sources and what they had read: the text entry grew 8-14 MiB an
        # evaluation.
        batcher = self.create_batcher_fn(self._sample_stream())
        for batch in batcher:
            self._deliver(batch.pop('loader_records'), batch)
            batch['state_dict'] = self._current_state
            if self.device is not None:
                for k, v in batch.items():
                    if k not in _NOT_TENSORS and isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)
            yield batch

    def _deliver(self, records, batch) -> None:
        """Check that each source's delivered pulls run 1..n, then move the snapshot; nothing moves on a refusal."""
        by_source = {}
        for record in records:
            by_source.setdefault(record['source'], []).append(record)
        delivered, states = {}, {}
        for name, recs in by_source.items():
            have, pulls = self._delivered_pulls[name], sorted(r['pull'] for r in recs)
            expected = list(range(have + 1, have + len(recs) + 1))
            if pulls != expected:
                repeated = sorted({p for p in pulls if pulls.count(p) > 1})
                before = sorted({p for p in pulls if p <= have})
                missing = sorted(set(range(have + 1, max(pulls) + 1)) - set(pulls))
                so_far = f"after its pulls 1..{have}" if have else "before any of its pulls was delivered"
                raise ValueError(
                    f"source {name!r}: {so_far}, this batch delivers pulls {pulls} (missing {missing}, repeated in "
                    f"this batch {repeated}, delivered before {before}). Delivered pulls must be exactly 1..n: a "
                    f"batcher may reorder samples inside a batch, but must deliver each source's samples once and in "
                    f"pull order across batches, or a resume would skip or repeat them")
            delivered[name] = expected[-1]
            states[name] = max(recs, key=lambda r: r['pull'])['state_after']
        self._delivered_pulls.update(delivered)
        self._delivered.update(states)
        self._current_state = {'source_states': dict(self._delivered)}

    # ------------------------------------------------------------------ checkpoint interface
    def get_state(self) -> Optional[Dict[str, Any]]:
        return self._current_state

    def set_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or 'source_states' not in state:
            raise ValueError(f"unsupported dataloader state: {type(state).__name__} without source_states")
        states = state['source_states']
        if isinstance(states, list):
            raise ValueError(
                f"this data snapshot is in the list format (nanoinfra before 0.3.0): its states are matched to "
                f"sources by position, which cannot be checked, so name the source of each state in the list's "
                f"order. Either wrap this loader before the Trainer restores it, leaving the checkpoint as it is: "
                f"`from core import compatibility; compatibility.accept_list_snapshot(loader, source_order=[...])` "
                f"(source_order may be left out when the loader has one source); or rewrite the checkpoint's "
                f"meta.json once, keeping a backup: `python -m core.compatibility.loader_snapshot <checkpoint_dir> "
                f"<source names in the list's order>`. This loader's sources: {list(self.sources)}.")
        if not isinstance(states, dict):
            raise ValueError(f"source_states must be a dict keyed by source name, not {type(states).__name__}")
        if set(states) != set(self.sources):
            raise ValueError(f"the snapshot's sources and this loader's differ: "
                             f"{_name_diff(states, self.sources, 'the snapshot', 'this loader')}")
        for name, ss in states.items():
            if ss is not None:                                 # None: a source that keeps no state, left as it is
                self.sources[name].set_state(copy_state(ss))   # a private copy: a source may edit what it is given
                # this rank's entry is what the source took: its share of the snapshot, not the snapshot itself
                self._delivered[name] = copy_state(self.sources[name].get_state())
            else:
                self._delivered[name] = None
        # mixing counts are never saved: a restore starts a new phase, pull numbers start again from 1
        self.mixer.reset_phase()
        self._reset_counts()
        self._current_state = {'source_states': dict(self._delivered)}

    def merge_rank_snapshots(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The run's data snapshot from every rank's, in rank order (the same loader on every rank): each source's
        states merged by its merge_rank_states, or rank 0's (snapshots[0]) for a source without one, approximate and
        said once per source when there is more than one rank. Pure: reads only the snapshots and the sources' merge
        functions, never a source's live state."""
        merged = {}
        for name, source in self.sources.items():
            states = [snap['source_states'][name] for snap in snapshots]
            merge = getattr(source, 'merge_rank_states', None)
            if all(st is None for st in states):
                merged[name] = None
            elif merge is not None:
                merged[name] = merge([copy_state(st) for st in states])
            else:
                if len(states) > 1 and name not in self._told_approximate:
                    print0(f"MixedDataLoader: source {name!r} ({type(source).__name__}) has no merge_rank_states: "
                          f"the checkpoint keeps rank 0's state, and a resume may skip or repeat samples a rank "
                          f"consumed more or fewer of than rank 0")
                    self._told_approximate.add(name)
                merged[name] = copy_state(states[0])
        return {'source_states': merged}

    def __repr__(self) -> str:
        """Compact repr for checkpoint logging; each source renders its own state."""
        return f"MixedDataLoader state: {', '.join(f'{n}={s!r}' for n, s in self.sources.items())}"


def compute_sample_mixing_weights(source_configs: List[Dict], sources: List[DataSource],
                                  names: Optional[List[str]] = None):
    """Per-source sample weights from config. Returns (weights summing to 1, the total token budget or None).

        all `weight: auto`:      proportional to each source's token budget; total = sum of budgets
        all explicit:            normalized proportions; no total budget
        both:                    explicit weights are absolute shares (sum < 1); auto sources share the rest by
                                 budget; total = auto budgets / (1 - explicit sum)
    """
    n = len(source_configs)
    names = names or [f"#{i}" for i in range(n)]       # from_config passes the source names, for the messages
    auto_entries, explicit_entries = [], []
    for i, (source, sc) in enumerate(zip(sources, source_configs)):
        w = sc.get('weight', 'auto')
        if w == 'auto':
            budget = source.budget_tokens()
            if budget is None:
                raise ValueError(f"source {names[i]!r} has weight=auto but no budget (its budget_tokens() is None; "
                                 f"the shipped sources read it from 'tokens')")
            auto_entries.append((i, budget))
        else:
            w = float(w)
            if source.budget_tokens() is not None:
                raise ValueError(f"source {names[i]!r} has explicit weight={w} and a token budget — "
                                 f"this is ambiguous. Use weight=auto to derive the weight from the budget, or "
                                 f"remove the budget ('tokens') for an explicit weight.")
            explicit_entries.append((i, w))
    weights = [0.0] * n
    if auto_entries and explicit_entries:
        explicit_sum = sum(w for _, w in explicit_entries)
        if explicit_sum >= 1.0:
            raise ValueError(f"Sum of explicit weights ({explicit_sum:.4f}) must be < 1.0 to leave room for auto "
                             f"sources")
        for i, w in explicit_entries:
            if w < 0:
                raise ValueError(f"source {names[i]!r}: explicit weight must be >= 0, got {w}")
            weights[i] = w
        auto_remaining = 1.0 - explicit_sum
        auto_budget_sum = sum(b for _, b in auto_entries)
        if auto_budget_sum > 0:
            for i, budget in auto_entries:
                weights[i] = auto_remaining * (budget / auto_budget_sum)
        total_budget = int(auto_budget_sum / auto_remaining) if auto_remaining > 0 else None
    elif auto_entries:
        auto_budget_sum = sum(b for _, b in auto_entries)
        if auto_budget_sum > 0:
            for i, budget in auto_entries:
                weights[i] = budget / auto_budget_sum
        total_budget = auto_budget_sum
    else:
        raw_sum = sum(w for _, w in explicit_entries)
        if raw_sum <= 0:
            raise ValueError("Sum of explicit weights must be > 0")
        for i, w in explicit_entries:
            weights[i] = w / raw_sum
        total_budget = None
    return weights, total_budget
