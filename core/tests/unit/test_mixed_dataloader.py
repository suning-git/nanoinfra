"""MixedDataLoader = sources -> SampleMixer -> per-sample supervision -> batcher, with the data snapshot rank 0
writes into every checkpoint (interfaces revised 2026-10: sources and snapshots by name, loader records back inside
the batch, the delivery check; one mapping name -> (source, weight)). CPU only.
"""
import gc
import weakref
from functools import partial

import pytest
import torch

from core.data import MixedDataLoader, PackingBatcher, PaddingBatcher, build_NTP_supervised_sample
from core.data.layout import collate_samples, padding_layout, sample_length
from core.data.mixed_dataloader import compute_sample_mixing_weights
from core.data.supervision import IGNORE

PADS = {"idx": 0, "targets": IGNORE}


class CountingSource:
    """Deterministic indexable source: sample k has length lengths[k % len] and tokens base + k; state = {"index"}.
    It edits ONE state dict in place (the case a loader must not leak into handed-out snapshots)."""

    def __init__(self, base, lengths=(5,), budget=None):
        self.base, self.lengths, self.budget = base, list(lengths), budget
        self.state = {"index": 0}

    def __iter__(self):
        while True:
            k = self.state["index"]
            self.state["index"] = k + 1                      # in place
            n = self.lengths[k % len(self.lengths)]
            yield {"tokens": torch.full((n + 1,), self.base + k, dtype=torch.long), "meta": {"k": k, "src": self.base}}

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state                                   # keeps the dict it is given, and edits it later

    def budget_tokens(self):
        return self.budget


def padding(B, T):
    return partial(PaddingBatcher, shape=(B, T), pad_values=PADS)


def packing(B, T):
    return partial(PackingBatcher, shape=(B, T), pad_values=PADS)


def loader_of(sources, weights=None, *, batcher=None, fn=build_NTP_supervised_sample, **kw):
    weights = weights or {name: 1.0 for name in sources}
    return MixedDataLoader({name: (source, weights[name]) for name, source in sources.items()},
                           build_supervised_sample_fn=fn, create_batcher_fn=batcher or padding(2, 8), **kw)


def ks(batch):
    return [m["k"] for m in batch["sample_meta"]]


# ------------------------------------------------- construction

def test_both_functions_are_required_and_the_old_keywords_are_gone():
    with pytest.raises(TypeError):
        MixedDataLoader({"a": (CountingSource(0), 1.0)}, create_batcher_fn=padding(2, 8))
    with pytest.raises(TypeError):
        MixedDataLoader({"a": (CountingSource(0), 1.0)}, build_supervised_sample_fn=build_NTP_supervised_sample)
    with pytest.raises(TypeError):
        MixedDataLoader({"a": (CountingSource(0), 1.0)}, supervise_sample=build_NTP_supervised_sample,
                        create_batcher=padding(2, 8))


class NothingToPredictSource(CountingSource):
    """Every sample's supervision_mask selects only its first token, so no target survives the shift."""
    def __iter__(self):
        for raw in super().__iter__():
            mask = torch.zeros_like(raw["tokens"], dtype=torch.bool)
            mask[0] = True
            yield {**raw, "supervision_mask": mask}


def test_a_sample_with_nothing_to_predict_is_refused_naming_its_source():
    loader = loader_of({"text": CountingSource(0), "motion": NothingToPredictSource(100)},
                       {"text": 0.0, "motion": 1.0})
    with pytest.raises(ValueError, match=r"data source 'motion' \(pull 1 since this iteration started or the loader was "
                                         r"restored, sample meta \{'k': 0, 'src': 100\}\): no supervised target after the shift"):
        next(iter(loader))


def test_the_loader_mixes_by_the_weights_it_is_given():
    """a 1, b 3 -> a, b, b, b (equal weights would give a, b, a, b)."""
    b = next(iter(loader_of({"a": CountingSource(0), "b": CountingSource(1000)}, {"a": 1.0, "b": 3.0},
                            batcher=padding(4, 8))))
    assert [m["src"] for m in b["sample_meta"]] == [0, 1000, 1000, 1000]


class NeverIterated(CountingSource):
    def __iter__(self):
        raise AssertionError("building a loader must not iterate a source")


def test_one_mapping_of_string_names_to_source_weight_pairs():
    with pytest.raises(TypeError):                      # the two dicts this branch took from 10-12 to 10-21
        MixedDataLoader({"a": CountingSource(0)}, {"a": 1.0}, build_supervised_sample_fn=None,
                        create_batcher_fn=padding(2, 8))
    with pytest.raises(TypeError, match="source 'b': each value is a \\(source, weight\\) pair, not a NeverIterated"):
        MixedDataLoader({"a": (CountingSource(0), 1.0), "b": NeverIterated(1)}, build_supervised_sample_fn=None,
                        create_batcher_fn=padding(2, 8))
    with pytest.raises(ValueError, match="finite, nonnegative and not all zero"):    # the mixer checks it
        loader_of({"a": CountingSource(0)}, {"a": -1.0})
    with pytest.raises(ValueError, match="strings"):
        loader_of({0: CountingSource(0)}, {0: 1.0})


# ------------------------------------------------- batches and the snapshot

def test_padding_batches_and_snapshot_is_the_state_after_the_last_delivered_sample():
    it = iter(loader_of({"a": CountingSource(100, lengths=[5])}))
    b = next(it)
    assert ks(b) == [0, 1] and b["idx"].shape == (2, 8) and "loader_records" not in b
    assert b["targets"][0].tolist() == [100] * 5 + [IGNORE] * 3 and b["doc_id"][0].tolist() == [0] * 5 + [-1] * 3
    assert b["state_dict"] == {"source_states": {"a": {"index": 2}}}
    assert ks(next(it)) == [2, 3] and next(it)["state_dict"] == {"source_states": {"a": {"index": 6}}}


def test_a_pending_pull_does_not_advance_its_source_entry():
    """T=8, lengths 5 then 4: the second sample does not fit the row and is pending; the snapshot must say index 1
    (only sample 0 delivered) although the live source already stands at 2."""
    src = CountingSource(0, lengths=[5, 4])
    made = []                                        # the iteration's batcher, to look at what it holds

    def batcher(stream):
        made.append(packing(1, 8)(stream))
        return made[-1]
    loader = loader_of({"a": src}, batcher=batcher)
    it = iter(loader)
    b = next(it)
    assert ks(b) == [0] and src.state["index"] == 2 and made[0].pending["meta"]["k"] == 1
    assert b["state_dict"] == {"source_states": {"a": {"index": 1}}}
    b2 = next(it)
    assert ks(b2)[0] == 1 and b2["state_dict"]["source_states"]["a"]["index"] == ks(b2)[-1] + 1


def test_an_iteration_left_before_its_end_frees_the_loader_by_reference_counting():
    """An evaluator builds a loader per evaluation and stops after its batches. While the loader held the batcher (which
    holds the sample stream, which holds the loader), that cycle kept the loader and its sources until a full
    collection of Python's cyclic collector, rare in a training process: the text entry grew 8-14 MiB an evaluation
    (2026-10-22). Reference counting alone must free them, with a sample pending in the batcher."""
    src = CountingSource(0, lengths=[5, 4])
    loader = loader_of({"a": src}, batcher=packing(1, 8))
    it = iter(loader)
    next(it)                                         # left with the second sample pending in the batcher
    refs = [weakref.ref(loader), weakref.ref(src)]
    gc.disable()
    try:
        del loader, src, it
        assert [r() for r in refs] == [None, None]
    finally:
        gc.enable()


def test_handed_out_snapshots_never_change():
    src = CountingSource(0)
    it = iter(loader_of({"a": src}))
    first = next(it)["state_dict"]
    frozen = {"source_states": {"a": dict(first["source_states"]["a"])}}
    for _ in range(3):
        next(it)
    src.state["index"] = 999                                # the source edits its live state in place
    assert first == frozen


def test_two_sources_snapshot_by_name():
    it = iter(loader_of({"x": CountingSource(0), "y": CountingSource(1000)}))
    snap = next(it)["state_dict"]
    assert snap == {"source_states": {"x": {"index": 1}, "y": {"index": 1}}}


# ------------------------------------------------- set_state

def test_set_state_resumes_where_the_snapshot_says_by_name():
    """Different states per source, the loader built with the names in the other order: a position-based pairing
    would hand a's state to b."""
    snap = {"source_states": {"a": {"index": 3}, "b": {"index": 9}}}
    loader = loader_of({"b": CountingSource(1000), "a": CountingSource(0)})
    loader.set_state(snap)
    assert loader.mixer.samples_drawn_per_source == {"b": 0, "a": 0}
    got = {m["src"]: m["k"] for m in next(iter(loader))["sample_meta"]}
    assert got == {0: 3, 1000: 9}
    # the restored sources got private copies: running on did not change the snapshot we restored from
    assert snap == {"source_states": {"a": {"index": 3}, "b": {"index": 9}}}


def test_a_snapshot_given_to_set_state_is_copied():
    snap = {"source_states": {"a": {"index": 4}}}
    loader = loader_of({"a": CountingSource(0)})
    loader.set_state(snap)
    snap["source_states"]["a"]["index"] = 99                 # the caller edits what it handed over
    assert loader.get_state() == {"source_states": {"a": {"index": 4}}}


def test_set_state_refuses_the_old_list_format_and_says_how_to_convert_it():
    loader = loader_of({"text": CountingSource(0)})
    for old in ({"source_states": [{"index": 1}]}, {"source_states": [{"index": 1}], "accum": [0.0]}):
        with pytest.raises(ValueError, match="accept_list_snapshot.*core.compatibility.loader_snapshot.*\\['text'\\]"):
            loader.set_state(old)


def test_set_state_refuses_other_names_and_a_foreign_state():
    loader = loader_of({"a": CountingSource(0)})
    with pytest.raises(ValueError, match="only in the snapshot \\['b'\\], only in this loader \\['a'\\]"):
        loader.set_state({"source_states": {"b": {"index": 1}}})
    with pytest.raises(ValueError, match="source_states"):
        loader.set_state({"rows": 64})


def test_a_source_not_yet_delivered_is_at_its_start_and_a_restore_puts_it_back_there():
    """Packing (1, 8), lengths 5 and 4: the first batch delivers a0 while b0 waits, so the snapshot holds b's starting
    state, taken when the loader was built. Restored into the SAME loader, whose b already stands past b0, b goes back
    to its start: b0 is delivered, not skipped."""
    loader = loader_of({"a": CountingSource(0, lengths=[5]), "b": CountingSource(1000, lengths=[4])},
                       batcher=packing(1, 8))
    first = next(iter(loader))
    snap = first["state_dict"]
    assert ks(first) == [0] and snap["source_states"] == {"a": {"index": 1}, "b": {"index": 0}}
    assert loader.pulls["b"] == 1
    loader.set_state(snap)
    it = iter(loader)
    delivered = [(m["src"], m["k"]) for _ in range(3) for m in next(it)["sample_meta"]]
    assert [k for src, k in delivered if src == 1000][0] == 0


class StatelessSource(CountingSource):
    def get_state(self):
        return None


def test_a_source_that_keeps_no_state_is_left_where_it_stands():
    loader = loader_of({"s": StatelessSource(0)})
    snap = next(iter(loader))["state_dict"]
    assert snap == {"source_states": {"s": None}}
    loader.set_state(snap)                                  # nothing to put back: the source stays where it is
    assert ks(next(iter(loader))) == [2, 3]


def test_a_source_not_drawn_since_restore_keeps_its_baseline():
    loader = loader_of({"a": CountingSource(0), "z": CountingSource(500)}, {"a": 1.0, "z": 0.0})
    loader.set_state({"source_states": {"a": {"index": 0}, "z": {"index": 7}}})
    snap = next(iter(loader))["state_dict"]
    assert snap["source_states"] == {"a": {"index": 2}, "z": {"index": 7}}


# ------------------------------------------------- loader records and the delivery check

class Regrouping:
    """A batcher that collates groups of its stream's pulls by position: [[1, 0], [2]] delivers pulls 2 and 1, then
    3."""

    def __init__(self, sample_stream, *, groups):
        self.stream, self.groups = sample_stream, groups

    def __iter__(self):
        it, pulled = iter(self.stream), []
        for group in self.groups:
            while len(pulled) <= max(group):
                pulled.append(next(it))
            samples = [pulled[i] for i in group]
            lengths = [sample_length(s) for s in samples]
            batch = collate_samples(samples, padding_layout(lengths, (len(samples), 8)), shape=(len(samples), 8),
                                    pad_values=PADS)
            yield batch


def regroup(groups, **kw):
    return partial(Regrouping, groups=groups, **kw)


def test_order_inside_a_batch_is_free():
    it = iter(loader_of({"a": CountingSource(0)}, batcher=regroup([[1, 0], [2]])))
    b = next(it)
    assert ks(b) == [1, 0] and b["state_dict"] == {"source_states": {"a": {"index": 2}}}
    assert next(it)["state_dict"] == {"source_states": {"a": {"index": 3}}}


def test_delivering_a_later_pull_while_an_earlier_one_is_held_back_is_refused():
    it = iter(loader_of({"a": CountingSource(0)}, batcher=regroup([[1], [0]])))
    with pytest.raises(ValueError, match="before any of its pulls was delivered, .*missing \\[1\\]"):
        next(it)


def test_delivering_a_pull_twice_is_refused():
    it = iter(loader_of({"a": CountingSource(0)}, batcher=regroup([[0], [0]])))
    next(it)
    with pytest.raises(ValueError, match="delivered before \\[1\\]"):
        next(it)
    it = iter(loader_of({"a": CountingSource(0)}, batcher=regroup([[0, 0]])))
    with pytest.raises(ValueError, match="repeated in this batch \\[1\\]"):
        next(it)


def test_a_refused_batch_moves_no_source():
    """Source x's records pass, y's do not: neither snapshot may move."""
    # the mixer alternates x, y (equal weights, ties in insertion order): positions 0..5 are x1 y1 x2 y2 x3 y3
    loader = loader_of({"x": CountingSource(0), "y": CountingSource(1000)}, batcher=regroup([[0, 1], [2, 5]]))
    it = iter(loader)
    first = next(it)["state_dict"]
    with pytest.raises(ValueError, match="source 'y'.*missing \\[2\\]"):
        next(it)                                            # x2 is fine, y3 comes while y2 is held back
    assert loader.get_state() == first and loader._delivered_pulls == {"x": 1, "y": 1}
    assert loader._delivered == first["source_states"]      # x's state did not move on its own either


def test_collate_without_records_adds_no_records_key():
    plain = build_NTP_supervised_sample(torch.arange(1, 5))
    batch = collate_samples([plain, plain], padding_layout([3, 3], (2, 8)), shape=(2, 8), pad_values=PADS)
    assert "loader_records" not in batch                    # collate is usable without a loader


def test_pull_numbers_start_again_on_every_iter_and_every_restore():
    loader = loader_of({"a": CountingSource(0)})
    next(iter(loader))
    assert loader.pulls == {"a": 2}
    next(iter(loader))                                      # a new stream: counted from 1 again, the check passes
    assert loader.pulls == {"a": 2}
    loader.set_state({"source_states": {"a": {"index": 10}}})
    assert loader.pulls == {"a": 0}
    assert ks(next(iter(loader))) == [10, 11]


# ------------------------------------------------- what passes through untouched

def test_prepared_samples_pass_through_with_no_function_and_batches_move_to_device():
    class Prepared(CountingSource):
        def __iter__(self):
            for raw in super().__iter__():
                yield build_NTP_supervised_sample(raw)
    b = next(iter(loader_of({"a": Prepared(3)}, batcher=padding(1, 8), fn=None, device="cpu")))
    assert b["idx"].device.type == "cpu" and isinstance(b["sample_meta"], list) and ks(b) == [0]


def test_the_projects_meta_reaches_the_batch_untouched():
    """The loader's record must not overwrite or remove a project's meta, even keys with record-like names."""
    class Tagged(CountingSource):
        def __iter__(self):
            for raw in super().__iter__():
                raw["meta"].update({"_source": "mine", "_pull": -1, "_state_after": "keep"})
                yield raw
    b = next(iter(loader_of({"a": Tagged(0)})))
    assert b["sample_meta"] == [{"k": 0, "src": 0, "_source": "mine", "_pull": -1, "_state_after": "keep"},
                                {"k": 1, "src": 0, "_source": "mine", "_pull": -1, "_state_after": "keep"}]
    assert b["state_dict"] == {"source_states": {"a": {"index": 2}}}


def test_one_cached_sample_drawn_twice_is_two_occurrences():
    cached = build_NTP_supervised_sample(torch.arange(1, 5))

    class Same:
        def __iter__(self):
            while True:
                yield cached
        def get_state(self):
            return None
        def set_state(self, s):
            pass
    b = next(iter(loader_of({"same": Same()}, batcher=packing(1, 8), fn=None)))
    assert b["doc_id"][0].tolist() == [0, 0, 0, 1, 1, 1, -1, -1]
    assert cached["meta"] == {} and "loader_record" not in cached


def test_sample_weights_from_config_keep_the_old_row_rule():
    src = lambda budget: CountingSource(0, budget=budget)   # noqa: E731
    w, total = compute_sample_mixing_weights([{"type": "a"}, {"type": "b"}], [src(300), src(100)])
    assert w == [0.75, 0.25] and total == 400
    w, total = compute_sample_mixing_weights([{"type": "a", "weight": 2}, {"type": "b", "weight": 6}],
                                             [src(None), src(None)])
    assert w == [0.25, 0.75] and total is None
    w, total = compute_sample_mixing_weights([{"type": "a", "weight": 0.2}, {"type": "b"}], [src(None), src(800)])
    assert w == [0.2, 0.8] and total == 1000
    with pytest.raises(ValueError, match="ambiguous"):
        compute_sample_mixing_weights([{"type": "a", "weight": 0.5}], [src(10)])


# ------------------------------------------------- from_config

class ConfigSource(CountingSource):
    """A CountingSource built the way from_config builds sources: (its config minus type, weight and name)."""

    def __init__(self, config, tokenizers):
        assert not {"type", "weight", "name"} & set(config)
        super().__init__(config["base"])


def from_config(loader_config=None, **kw):
    lc = loader_config or {"data": {"sources": [{"type": "counting", "weight": 1.0, "base": 0}]}}
    kw.setdefault("create_batcher_fn", padding(3, 7))
    return MixedDataLoader.from_config(lc, {}, {"counting": ConfigSource},
                                       build_supervised_sample_fn=build_NTP_supervised_sample, **kw)


def test_from_config_takes_the_batch_shape_from_create_batcher_fn_only():
    b = next(iter(from_config()))
    assert tuple(b["idx"].shape) == (3, 7)
    with pytest.raises(TypeError):
        MixedDataLoader.from_config({"data": {"sources": []}}, {}, {"counting": ConfigSource},
                                    build_supervised_sample_fn=build_NTP_supervised_sample)
    for retired in ({"pad_values": PADS}, {"batch_size": 3}, {"row_capacity": 7}):
        with pytest.raises(TypeError):
            from_config(**retired)


def test_from_config_names_sources_by_name_or_by_their_only_type():
    b = next(iter(from_config()))
    assert set(b["state_dict"]["source_states"]) == {"counting"}
    lc = {"data": {"sources": [{"type": "counting", "name": "low", "weight": 1.0, "base": 0},
                               {"type": "counting", "name": "high", "weight": 1.0, "base": 500}]}}
    b = next(iter(from_config(lc)))
    assert set(b["state_dict"]["source_states"]) == {"low", "high"}


def test_from_config_pairs_weights_with_sources_by_name():
    """Unequal weights, names in the reverse of alphabetical order: low 0.25, high 0.75 -> low, high, high, high."""
    lc = {"data": {"sources": [{"type": "counting", "name": "low", "weight": 0.25, "base": 0},
                               {"type": "counting", "name": "high", "weight": 0.75, "base": 500}]}}
    b = next(iter(from_config(lc, create_batcher_fn=padding(4, 8))))
    assert [m["src"] for m in b["sample_meta"]] == [0, 500, 500, 500]


def test_from_config_needs_a_name_whenever_a_type_repeats():
    lc = {"data": {"sources": [{"type": "counting", "name": "low", "weight": 1.0, "base": 0},
                               {"type": "counting", "weight": 1.0, "base": 500}]}}
    with pytest.raises(ValueError, match="several sources of type \\['counting'\\]"):
        from_config(lc)


def test_from_config_refuses_repeated_names_before_building_a_dict():
    lc = {"data": {"sources": [{"type": "counting", "name": "same", "weight": 1.0, "base": 0},
                               {"type": "counting", "name": "same", "weight": 1.0, "base": 500}]}}
    with pytest.raises(ValueError, match="repeats source names \\('same' at entries \\[0, 1\\]\\)"):
        from_config(lc)


def test_from_config_refuses_the_retired_batch_size_key():
    lc = {"batch_size": 3, "data": {"sources": [{"type": "counting", "weight": 1.0, "base": 0}]}}
    with pytest.raises(ValueError, match="not read"):
        from_config(lc)
