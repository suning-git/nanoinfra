"""The data pipeline's parts: the per-sample NTP step, the sample mixer, the two layouts + collate and the two
batchers, imported from core.

CPU only. It checks: both NTP input forms share one semantics; the supervision_mask shift; the legacy
protocol refused; inputs never modified; both batchers used the same way; random finite streams come out with no
sample lost, duplicated or reordered, over several rows.
"""
import random

import pytest
import torch

from core.data import (PackingBatcher, PaddingBatcher, SampleMixer, build_NTP_supervised_sample, collate_samples,
                       padding_layout, sequential_layout)
from core.data.supervision import IGNORE

PAD, TYPE_PAD = 0, 0
PADS = {"idx": PAD, "targets": IGNORE, "token_types": TYPE_PAD, "target_types": TYPE_PAD}


def toks(*v):
    return torch.tensor(v, dtype=torch.long)


# ----------------------------------------------------------------------------------------------- NTP, one sample
def test_ntp_shift_with_supervision_mask():
    """The worked example: supervise only a2, a3."""
    tokens = toks(10, 11, 12, 13)
    s = build_NTP_supervised_sample(tokens, torch.tensor([False, False, True, True]))
    assert s["data"]["idx"].tolist() == [10, 11, 12]
    assert s["data"]["targets"].tolist() == [IGNORE, 12, 13]
    assert s["meta"] == {}


@pytest.mark.parametrize("mask", [[False, False, False, False], [True, False, False, False]])
def test_ntp_refuses_a_sample_with_no_supervised_target(mask):
    """Nothing selected, or only the first token (which no position predicts): nothing left after the shift."""
    with pytest.raises(ValueError, match="no supervised target after the shift"):
        build_NTP_supervised_sample(toks(10, 11, 12, 13), torch.tensor(mask))


def test_ntp_without_mask_supervises_every_predictable_token():
    s = build_NTP_supervised_sample(toks(1, 2, 3))
    assert s["data"]["idx"].tolist() == [1, 2] and s["data"]["targets"].tolist() == [2, 3]


def test_ntp_two_input_forms_agree_and_shift_types():
    tokens, types = toks(5, 6, 7, 8, 9), toks(0, 0, 2, 0, 0)
    mask, meta = torch.tensor([True, False, True, True, False]), {"sample_id": ("fineweb", 3)}
    a = build_NTP_supervised_sample(tokens, mask, token_types=types, meta=meta)
    b = build_NTP_supervised_sample({"tokens": tokens, "supervision_mask": mask, "token_types": types, "meta": meta})
    assert a["data"].keys() == b["data"].keys() == {"idx", "targets", "token_types", "target_types"}
    for k in a["data"]:
        assert torch.equal(a["data"][k], b["data"][k])
    assert a["meta"] == b["meta"] == meta and a["meta"] is not meta
    assert a["data"]["token_types"].tolist() == [0, 0, 2, 0] and a["data"]["target_types"].tolist() == [0, 2, 0, 0]
    assert a["data"]["targets"].tolist() == [IGNORE, 7, 8, IGNORE]


def test_ntp_does_not_modify_inputs():
    tokens, mask = toks(1, 2, 3, 4), torch.tensor([True, False, False, True])
    t0, m0 = tokens.clone(), mask.clone()
    s = build_NTP_supervised_sample(tokens, mask)
    s["data"]["targets"].fill_(99)             # a rewritten target is a fresh tensor
    assert torch.equal(tokens, t0) and torch.equal(mask, m0)


@pytest.mark.parametrize("field", ["loss_mask", "loss_weights", "attention_mask"])
def test_ntp_raw_form_refuses_legacy_fields(field):
    with pytest.raises(ValueError, match="legacy"):
        build_NTP_supervised_sample({"tokens": toks(1, 2, 3), field: torch.ones(3)})


@pytest.mark.parametrize("bad", [
    dict(tokens=toks(1)),                                                   # fewer than two tokens
    dict(tokens=torch.tensor([1.0, 2.0])),                                  # not integer
    dict(tokens=toks(1, 2, 3), supervision_mask=torch.tensor([1, 0, 1])),   # mask not bool
    dict(tokens=toks(1, 2, 3), supervision_mask=torch.tensor([True, False])),  # mask misaligned
    dict(tokens=toks(1, 2, 3), token_types=toks(0, 0)),                    # types misaligned
])
def test_ntp_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        build_NTP_supervised_sample(bad.pop("tokens"), **bad)


def test_ntp_raw_form_refuses_a_field_from_two_sources():
    with pytest.raises(ValueError, match="two sources"):
        build_NTP_supervised_sample({"tokens": toks(1, 2, 3)}, supervision_mask=torch.tensor([True] * 3))


# ----------------------------------------------------------------------------------------------- layout + collate
def sample(n, start=0, sid=None, extra=None):
    t = torch.arange(start, start + n + 1, dtype=torch.long) + 1          # raw tokens, never PAD
    s = build_NTP_supervised_sample(t, token_types=torch.zeros(n + 1, dtype=torch.long),
                                    meta={"_id": sid} if sid is not None else None)
    if extra:
        s["data"].update(extra(n))
    return s


def test_layouts():
    assert padding_layout([3, 2], (2, 6)) == [(0, 0, 0), (1, 1, 0)]
    assert sequential_layout([3, 2, 4, 1], (2, 6)) == [(0, 0, 0), (1, 0, 3), (2, 1, 0), (3, 1, 4)]
    with pytest.raises(ValueError):
        sequential_layout([4, 4, 4], (1, 6))        # exceeds capacity
    with pytest.raises(ValueError):
        padding_layout([7], (1, 6))                  # longer than a row


def test_collate_design_example():
    """Lengths 3 and 2 into [1, 6]."""
    a, b = sample(3, 0, "A"), sample(2, 10, "B")
    batch = collate_samples([a, b], sequential_layout([3, 2], (1, 6)), shape=(1, 6), pad_values=PADS)
    assert batch["idx"].tolist() == [[1, 2, 3, 11, 12, PAD]]
    assert batch["targets"].tolist() == [[2, 3, 4, 12, 13, IGNORE]]
    assert batch["doc_id"].tolist() == [[0, 0, 0, 1, 1, -1]]
    assert batch["doc_pos"].tolist() == [[0, 1, 2, 0, 1, 0]]
    assert batch["sample_meta"] == [{"_id": "A"}, {"_id": "B"}]
    assert batch["layout_facts"]["one_sample_per_row_left_aligned"] is False
    assert batch["doc_id"].dtype == batch["doc_pos"].dtype == torch.long


def test_collate_moves_every_field_including_trailing_dims_and_does_not_touch_inputs():
    feat = lambda n: {"feat": torch.arange(n * 3, dtype=torch.float32).view(n, 3)}   # noqa: E731
    samples = [sample(2, 0, 0, feat), sample(3, 5, 1, feat)]
    before = [{k: v.clone() for k, v in s["data"].items()} for s in samples]
    pads = {**PADS, "feat": -1.0}
    batch = collate_samples(samples, padding_layout([2, 3], (2, 4)), shape=(2, 4), pad_values=pads)
    assert batch["feat"].shape == (2, 4, 3)
    assert torch.equal(batch["feat"][1, :3], samples[1]["data"]["feat"]) and (batch["feat"][0, 2:] == -1).all()
    assert batch["layout_facts"]["one_sample_per_row_left_aligned"] is True
    batch["idx"].fill_(7)
    for s, b in zip(samples, before):
        for k in b:
            assert torch.equal(s["data"][k], b[k])


def test_collate_refusals():
    a, b = sample(2, 0), sample(2, 5)
    with pytest.raises(ValueError, match="pad_values"):
        collate_samples([a], [(0, 0, 0)], shape=(1, 4), pad_values={"idx": PAD, "targets": IGNORE})
    bad = sample(2)
    bad["data"]["doc_id"] = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="reserved"):
        collate_samples([bad], [(0, 0, 0)], shape=(1, 4), pad_values={**PADS, "doc_id": 0})
    for placements in ([(0, 0, 0), (1, 1, 0), (0, 0, 2)], [(0, 0, 0), (1, 0, 1)], [(0, 0, 3), (1, 1, 0)],
                       [(0, 0, 0)]):
        with pytest.raises(ValueError):                               # duplicate, overlap, overflow, missing
            collate_samples([a, b], placements, shape=(2, 4), pad_values=PADS)
    extra = sample(2, 5, extra=lambda n: {"w": torch.ones(n)})    # a field only the second sample carries
    with pytest.raises(ValueError, match="same data keys"):
        collate_samples([a, extra], [(0, 0, 0), (1, 1, 0)], shape=(2, 4), pad_values=PADS)
    f32 = sample(2, extra=lambda n: {"w": torch.zeros(n)})
    i64 = sample(2, 5, extra=lambda n: {"w": torch.full((n,), 2 ** 24 + 1)})   # float32 would round it to 2**24
    with pytest.raises(ValueError, match="dtype"):
        collate_samples([f32, i64], [(0, 0, 0), (1, 1, 0)], shape=(2, 4), pad_values={**PADS, "w": 0.0})


# ----------------------------------------------------------------------------------------------- batchers
class Upstream:
    """A finite sample stream that records how many samples were pulled."""
    def __init__(self, samples):
        self.samples, self.pulled = samples, 0

    def __iter__(self):
        for s in self.samples:
            self.pulled += 1
            yield s


def delivered_ids(batches):
    out = []
    for b in batches:
        out += [m["_id"] for m in b["sample_meta"]]
    return out


def test_packing_next_fit_pending_and_tail():
    """B=1, T=8, lengths 6, 4, 3 -> [A] then [B, C]; B was pulled before the first batch was handed
    out (it did not fit) and is pending until the next one."""
    up = Upstream([sample(6, 0, "A"), sample(4, 10, "B"), sample(3, 20, "C")])
    pb = PackingBatcher(up, shape=(1, 8), pad_values=PADS)
    it = iter(pb)
    first = next(it)
    assert [m["_id"] for m in first["sample_meta"]] == ["A"] and up.pulled == 2 and pb.pending["meta"]["_id"] == "B"
    second = next(it)
    assert [m["_id"] for m in second["sample_meta"]] == ["B", "C"] and pb.pending is None
    assert second["doc_id"].tolist() == [[0, 0, 0, 0, 1, 1, 1, -1]]
    with pytest.raises(StopIteration):
        next(it)


def test_packing_exactly_full_does_not_pull_one_more():
    up = Upstream([sample(4, 0, "A"), sample(4, 10, "B"), sample(4, 20, "C")])
    it = iter(PackingBatcher(up, shape=(1, 8), pad_values=PADS))
    b = next(it)
    assert [m["_id"] for m in b["sample_meta"]] == ["A", "B"] and up.pulled == 2


def test_padding_batcher_one_per_row_tail_and_nothing_held_back():
    up = Upstream([sample(n, 10 * i, i) for i, n in enumerate([3, 1, 4, 2, 2])])
    it = iter(PaddingBatcher(up, shape=(2, 4), pad_values=PADS))
    first = next(it)
    assert up.pulled == 2                                          # B samples pulled for B rows, none held back
    batches = [first, *it]
    assert [len(b["sample_meta"]) for b in batches] == [2, 2, 1]
    assert all(b["layout_facts"]["one_sample_per_row_left_aligned"] for b in batches)
    assert batches[-1]["doc_id"][1].tolist() == [-1] * 4          # the tail batch's unused row is padding


@pytest.mark.parametrize("cls", [PaddingBatcher, PackingBatcher])
def test_batchers_refuse_a_sample_longer_than_a_row(cls):
    with pytest.raises(ValueError, match="longer than a row"):
        list(cls([sample(9)], shape=(2, 8), pad_values=PADS))


def test_packing_with_one_row_refuses_a_long_sample_instead_of_ending():
    # With B = 1 a sample longer than the row would become pending and the next batch would come out empty: without
    # the check the stream would just end there, the rest of the data never delivered.
    batches = PackingBatcher([sample(8), sample(9), sample(2)], shape=(1, 8), pad_values=PADS)
    it = iter(batches)
    assert len(next(it)["sample_meta"]) == 1
    with pytest.raises(ValueError, match="longer than a row"):
        next(it)


@pytest.mark.parametrize("cls", [PaddingBatcher, PackingBatcher])
def test_batchers_empty_stream_yields_nothing(cls):
    assert list(cls([], shape=(2, 8), pad_values=PADS)) == []


@pytest.mark.parametrize("cls", [PaddingBatcher, PackingBatcher])
@pytest.mark.parametrize("seed", range(20))
def test_random_finite_streams_lose_duplicate_or_reorder_nothing(cls, seed):
    rng = random.Random(seed)
    B, T = rng.choice([(1, 8), (2, 8), (3, 16), (4, 5)])
    n = rng.randint(0, 40)
    samples = [sample(rng.randint(1, T), 100 * i, i) for i in range(n)]
    batches = list(cls(samples, shape=(B, T), pad_values=PADS))
    assert delivered_ids(batches) == list(range(n))
    for b in batches:
        assert b["idx"].shape == (B, T) and len(b["sample_meta"]) >= 1
        assert (b["targets"] != IGNORE).any()                      # a tail batch holds a real sample, so a target
        for i, m in enumerate(b["sample_meta"]):                   # every real position carries its own sample
            sel = b["doc_id"] == i
            src = samples[m["_id"]]["data"]
            assert torch.equal(b["idx"][sel], src["idx"]) and torch.equal(b["targets"][sel], src["targets"])
            assert b["doc_pos"][sel].tolist() == list(range(len(src["idx"])))
        assert (b["idx"][b["doc_id"] < 0] == PAD).all() and (b["targets"][b["doc_id"] < 0] == IGNORE).all()
        if cls is PaddingBatcher:
            assert b["layout_facts"]["one_sample_per_row_left_aligned"]


# ----------------------------------------------------------------------------------------------- mixer
def test_mixer_counts_samples_and_follows_weights():
    a, b, z = (["a"] * 100), (["b"] * 100), (["z"] * 100)
    mixer = SampleMixer({"a": (a, 1.0), "b": (b, 3.0), "z": (z, 0.0)})
    picks = [p for _, p in zip(range(40), mixer)]        # range first: zip must not pull a 41st sample
    src = [name for name, _ in picks]
    assert src.count("z") == 0 and src.count("b") == 30 and src.count("a") == 10
    assert mixer.samples_drawn_per_source == {"a": 10, "b": 30, "z": 0}
    mixer.reset_phase()
    assert mixer.samples_drawn_per_source == {"a": 0, "b": 0, "z": 0}


def test_mixer_breaks_ties_in_the_mappings_order():
    """The documented example: text, t2m, text, text, t2m for weights 2 : 1."""
    mixer = SampleMixer({"text": (["t"] * 9, 2), "t2m": (["m"] * 9, 1)})
    assert [pick[0] for _, pick in zip(range(5), mixer)] == ["text", "t2m", "text", "text", "t2m"]


def test_mixer_ends_when_an_active_source_ends_and_never_iterates_zero_weight():
    class Boom:
        def __iter__(self):
            raise AssertionError("a zero-weight source must not be iterated")
    out = list(SampleMixer({"a": (["a"] * 2, 1), "b": (["b"] * 10, 1), "z": (Boom(), 0)}))
    assert [s for s, _ in out] == ["a", "b", "a", "b"]              # source a ends on its third draw -> mix ends


@pytest.mark.parametrize("weights", [[1, -1], [0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_mixer_rejects_bad_weights(weights):
    with pytest.raises(ValueError, match="finite, nonnegative and not all zero"):
        SampleMixer({f"s{i}": ([1], w) for i, w in enumerate(weights)})


def test_mixer_needs_a_source_and_a_weight_with_each():
    class NeverIterated:
        def __iter__(self):
            raise AssertionError("building a mixer must not iterate a source")
    with pytest.raises(ValueError, match="at least one source"):
        SampleMixer({})
    with pytest.raises(TypeError, match="source 'a': each value is a \\(source, weight\\) pair, not a NeverIterated"):
        SampleMixer({"a": NeverIterated()})
    with pytest.raises(TypeError, match="pair, not a list"):
        SampleMixer({"a": [[1], 1.0]})
    with pytest.raises(TypeError, match="pair, not a tuple of 1"):              # a stray comma: (source,)
        SampleMixer({"a": ([1],)})
