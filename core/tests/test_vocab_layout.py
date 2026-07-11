"""Unit tests for VocabLayout: bands, classify, offset service, sizes.

Replaces the old hardcoded classify_token_types(control_offset, motion_offset)
test — types are now a property of an assembled layout, not fixed offsets.
"""

import pytest
import torch

from core.tokenization.vocab_layout import VocabLayout


def _layout():
    # text [0,100) type 0, motion [100,164) type 1, control [164,182) type 2
    layout = VocabLayout()
    layout.add_range(0, 0, 100)
    layout.add_range(1, 100, 164)
    layout.add_range(2, 164, 182)
    return layout


def test_add_range_rejects_overlap():
    layout = VocabLayout()
    layout.add_range(0, 0, 100)
    with pytest.raises(ValueError):
        layout.add_range(1, 99, 150)


def test_sizes_and_offsets():
    layout = _layout()
    assert layout.vocab_size == 182
    assert layout.n_token_types == 3
    assert layout.offset(0) == 0
    assert layout.offset(1) == 100
    assert layout.offset(2) == 164
    # gaps allowed: n_token_types = max type_id + 1 (type_emb is indexed by type_id)
    sparse = VocabLayout()
    sparse.add_range(0, 0, 10)
    sparse.add_range(2, 10, 20)
    assert sparse.n_token_types == 3


def test_classify_token_types():
    layout = _layout()
    ids = torch.tensor([[0, 99, 100, 163], [164, 181, 5, 150]])
    types = layout.classify_token_types(ids)
    assert types.tolist() == [[0, 0, 1, 1], [2, 2, 0, 1]]
    # IDs outside every band classify to IGNORE_INDEX
    out = layout.classify_token_types(torch.tensor([182, 999]))
    assert out.tolist() == [VocabLayout.IGNORE_INDEX, VocabLayout.IGNORE_INDEX]


def test_offset_round_trip():
    layout = _layout()
    local = torch.tensor([0, 5, 63])
    glob = layout.to_global(1, local)
    assert glob.tolist() == [100, 105, 163]
    assert (layout.classify_token_types(glob) == 1).all()
    assert torch.equal(layout.to_local(glob), local)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
