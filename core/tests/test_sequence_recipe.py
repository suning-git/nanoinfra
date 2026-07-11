"""
SequenceRecipe mechanism tests against the control_resolver CONTRACT:
resolve(name) -> int | None — the recipe holds no name list of its own.
Everything here is modality-free: fake resolver, fake content tokenizer,
hand-built layout.
"""

import pytest
import torch

from core.data.sequence_recipe import SequenceRecipe
from core.tokenization.vocab_layout import VocabLayout


class FakeResolver:
    """Minimal object satisfying the control_resolver contract."""

    def __init__(self, names, offset):
        self._ids = {n: i for i, n in enumerate(names)}
        self._offset = offset

    def resolve(self, name):
        local = self._ids.get(name)
        return None if local is None else self._offset + local


class FakeContentTokenizer:
    def encode(self, text):
        return [ord(c) % 100 for c in text]  # deterministic ids in the content band


@pytest.fixture
def world():
    layout = VocabLayout()
    layout.add_range(0, 0, 100)      # content band
    layout.add_range(2, 100, 104)    # control band: bos, eos, x_start, x_end
    resolver = FakeResolver(["bos", "eos", "x_start", "x_end"], offset=100)
    return layout, resolver


def test_assemble_resolution_order_and_types(world):
    layout, resolver = world
    recipe = SequenceRecipe(template=['bos', 'text_tokens', 'eos'])
    out = recipe.assemble({'text_tokens': [5, 6, 7]}, layout, resolver)
    assert out['tokens'].tolist() == [100, 5, 6, 7, 101]
    assert out['token_types'].tolist() == [2, 0, 0, 0, 2]
    assert out['loss_mask'].tolist() == [1, 1, 1, 1, 1]  # supervise: all


def test_supervise_by_segment_name(world):
    layout, resolver = world
    recipe = SequenceRecipe(
        template=['bos', 'x_start', 'text_tokens', 'x_end', 'eos'],
        supervise=['text_tokens', 'x_end'],
    )
    out = recipe.assemble({'text_tokens': [1, 2]}, layout, resolver)
    assert out['loss_mask'].tolist() == [0, 0, 1, 1, 1, 0]


def test_constants_need_content_tokenizer(world):
    layout, resolver = world
    recipe = SequenceRecipe(template=['bos', 'cap', 'eos'], constants={'cap': 'hi'})
    with pytest.raises(ValueError, match="content_tokenizer"):
        recipe.assemble({}, layout, resolver)
    out = recipe.assemble({}, layout, resolver, content_tokenizer=FakeContentTokenizer())
    assert out['tokens'].tolist() == [100, ord('h') % 100, ord('i') % 100, 101]


def test_unknown_segment_raises(world):
    layout, resolver = world
    recipe = SequenceRecipe(template=['bos', 'nope'])
    with pytest.raises(ValueError, match="Unknown segment 'nope'"):
        recipe.assemble({}, layout, resolver)


def test_resolver_is_the_authority_not_a_name_list(world):
    """A name unknown to THIS resolver is not a control token — even one that
    'looks special'. Membership = resolve() is not None, nothing else."""
    layout, resolver = world
    recipe = SequenceRecipe(template=['user_start'])
    with pytest.raises(ValueError, match="Unknown segment"):
        recipe.assemble({}, layout, resolver)
    # ...but the same name works as a plain dataset field
    out = recipe.assemble({'user_start': [9]}, layout, resolver)
    assert out['tokens'].tolist() == [9]


def test_build_fixed_layout_and_overhead(world):
    layout, resolver = world
    recipe = SequenceRecipe(
        template=['bos', 'cap', 'text_tokens', 'eos'],
        supervise=['text_tokens'],
        constants={'cap': 'ab'},
    )
    tok = FakeContentTokenizer()
    assert recipe.overhead_tokens(resolver, content_tokenizer=tok) == 4  # bos+eos+2 const
    fixed = recipe.build_fixed_layout({'text_tokens': 3}, layout, resolver,
                                      content_tokenizer=tok)
    assert fixed['field_slices'] == {'text_tokens': (3, 6)}
    assert fixed['token_types'].tolist() == [2, 0, 0, 0, 0, 0, 2]
    assert fixed['loss_mask'].tolist() == [0, 0, 0, 1, 1, 1, 0]
