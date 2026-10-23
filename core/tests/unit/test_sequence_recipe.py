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
    assert out['supervision_mask'].tolist() == [True] * 5  # supervise: all
    assert out['supervision_mask'].dtype == torch.bool


def test_supervise_by_segment_name(world):
    layout, resolver = world
    recipe = SequenceRecipe(
        template=['bos', 'x_start', 'text_tokens', 'x_end', 'eos'],
        supervise=['text_tokens', 'x_end'],
    )
    out = recipe.assemble({'text_tokens': [1, 2]}, layout, resolver)
    assert out['supervision_mask'].tolist() == [False, False, True, True, True, False]


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
    assert fixed['supervision_mask'].tolist() == [False, False, False, True, True, True, False]


def test_a_fixed_layout_filled_with_fields_is_the_assembled_sequence(world):
    """The fixed layout and assemble come from one walk: filling field_slices with a sample's fields gives exactly
    what assemble returns for it, and overhead_tokens counts the rest."""
    layout, resolver = world
    tok = FakeContentTokenizer()
    recipe = SequenceRecipe(template=['bos', 'x_start', 'cap', 'text_tokens', 'x_end', 'motion', 'eos'],
                            supervise=['text_tokens', 'x_end'], constants={'cap': 'hey'})
    fields = {'text_tokens': [11, 12, 13, 14], 'motion': [70, 71]}
    fixed = recipe.build_fixed_layout({name: len(t) for name, t in fields.items()}, layout, resolver,
                                      content_tokenizer=tok)
    filled = fixed['token_template'].clone()
    for name, (a, b) in fixed['field_slices'].items():
        filled[a:b] = torch.tensor(fields[name])
    out = recipe.assemble(fields, layout, resolver, content_tokenizer=tok)
    assert filled.tolist() == out['tokens'].tolist()
    assert fixed['token_types'].tolist() == out['token_types'].tolist()
    assert fixed['supervision_mask'].tolist() == out['supervision_mask'].tolist()
    assert recipe.overhead_tokens(resolver, content_tokenizer=tok) == out['tokens'].numel() - 6


def test_a_name_has_one_kind(world):
    """A control token that is also a constant, or a control token or constant also given as a field, is refused
    (precedence would quietly drop one of them; overhead_tokens, which sees no fields, would count differently)."""
    layout, resolver = world
    tok = FakeContentTokenizer()
    both = SequenceRecipe(template=['bos', 'eos'], constants={'bos': 'ab'})
    with pytest.raises(ValueError, match="both a control token and a constant"):
        both.overhead_tokens(resolver, content_tokenizer=tok)
    with pytest.raises(ValueError, match="is a constant and is also given as a field"):
        SequenceRecipe(template=['bos', 'cap'], constants={'cap': 'ab'}).assemble(
            {'cap': [1, 2, 3]}, layout, resolver, content_tokenizer=tok)
    with pytest.raises(ValueError, match="is a control token and is also given as a field"):
        SequenceRecipe(template=['bos', 'eos']).assemble({'bos': [1]}, layout, resolver)
    # a field the template does not use is not a conflict
    out = SequenceRecipe(template=['bos']).assemble({'eos': [1]}, layout, resolver)
    assert out['tokens'].tolist() == [100]


def test_build_fixed_layout_refuses_a_repeated_field_and_a_negative_length(world):
    """field_slices holds one slice per field, so a field that appears twice would leave its first place unfilled;
    assemble itself takes a repeated field."""
    layout, resolver = world
    recipe = SequenceRecipe(template=['x', 'bos', 'x'])
    with pytest.raises(ValueError, match="more than once"):
        recipe.build_fixed_layout({'x': 2}, layout, resolver)
    assert recipe.assemble({'x': [8, 9]}, layout, resolver)['tokens'].tolist() == [8, 9, 100, 8, 9]
    with pytest.raises(ValueError, match=">= 0"):
        SequenceRecipe(template=['bos', 'x']).build_fixed_layout({'x': -1}, layout, resolver)


def test_supervise_is_all_or_a_list():
    """A single string would become a set of its characters and supervise nothing."""
    with pytest.raises(ValueError, match="supervise is 'all' or a list"):
        SequenceRecipe(template=['bos', 'text_tokens'], supervise='text_tokens')


def test_overhead_without_a_tokenizer_says_so(world):
    _, resolver = world
    recipe = SequenceRecipe(template=['bos', 'cap'], constants={'cap': 'ab'})
    with pytest.raises(ValueError, match="content_tokenizer"):
        recipe.overhead_tokens(resolver)


def test_build_fixed_layout_slices_and_types_with_two_fields_and_dummy_ids():
    """The path the motion and video sources take, pinned: field_slices in template order whatever the order of
    field_lengths, each field typed by its dummy ID's band, the dummy IDs left in token_template."""
    layout = VocabLayout()
    layout.add_range(0, 0, 100)      # content band
    layout.add_range(2, 100, 104)    # control band
    layout.add_range(3, 104, 200)    # another modality's band
    resolver = FakeResolver(["bos", "eos", "x_start", "x_end"], offset=100)
    recipe = SequenceRecipe(template=['bos', 'text_tokens', 'x_start', 'motion_tokens', 'x_end', 'eos'],
                            supervise=['motion_tokens', 'x_end'])
    fixed = recipe.build_fixed_layout({'motion_tokens': 3, 'text_tokens': 2}, layout, resolver,
                                      field_dummy_ids={'motion_tokens': 150, 'text_tokens': 7})
    assert list(fixed['field_slices'].items()) == [('text_tokens', (1, 3)), ('motion_tokens', (4, 7))]
    assert fixed['token_template'].tolist() == [100, 7, 7, 102, 150, 150, 150, 103, 101]
    assert fixed['token_types'].tolist() == [2, 0, 0, 2, 3, 3, 3, 2, 2]
    assert fixed['supervision_mask'].tolist() == [False, False, False, False, True, True, True, True, False]
