"""
SequenceRecipe: declarative sequence template + supervision mask.

Defines how raw tokenized fields are assembled into a complete training sequence with token types and a
supervision mask (bool, True = supervised target).

A recipe has four fields:
    - template: ordered list of segment names
    - supervise_tags: per-segment tag for supervision selection (optional)
    - supervise: which tags to include in loss ('all' or list of tag values)
    - constants: named literal text strings, tokenized at assembly time (optional)

Three expression levels (same infrastructure):
    # Level 1: supervise all (default, zero config)
    supervise: all

    # Level 2: no supervise_tags → auto = segment names → select by name
    supervise: [motion_start, motion_codes, motion_end]

    # Level 3: explicit tags → horizontal readable layout
    supervise_tags: [1, 1, 0, 1, 1, 1, 1, 1]
    supervise: [1]

Usage:
    result = recipe.assemble({'text_tokens': [...]}, layout, control_resolver)
    # result = {tokens: [S], token_types: [S], supervision_mask: [S] (bool)}
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# LATE-BOUND: token types come from a VocabLayout (id -> type); protocol names
# resolve via the control_resolver (contract: resolve(name) -> int | None) —
# THE authority on control-token names, assembled by the orchestrator from the
# control modality x the layout. The recipe is decoupled from any tokenizer
# and holds no name list of its own.


class SequenceRecipe:
    """Declarative sequence template with supervision mask.

    Each template entry is one of three kinds, decided once in _segments for all three methods:
        1. Control token (bos, text_start, etc.) → single token ID (via resolver)
        2. Constant (key in constants dict) → tokenized literal (content tokenizer)
        3. Dataset field (for example, text_tokens) → token list from the fields dict
    A name has one kind: a name that is both a control token and a constant, or a control token or constant
    that is also given as a field, is refused rather than resolved by precedence.

    Token types come from the VocabLayout (classify by ID band), not hardcoded offsets.
    """

    def __init__(
        self,
        template: List[str],
        supervise: Union[str, List] = 'all',
        supervise_tags: Optional[List] = None,
        constants: Optional[Dict[str, str]] = None,
    ):
        self.template = template
        self.supervise = supervise
        self.constants = constants or {}

        # supervise_tags: if not provided, auto-default to segment names
        if supervise_tags is not None:
            if len(supervise_tags) != len(template):
                raise ValueError(
                    f"supervise_tags length ({len(supervise_tags)}) != "
                    f"template length ({len(template)})"
                )
            self.supervise_tags = list(supervise_tags)
        else:
            self.supervise_tags = list(template)

        # Pre-compute supervised tag set. A single string other than 'all' would become a set of its
        # characters and quietly supervise nothing.
        if self.supervise == 'all':
            self._supervised_set = None  # all supervised
        elif isinstance(self.supervise, str):
            raise ValueError(f"supervise is 'all' or a list of tags, got the string {self.supervise!r}")
        else:
            self._supervised_set = set(self.supervise)

        # Cache for constant tokenization (lazy, filled on first use)
        self._constant_cache: Dict[str, List[int]] = {}

    def _constant_tokens(self, name: str, content_tokenizer) -> List[int]:
        """A constant's tokens, encoded once by the recipe's own modality tokenizer."""
        if name not in self._constant_cache:
            if content_tokenizer is None:
                raise ValueError(f"constant '{name}' needs a content_tokenizer")
            self._constant_cache[name] = content_tokenizer.encode(self.constants[name])
        return self._constant_cache[name]

    def _segments(self, control_resolver, content_tokenizer) -> List[Tuple[int, str, str, Optional[List[int]]]]:
        """The template as (index, name, kind, tokens): kind 'control' or 'constant' with its tokens, or 'field'
        with None (filled per sample).

        control_resolver: resolve(name) -> global ID | None (the protocol authority).
        content_tokenizer: the recipe's own modality tokenizer, for constants only.
        """
        segments = []
        for i, name in enumerate(self.template):
            control_id = control_resolver.resolve(name)
            if control_id is not None:
                if name in self.constants:
                    raise ValueError(f"segment '{name}' is both a control token and a constant")
                segments.append((i, name, 'control', [control_id]))
            elif name in self.constants:
                segments.append((i, name, 'constant', self._constant_tokens(name, content_tokenizer)))
            else:
                segments.append((i, name, 'field', None))
        return segments

    def _is_supervised(self, idx: int) -> bool:
        """Check if segment at template index idx participates in loss."""
        if self._supervised_set is None:
            return True
        return self.supervise_tags[idx] in self._supervised_set

    def _assemble(
        self, fields: Dict[str, List[int]], layout, control_resolver, content_tokenizer
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[Tuple[int, int]]]]:
        """The assembled sequence and, per field name, the (start, end) of each place it was filled."""
        all_tokens: List[int] = []
        all_supervised: List[bool] = []
        field_places: Dict[str, List[Tuple[int, int]]] = {}

        for i, name, kind, seg_tokens in self._segments(control_resolver, content_tokenizer):
            if kind == 'field':
                if name not in fields:
                    raise ValueError(
                        f"Unknown segment '{name}'. Not a control token, not in "
                        f"fields {list(fields.keys())}, or constants {list(self.constants.keys())}"
                    )
                seg_tokens = fields[name] if isinstance(fields[name], list) else list(fields[name])
                field_places.setdefault(name, []).append((len(all_tokens), len(all_tokens) + len(seg_tokens)))
            elif name in fields:
                raise ValueError(f"segment '{name}' is a {'control token' if kind == 'control' else kind} "
                                 f"and is also given as a field")
            all_tokens.extend(seg_tokens)
            all_supervised.extend([bool(self._is_supervised(i))] * len(seg_tokens))

        tokens = torch.tensor(all_tokens, dtype=torch.long)
        supervision_mask = torch.tensor(all_supervised, dtype=torch.bool)

        # Token types: from the VocabLayout (id -> type), not a hardcoded offset
        token_types = layout.classify_token_types(tokens)

        return {
            'tokens': tokens,
            'token_types': token_types,
            'supervision_mask': supervision_mask,
        }, field_places

    def assemble(
        self, fields: Dict[str, List[int]], layout, control_resolver, content_tokenizer=None
    ) -> Dict[str, torch.Tensor]:
        """
        Assemble full sequence from raw tokenized fields.

        Args:
            fields: e.g. {'text_tokens': [int, ...]}
            layout: VocabLayout — classifies global IDs into token types.
            control_resolver: resolve(name) -> global ID | None (protocol authority).
            content_tokenizer: the recipe's modality tokenizer (constants only; optional).

        Returns:
            {tokens: Tensor[S], token_types: Tensor[S], supervision_mask: Tensor[S] (bool)}
        """
        return self._assemble(fields, layout, control_resolver, content_tokenizer)[0]

    def build_fixed_layout(
        self, field_lengths: Dict[str, int], layout, control_resolver,
        content_tokenizer=None, field_dummy_ids: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-compute layout for fixed-length sequences.

        Optimization for DataSources where every sample has identical structure
        (e.g., TextDataSource where text always fills to a fixed length).
        Assembles once with dummy token IDs and returns the result.
        Caller then reuses token_types/supervision_mask directly per sample,
        only filling in field positions via field_slices.

        Args:
            field_lengths: fixed token count per field, e.g. {'text_tokens': 508}; each >= 0, and each field
                appears once in the template (field_slices holds one slice per field).
            layout: VocabLayout — classifies global IDs into token types.
            control_resolver: resolve(name) -> global ID | None (protocol authority).
            content_tokenizer: the recipe's modality tokenizer (constants only; optional).
            field_dummy_ids: the token ID that stands for every position of a field, which decides the type
                the layout gives that region. Defaults to 0, whichever band holds ID 0 in this layout. Pass an
                ID inside the field's own band; a field whose positions are of more than one type cannot be
                described by one dummy ID.

        Returns:
            token_template: Tensor[S] — filled except at field positions (which hold the dummy IDs)
            token_types: Tensor[S] — correct for all positions
            supervision_mask: Tensor[S] (bool) — correct for all positions
            field_slices: {field_name: (start, end)} — positions to fill per sample
        """
        negative = {name: length for name, length in field_lengths.items() if length < 0}
        if negative:
            raise ValueError(f"field lengths must be >= 0, got {negative}")
        dummy_fields = {
            name: [(field_dummy_ids or {}).get(name, 0)] * length
            for name, length in field_lengths.items()
        }
        result, field_places = self._assemble(dummy_fields, layout, control_resolver, content_tokenizer)
        repeated = {name: len(places) for name, places in field_places.items() if len(places) > 1}
        if repeated:
            raise ValueError(f"build_fixed_layout: fields appear more than once in the template {repeated}; "
                             f"field_slices holds one slice per field")

        return {
            'token_template': result['tokens'],
            'token_types': result['token_types'],
            'supervision_mask': result['supervision_mask'],
            'field_slices': {name: places[0] for name, places in field_places.items()},
        }

    def overhead_tokens(self, control_resolver, content_tokenizer=None) -> int:
        """Total tokens consumed by non-field segments (control tokens + constants).
        content_tokenizer needed only if the recipe has constants."""
        return sum(len(seg_tokens) for _, _, kind, seg_tokens in self._segments(control_resolver, content_tokenizer)
                   if kind != 'field')
