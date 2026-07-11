"""
SequenceRecipe: declarative sequence template + supervision mask.

Defines how raw tokenized fields are assembled
into a complete training sequence with token types and loss mask.

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
    # result = {tokens: [S], token_types: [S], loss_mask: [S]}
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

    Segment resolution order (per template entry):
        1. Control token (bos, text_start, etc.) → single token ID (via resolver)
        2. Dataset field (for example, text_tokens) -> token list from fields dict
        3. Constant (key in constants dict) → tokenized literal (content tokenizer)

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

        # Pre-compute supervised tag set
        if self.supervise == 'all':
            self._supervised_set = None  # all supervised
        else:
            self._supervised_set = set(self.supervise)

        # Cache for constant tokenization (lazy, filled on first use)
        self._constant_cache: Dict[str, List[int]] = {}

    def _resolve_segment(
        self, name: str, fields: Dict[str, List[int]], control_resolver, content_tokenizer
    ) -> List[int]:
        """Resolve segment name to a list of token IDs.

        control_resolver: resolve(name) -> global ID | None (the protocol authority).
        content_tokenizer: the recipe's own modality tokenizer, for constants only.
        """
        # 1. Control token -> shared vocab (control's band), not a tokenizer
        control_id = control_resolver.resolve(name)
        if control_id is not None:
            return [control_id]

        # 2. Dataset field
        if name in fields:
            tokens = fields[name]
            return tokens if isinstance(tokens, list) else list(tokens)

        # 3. Constant -> the recipe's own modality tokenizer
        if name in self.constants:
            if name not in self._constant_cache:
                if content_tokenizer is None:
                    raise ValueError(f"constant '{name}' needs a content_tokenizer")
                self._constant_cache[name] = content_tokenizer.encode(self.constants[name])
            return self._constant_cache[name]

        raise ValueError(
            f"Unknown segment '{name}'. Not a control token, not in "
            f"fields {list(fields.keys())}, or constants {list(self.constants.keys())}"
        )

    def _is_supervised(self, idx: int) -> bool:
        """Check if segment at template index idx participates in loss."""
        if self._supervised_set is None:
            return True
        return self.supervise_tags[idx] in self._supervised_set

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
            {tokens: Tensor[S], token_types: Tensor[S], loss_mask: Tensor[S]}
        """
        all_tokens: List[int] = []
        all_loss_mask: List[int] = []

        for i, name in enumerate(self.template):
            seg_tokens = self._resolve_segment(name, fields, control_resolver, content_tokenizer)
            all_tokens.extend(seg_tokens)
            supervised = 1 if self._is_supervised(i) else 0
            all_loss_mask.extend([supervised] * len(seg_tokens))

        tokens = torch.tensor(all_tokens, dtype=torch.long)
        loss_mask = torch.tensor(all_loss_mask, dtype=torch.long)

        # Token types: from the VocabLayout (id -> type), not a hardcoded offset
        token_types = layout.classify_token_types(tokens)

        return {
            'tokens': tokens,
            'token_types': token_types,
            'loss_mask': loss_mask,
        }

    def build_fixed_layout(
        self, field_lengths: Dict[str, int], layout, control_resolver,
        content_tokenizer=None, field_dummy_ids: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-compute layout for fixed-length sequences.

        Optimization for DataSources where every sample has identical structure
        (e.g., TextDataSource where text always fills to a fixed length).
        Runs assemble() once with dummy token IDs, caches the result.
        Caller then reuses token_types/loss_mask directly per sample,
        only filling in field positions via field_slices.

        Args:
            field_lengths: fixed token count per field, e.g. {'text_tokens': 508}
            layout: VocabLayout — classifies global IDs into token types.
            control_resolver: resolve(name) -> global ID | None (protocol authority).
            content_tokenizer: the recipe's modality tokenizer (constants only; optional).
            field_dummy_ids: representative token ID per field for correct
                token_type classification. Defaults to 0 (text band). For a
                non-text field, pass a representative ID inside that modality's
                band so the layout classifies the region correctly.

        Returns:
            token_template: Tensor[S] — filled except at field positions (placeholder 0s)
            token_types: Tensor[S] — correct for all positions
            loss_mask: Tensor[S] — correct for all positions
            field_slices: {field_name: (start, end)} — positions to fill per sample
        """
        # Use dummy field values to compute the full layout.
        # Each field is filled with a representative token ID so that
        # classify_token_types produces the correct type for that region.
        dummy_fields = {
            name: [(field_dummy_ids or {}).get(name, 0)] * length
            for name, length in field_lengths.items()
        }
        result = self.assemble(dummy_fields, layout, control_resolver, content_tokenizer)

        # Compute field positions by walking the template.
        # Same precedence as _resolve_segment: control > field > constant.
        field_slices: Dict[str, Tuple[int, int]] = {}
        pos = 0
        for name in self.template:
            if control_resolver.resolve(name) is not None:
                pos += 1
            elif name in field_lengths:
                length = field_lengths[name]
                field_slices[name] = (pos, pos + length)
                pos += length
            elif name in self.constants:
                if name not in self._constant_cache:
                    self._constant_cache[name] = content_tokenizer.encode(
                        self.constants[name]
                    )
                pos += len(self._constant_cache[name])

        return {
            'token_template': result['tokens'],
            'token_types': result['token_types'],
            'loss_mask': result['loss_mask'],
            'field_slices': field_slices,
        }

    def overhead_tokens(self, control_resolver, content_tokenizer=None) -> int:
        """Total tokens consumed by non-field segments (control tokens + constants).
        content_tokenizer needed only if the recipe has constants."""
        total = 0
        for name in self.template:
            if control_resolver.resolve(name) is not None:
                total += 1
            elif name in self.constants:
                if name not in self._constant_cache:
                    self._constant_cache[name] = content_tokenizer.encode(
                        self.constants[name]
                    )
                total += len(self._constant_cache[name])
        return total
