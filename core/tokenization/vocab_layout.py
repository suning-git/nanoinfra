"""
VocabLayout — the shared token-ID space, laid out into per-modality typed bands.

PURE INTEGER STRUCTURE — no strings, no tokenizers, no special-token names. It owns:
- the partition of the ID space:   {type_id: [start, end)}
- classify:        global ID -> type_id
- offset service:  local <-> global translation (each modality only knows its own
                   local [0, size); the layout assigns offsets and translates)
- sizes:           vocab_size, n_token_types

The orchestrator assembles a layout from modality contributions and passes it
explicitly to the data pipeline (classify) and the model config (n_token_types) —
core never hardcodes band boundaries. Special-token NAME resolution lives in the
control modality, not here. Per-type LOSS weighting is a head policy, not vocab
structure.
"""

import torch
from torch import Tensor


class VocabLayout:
    IGNORE_INDEX = -1

    def __init__(self):
        self.ranges: dict[int, tuple[int, int]] = {}  # {type_id: (start, end)}

    # ---- construction -----------------------------------------------------
    def add_range(self, type_id: int, start: int, end: int) -> None:
        for existing_id, (s, e) in self.ranges.items():
            if start < e and end > s:
                raise ValueError(
                    f"Range [{start}, {end}) for type {type_id} overlaps "
                    f"[{s}, {e}) for type {existing_id}"
                )
        self.ranges[type_id] = (start, end)

    # ---- sizes ------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return max((end for _, end in self.ranges.values()), default=0)

    @property
    def n_token_types(self) -> int:
        """type_emb is indexed by type_id; size = max type_id + 1 (gaps allowed)."""
        return max(self.ranges.keys(), default=-1) + 1

    def offset(self, type_id: int) -> int:
        """The global start of this type's band (= the modality's offset)."""
        return self.ranges[type_id][0]

    # ---- classify (global ID -> type) ------------------------------------
    def classify_token_types(self, token_ids: Tensor) -> Tensor:
        result = torch.full_like(token_ids, self.IGNORE_INDEX)
        for type_id, (start, end) in self.ranges.items():
            mask = (token_ids >= start) & (token_ids < end)
            result[mask] = type_id
        return result

    # ---- offset service (local <-> global) -------------------------------
    def to_global(self, type_id: int, local_ids: Tensor) -> Tensor:
        """A modality's local [0,size) IDs -> shared-vocab global IDs. Vectorized."""
        return local_ids + self.ranges[type_id][0]

    def to_local(self, global_ids: Tensor) -> Tensor:
        """Shared-vocab global IDs -> each modality's local IDs. Vectorized over the
        handful of ranges (never per-token)."""
        result = global_ids.clone()
        for _type_id, (start, end) in self.ranges.items():
            mask = (global_ids >= start) & (global_ids < end)
            result[mask] = global_ids[mask] - start
        return result
