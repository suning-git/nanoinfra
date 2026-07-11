"""
DataSource: unified interface for individual sample production.

Each DataSource is an infinite iterator yielding samples in a standard format:
    {tokens: [S], token_types: [S], attention_mask: [S], loss_weights: [S]}

All samples are padded to sequence_len. Supervision is NOT applied here —
it's applied once at the MixedDataLoader level.

Sequence assembly is delegated to SequenceRecipe (see sequence_recipe.py).
Each concrete source lives with its modality (e.g. the text modality's
TextDataSource); orchestrators inject them via MixedDataLoader's source_types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Optional

import torch


class DataSource(ABC):
    """
    Base class for data sources.

    Subclasses produce individual samples (not batches) in a unified format,
    padded to a fixed sequence_len. Each source handles its own distributed
    sharding internally.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield {tokens: [S], token_types: [S], attention_mask: [S], loss_weights: [S]}."""
        ...

    @abstractmethod
    def get_state(self) -> Optional[Dict[str, Any]]:
        """Return current state for checkpointing."""
        ...

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint state."""
        ...

    def budget_tokens(self) -> Optional[int]:
        """Return total token budget if configured (for weight:auto), else None."""
        return None

    def __repr__(self) -> str:
        """Compact one-line state summary for checkpoint logs. A modality may
        override this for a prettier label; the default names the source by its
        class, so core never has to know which modality it is."""
        return f"{type(self).__name__}:{self.get_state()}"
