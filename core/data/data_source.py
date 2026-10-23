"""
DataSource: unified interface for individual sample production.

Each DataSource is an iterator yielding RAW samples:
    {tokens: [L], token_types: [L] (optional), supervision_mask: [L] bool (optional), meta: {...} (optional)}

A sample is what the source decides it is (a text window, a motion clip, a caption + motion pair) at its own true
length L: no mechanical padding and no attention_mask — batching pads or packs later, and doc_id marks the real
positions. Supervision is NOT applied here: the loader prepares each sample (e.g. build_NTP_supervised_sample) before
batching. A CPU-native source yields CPU tensors; the loader moves whole batches to the device.

Resume state: get_state() describes THIS rank's logical progress; the run's state, which a checkpoint keeps, is
merge_rank_states of every rank's (optional: without it the loader keeps rank 0's, approximate); set_state() takes
the run's state and interprets it under the current rank / world_size (an indexable source should use
ResumableDistributedSampler, which does all three). The loader takes a private copy of the state once when it is
built (the source's starting point) and after every pull; a source may keep editing its live state.

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

    Subclasses produce individual raw samples (not batches), each at its own length.
    Each source handles its own distributed sharding internally.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield raw samples {tokens: [L], token_types: [L], supervision_mask: [L] bool, meta: {...}}
        (token_types, supervision_mask and meta optional)."""
        ...

    @abstractmethod
    def get_state(self) -> Optional[Dict[str, Any]]:
        """Return current state for checkpointing, or None if the source keeps none.

        It is a position to come back to, also before the first sample: the loader records each source's state when
        it is built, as the source's starting point. A restore hands the run's state back through set_state, and
        get_state() right after it must report where that put THIS rank (the loader records that as the source's
        entry until it is drawn again: for a source with merge_rank_states, this rank's share). A source that
        returns None keeps no position: a restore leaves it where it stands (in a new process, at its beginning).

        The state (merged over the ranks when the source has merge_rank_states) is written to the checkpoint's
        meta.json as JSON and handed back to set_state on resume, so it must come back exactly as returned. Use only
        dict, list, str, int, float, bool and None:
          - dict keys are strings (an int key, e.g. a shard number, comes back a string: store str(n), convert back
            in set_state);
          - lists, not tuples (a tuple comes back a list);
          - no numpy numbers or tensors (int(x), float(x), .tolist()).
        Nothing checks this when a checkpoint is saved. Each source tests it: after a few samples,
        json.loads(json.dumps(state)) == state.
        """
        ...

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint state: the run's (merge_rank_states of every rank's, when the source has it), read
        under the current rank / world_size."""
        ...

    # Optional, found by the loader with getattr:
    #
    #   def merge_rank_states(self, states: list) -> state
    #
    # The run's state from every rank's state (the ranks of one run, each once; each state is the source's after the
    # last sample that rank delivered). It is what makes a resume, on another number of ranks too, neither skip nor
    # repeat a sample: the furthest rank's position, and the samples before it that no rank has trained, which
    # set_state then splits over the ranks of the resume and the source hands out first. It must be pure (read the
    # states, never the source's live position) and give the same result on every rank. A source that marks one
    # rank's state as such (ResumableDistributedSampler adds "shard") can refuse it in set_state when it holds that
    # rank's part of what is left.
    # Without the method, the loader keeps rank 0's state: exact only when every rank consumed the same number of
    # the source's samples (one rank; padding over a sampler).

    def budget_tokens(self) -> Optional[int]:
        """Return total token budget if configured (for weight:auto), else None."""
        return None

    def __repr__(self) -> str:
        """Compact one-line state summary for checkpoint logs. A modality may
        override this for a prettier label; the default names the source by its
        class, so core never has to know which modality it is."""
        return f"{type(self).__name__}:{self.get_state()}"
