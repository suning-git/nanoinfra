"""
TextDataSource — text's sample producer (FineWeb streaming + recipe assembly),
plus text's conventional single-modality recipe defaults (TEXT_RECIPE /
create_recipe). Moved out of core: core keeps only the
mechanisms (DataSource ABC, SequenceRecipe, MixedDataLoader).
"""

import os
from typing import Any, Dict, Iterator, Optional

import torch

from core.data.data_source import DataSource
from core.data.sequence_recipe import SequenceRecipe
from core.utils import process_identity

from modalities.text.fineweb import (START, RowGroups, exact_windows, merge_exact_states, rank_state, run_state,
                                     token_data_loader)


# ---------------------------------------------------------------------------
# Text's conventional recipes (a modality's own source defaulting to its own
# single-modality recipe — convenience constants, not hooks core calls)
# ---------------------------------------------------------------------------

TEXT_RECIPE = SequenceRecipe(
    template=['bos', 'text_start', 'text_tokens', 'text_end', 'eos'],
    supervise='all',
)


def create_recipe(recipe_config: Optional[Dict] = None,
                  source_type: str = 'text') -> SequenceRecipe:
    """Build a SequenceRecipe from config. ``None`` means NO assembly (raw packing).

    There is deliberately no invisible default. The predecessor did::

        if recipe_config is None:
            return _DEFAULT_RECIPES[source_type]   # never appeared in yaml or logs

    so training silently assembled ``bos/text_start/…/text_end/eos`` while the
    evaluator, which never went through here at all, packed raw text with only a
    bos. The two rulers disagreed for months without a single log line. Now the
    recipe is named in config and printed at startup by both sides.
    """
    if recipe_config is None:
        return None

    return SequenceRecipe(
        template=recipe_config['template'],
        supervise=recipe_config.get('supervise', 'all'),
        supervise_tags=recipe_config.get('supervise_tags'),
        constants=recipe_config.get('constants'),
    )


class TextDataSource(DataSource):
    """
    FineWeb text data source with SequenceRecipe-based assembly.

    Streams text from the nanoinfra base loader, wraps each chunk with
    delimiters defined by the recipe (default: [bos, text_start, ..., text_end, eos]).

    Uses build_fixed_layout(): the recipe structure is pre-computed once, and only
    the text tokens are filled in per sample. Every window is full (`sample_len`
    raw tokens: T + 1 for rows of T positions after the next-token shift), so a sample
    carries no padding and no attention_mask. The source yields CPU tensors unless its
    config gives another `device`; the loader moves whole batches. A sample's meta says
    where its text came from: {"from": [[epoch, file, row group, document, first
    token, count], ...]}.

    The types are the template's: the whole text field is typed text. A window is cut from
    the documents laid end to end, each opened by the tokenizer's bos, so a window often
    holds bos tokens, and those are typed text too, while the recipe's own bos opening the
    row is typed control. This is deliberate (2026-10-23): that bos only separates
    documents, and typing it by its ID made no measurable difference to a d6 run trained
    from scratch (end-point val CE +0.0007 and -0.0008 over two seeds). Documents in
    samples of their own (packing, no attention across them) would make it moot.

    Resume (FineWeb): exact. The state is a position down to the document and the token
    (fineweb.py, "Exact positions"); get_state() is this rank's, merge_rank_states makes
    the run's from every rank's, and set_state takes the run's under the current rank and
    world size, so a resume, on any number of ranks that each get a row group, neither skips nor repeats a token
    (the windows may be cut at other places than in an uninterrupted run: what is kept is every token position).
    It assumes the files (order and contents) and the tokenizer are the ones the state was
    saved with: `files` is recorded to diagnose, not checked. A state from before
    2026-10-22 ({files, pq_idx, rg_idx, epoch}) resumes at the next row group, approximate
    as it was. With a `data_path` (the synthetic byte stream of tests and offline runs)
    the state is that stream's and resumes approximately, as rank 0's.

    `rank` / `world_size` (config, optional, both or neither): which share of the
    files' row groups (and of a resumed state's leftover) this source reads. Unset, the
    launcher's process (what training wants); a source every rank reads whole, a
    validation set each rank freezes in full, gives rank 0 of 1.
    """

    def __init__(self, config: Dict[str, Any], tokenizers: Dict):
        self.text_tokenizer = tokenizers['text']
        # LATE-BOUND: layout (id->type) + control_resolver (name->id) come from the
        # orchestrator's assembled vocab, passed through the shared `tokenizers` bag.
        self._layout = tokenizers['layout']
        self._control_resolver = tokenizers['control_resolver']

        # Recipe: resolved by the orchestrator and passed in as a dict. Required —
        # a source that assembles nothing must say so explicitly (`recipe: null`).
        if 'recipe' not in config:
            raise KeyError(
                "TextDataSource: `recipe` is required (name it in config; use "
                "`recipe: null` for raw packing). No invisible default — that is "
                "exactly how train/val drifted apart."
            )
        self.recipe_name = config.get('recipe_name', '<inline>')
        self.recipe = create_recipe(config['recipe'], source_type='text')
        if self.recipe is None:
            raise ValueError(
                "TextDataSource requires a real recipe; for raw packing use "
                "token_data_loader directly (see TextEvaluator's raw bridge stream)."
            )

        # Which files this source reads — resolved by the orchestrator, never inferred.
        self.files = config.get('files')
        if not self.files:
            raise KeyError(
                "TextDataSource: `files` is required — resolve it with "
                "modalities.text.datasets.resolve_split(data_cfg, dataset, split)."
            )

        if 'sequence_len' in config:
            raise KeyError("TextDataSource: `sequence_len` is now `sample_len`, the raw tokens per window "
                           "(T + 1 for rows of T positions; modalities.text.streams.resolve_sources fills it in)")
        self.sample_len = config['sample_len']
        # TEXT_RECIPE has no constants, so overhead is just the control-token count
        self.text_len = self.sample_len - self.recipe.overhead_tokens(self._control_resolver)

        self.split = config.get('split', 'train')
        self.buffer_batch_size = config.get('buffer_batch_size')   # rows per base batch, the data_path stream's only
        self.tokenizer_threads = config.get('tokenizer_threads', 4)
        self.tokenizer_batch_size = config.get('tokenizer_batch_size', 128)
        self.device = torch.device(config.get('device', 'cpu'))     # like the other sources: samples on the CPU
        self.data_path = config.get('data_path')
        self.vocab_size = int(config.get('vocab_size', self.text_tokenizer.get_vocab_size()))
        self.rank, self.world_size = config.get('rank'), config.get('world_size')

        # Budget for weight:auto
        self._budget_tokens = config.get('tokens')

        # Pre-compute fixed layout (token template, types, supervision_mask, field positions)
        fixed = self.recipe.build_fixed_layout(
            {'text_tokens': self.text_len}, self._layout, self._control_resolver,
            content_tokenizer=self.text_tokenizer,
        )
        self._token_template = fixed['token_template'].to(self.device)
        self._type_template = fixed['token_types'].to(self.device)
        self._supervision_template = fixed['supervision_mask'].to(self.device)

        text_start, text_end = fixed['field_slices']['text_tokens']
        self._text_start = text_start
        self._text_end = text_end

        self._row_groups = RowGroups(self.files)
        self._current_state = None      # this rank's state after the last yielded window
        self._resume_state = None       # the state the next iteration starts from: the run's, or None (the start)
        if config.get('resume_state') is not None:
            self.set_state(config['resume_state'])

        print(f"TextDataSource initialized:")
        print(f"  sample_len={self.sample_len}, text_len={self.text_len}")
        print(f"  split={self.split}" + (f", buffer_batch_size={self.buffer_batch_size} (data_path stream)"
                                         if self.data_path is not None else ""))
        print(f"  recipe[{self.recipe_name}] template={self.recipe.template}")
        print(f"  files={[f.rsplit('/', 1)[-1] for f in self.files]}")
        if self._resume_state is not None:
            print(f"  resume_state={self._resume_state}")

    def _sample(self, text_tokens, meta=None):
        tokens = self._token_template.clone()
        tokens[self._text_start:self._text_end] = text_tokens
        sample = {
            'tokens': tokens,
            'token_types': self._type_template,       # shared (read-only)
            'supervision_mask': self._supervision_template,  # shared (read-only)
        }
        if meta is not None:
            sample['meta'] = meta
        return sample

    def _start(self):
        """(front, taken, this rank's leftover segments, rank, world_size) the next iteration reads from: the run's
        state it was given (its share), else where this rank's reading stands, else the beginning."""
        if self.rank is None and self.world_size is None:      # the launcher's process
            identity = process_identity()
            rank, world_size = identity.rank, identity.world_size
        else:
            rank, world_size = int(self.rank), int(self.world_size)
        if self._resume_state is not None:
            run = self._resume_state
            return run['next'], 0, run['leftover'][rank::world_size], rank, world_size
        if self._current_state is not None:
            cur = self._current_state
            return cur['next'], cur['taken'], cur['leftover'], rank, world_size
        return START, 0, [], rank, world_size

    def _file_names(self):
        return [os.path.basename(f) for f in self.files]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.data_path is not None:      # the synthetic byte stream: its own state, approximate
            yield from self._iter_data_path()
            return
        front, taken, leftover, rank, world_size = self._start()
        self._resume_state = None
        for window, pieces, after in exact_windows(
                self.text_len, self.files, self.text_tokenizer, self._row_groups, front=front, taken=taken,
                leftover=leftover, rank=rank, world_size=world_size, tokenizer_threads=self.tokenizer_threads,
                tokenizer_batch_size=self.tokenizer_batch_size):
            self._current_state = after
            yield self._sample(torch.tensor(window, dtype=torch.long, device=self.device), {'from': pieces})

    def _iter_data_path(self):
        base_loader = token_data_loader(
            B=self.buffer_batch_size,
            T=self.text_len,
            files=self.files,
            tokenizer_threads=self.tokenizer_threads,
            tokenizer_batch_size=self.tokenizer_batch_size,
            device=self.device,
            resume_state_dict=self._resume_state,
            data_path=self.data_path,
            vocab_size=self.vocab_size,
            rank=self.rank,
            world_size=self.world_size,
        )
        self._resume_state = None
        for batch in base_loader:
            inputs = batch['idx']
            self._current_state = batch['state_dict']
            for i in range(inputs.shape[0]):
                yield self._sample(inputs[i])

    def get_state(self) -> Optional[Dict[str, Any]]:
        if self._current_state is not None:
            return self._current_state
        if self.data_path is not None:      # before the first batch: the stream starts at offset 0
            return self._resume_state or {"step": 0, "offset": 0}
        # before the first window: this rank's state where reading will start (the loader records it as this
        # source's starting state)
        front, taken, leftover, rank, world_size = self._start()
        return rank_state(self._file_names(), front, taken, rank, world_size, leftover)

    def merge_rank_states(self, states):
        """The run's state from every rank's (fineweb.merge_exact_states); with a data_path, rank 0's."""
        if self.data_path is not None:
            return states[0]
        return merge_exact_states(states, self._row_groups)

    def __repr__(self) -> str:
        s = self.get_state()
        if s is None:
            return "text:?"
        if "next" not in s:                                # the data_path stream records step / offset instead
            return f"text:(step={s.get('step')}, offset={s.get('offset')})"
        e, f, g = s["next"]
        return f"text:(front e{e} f{f} rg{g}, entered {s.get('taken', 0)}, leftover {len(s['leftover'])})"

    def set_state(self, state: Dict[str, Any]) -> None:
        """The run's state (merge_rank_states), to read under the current rank and world size. One rank's state: a
        one-rank run's is merged and taken; a multi-rank one that has entered row groups or holds leftover is only
        that rank's part, refused; one that has neither is taken as its front with nothing left over (approximate,
        like rank 0's state before 2026-10-22). A state from before 2026-10-22 resumes at the next row group (the
        start when it had not begun one)."""
        self._current_state = None
        if self.data_path is not None:
            self._resume_state = state
            return
        if "pq_idx" in state:                               # before 2026-10-22: {files, pq_idx, rg_idx, epoch}
            e, f, g = int(state.get("epoch", 1)), int(state["pq_idx"]), state["rg_idx"]
            front = (e, f, 0) if g is None else self._row_groups.successor((e, f, int(g)))
            state = run_state(state.get("files", self._file_names()), front, [])
        elif "shard" in state:                              # one rank's state
            rank, world_size = state["shard"]
            if world_size == 1:
                state = merge_exact_states([state], self._row_groups)
            elif state["taken"] or state["leftover"]:
                raise ValueError(
                    f"this is rank {rank}'s state of a {world_size}-rank run, and it is only that rank's part of the "
                    f"reading (entered {state['taken']} row groups, {len(state['leftover'])} leftover segments). "
                    f"Restore the run's state, which merge_rank_states makes from every rank's "
                    f"(MixedDataLoader.merge_rank_snapshots; core's checkpoint saving does it)")
            else:
                state = run_state(state["files"], state["next"], [])
        self._resume_state = state

    def budget_tokens(self) -> Optional[int]:
        if self._budget_tokens is None or self._budget_tokens == 'auto':
            return None
        return int(self._budget_tokens)
