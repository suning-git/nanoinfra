"""
TextDataSource — text's sample producer (FineWeb streaming + recipe assembly),
plus text's conventional single-modality recipe defaults (TEXT_RECIPE /
create_recipe). Moved out of core: core keeps only the
mechanisms (DataSource ABC, SequenceRecipe, MixedDataLoader).
"""

from typing import Any, Dict, Iterator, Optional

import torch

from core.data.data_source import DataSource
from core.data.sequence_recipe import SequenceRecipe

from modalities.text.fineweb import token_data_loader


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

    Uses build_fixed_layout() for GPU-optimized assembly: the recipe structure
    is pre-computed once, and only the text tokens are filled in per sample.
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

        self.sequence_len = config['sequence_len']
        # TEXT_RECIPE has no constants, so overhead is just the control-token count
        self.text_len = self.sequence_len - self.recipe.overhead_tokens(self._control_resolver)

        self.split = config.get('split', 'train')
        self.buffer_batch_size = config['buffer_batch_size']
        self.tokenizer_threads = config.get('tokenizer_threads', 4)
        self.tokenizer_batch_size = config.get('tokenizer_batch_size', 128)
        device_name = config.get('device', 'cuda')
        if device_name == 'cuda' and not torch.cuda.is_available():
            device_name = 'cpu'
        self.device = torch.device(device_name)
        self.data_path = config.get('data_path')
        self.vocab_size = int(config.get('vocab_size', self.text_tokenizer.get_vocab_size()))

        # Budget for weight:auto
        self._budget_tokens = config.get('tokens')

        # Pre-compute fixed layout (token template, types, loss_mask, field positions)
        fixed = self.recipe.build_fixed_layout(
            {'text_tokens': self.text_len}, self._layout, self._control_resolver,
            content_tokenizer=self.text_tokenizer,
        )
        self._token_template = fixed['token_template'].to(self.device)
        self._type_template = fixed['token_types'].to(self.device)
        self._loss_weights_template = fixed['loss_mask'].float().to(self.device)
        self._mask_template = torch.ones(self.sequence_len, dtype=torch.long, device=self.device)

        text_start, text_end = fixed['field_slices']['text_tokens']
        self._text_start = text_start
        self._text_end = text_end

        self._current_state = None
        self._resume_state = config.get('resume_state')

        print(f"TextDataSource initialized:")
        print(f"  sequence_len={self.sequence_len}, text_len={self.text_len}")
        print(f"  buffer_batch_size={self.buffer_batch_size}, split={self.split}")
        print(f"  recipe[{self.recipe_name}] template={self.recipe.template}")
        print(f"  files={[f.rsplit('/', 1)[-1] for f in self.files]}")
        if self._resume_state is not None:
            print(f"  resume_state={self._resume_state}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
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
        )
        self._resume_state = None

        for batch in base_loader:
            inputs = batch['idx']
            self._current_state = batch['state_dict']
            B = inputs.shape[0]
            for i in range(B):
                tokens = self._token_template.clone()
                tokens[self._text_start:self._text_end] = inputs[i]

                yield {
                    'tokens': tokens,
                    'token_types': self._type_template,       # shared (read-only)
                    'attention_mask': self._mask_template,     # shared (read-only)
                    'loss_weights': self._loss_weights_template, # shared (read-only)
                }

    def get_state(self) -> Optional[Dict[str, Any]]:
        return self._current_state

    def __repr__(self) -> str:
        s = self.get_state()
        return "text:?" if s is None else f"text:(pq={s['pq_idx']}, rg={s['rg_idx']})"

    def set_state(self, state: Dict[str, Any]) -> None:
        self._resume_state = state

    def budget_tokens(self) -> Optional[int]:
        if self._budget_tokens is None or self._budget_tokens == 'auto':
            return None
        return int(self._budget_tokens)
