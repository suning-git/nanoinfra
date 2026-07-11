"""
MixedDataLoader: mix multiple DataSources with configurable weights.

Implements the standard duck-typing loader interface:
    - Infinite __iter__ yielding {idx, token_types, targets, attention_mask, state_dict}
    - get_state() / set_state() for checkpoint resume

Batch allocation uses Bresenham-style accumulation to guarantee exact long-term
ratios while keeping each individual batch close to the target proportions.

Weight specification per source (default: auto):
    - weight: auto      — auto-compute from budget (requires tokens or epochs)
    - weight: <float>   — explicit proportion (absolute in mixed mode, relative in all-explicit)

Three weight modes (auto-detected):
    - All auto:     proportional by budget, total_budget = sum(budgets)
    - All explicit: normalize as proportions, no budget (must set max_steps manually)
    - Mixed:        explicit weights are absolute fractions (each < 1, sum < 1),
                    auto sources share the remainder proportionally by budget.
                    total_budget = auto_budget_sum / (1 - explicit_sum)

Example config (mixed mode):
    sources:
      - type: text
        weight: auto
        tokens: 10_000_000_000      # chinchilla budget → auto weight
      - type: motion
        weight: 0.05                # fixed 5% of each batch
        epochs: ...                 # NOT allowed with explicit weight
"""

import torch
from typing import Dict, Any, List, Optional, Iterator, Type

from core.data.data_source import DataSource
from core.data.supervision import NextTokenPrediction


class MixedDataLoader:
    """
    Mixed data loader combining multiple DataSources.

    Trainer interface (duck typing):
        - iter(loader) → infinite iterator of batches
        - Each batch: {idx, token_types, targets, [attention_mask], state_dict}
        - get_state() → checkpoint state
        - set_state(state) → resume from checkpoint

    Weight specification per source (default: auto):
        - weight: auto    — auto-compute from source budget (tokens or epochs)
        - weight: <float> — explicit proportion (absolute in mixed mode, relative otherwise)
    See module docstring for details on the three weight modes.
    """

    def __init__(self, loader_config: Dict[str, Any], tokenizers: Dict[str, Any],
                 source_types: Dict[str, Type[DataSource]],
                 resume_state_dict: Optional[Dict] = None):
        data_config = loader_config['data']
        self.sequence_len = data_config['sequence_len']
        self.batch_size = loader_config.get('batch_size', 16)

        # Create sources from config using orchestrator-provided type mapping
        self.sources: List[DataSource] = []
        source_configs = data_config['sources']
        for sc in source_configs:
            source_type = sc['type']
            if source_type not in source_types:
                raise ValueError(
                    f"Unknown data source type: {source_type}. "
                    f"Available: {list(source_types.keys())}"
                )
            config = {k: v for k, v in sc.items() if k not in ('type', 'weight')}
            source = source_types[source_type](config, tokenizers)
            self.sources.append(source)

        # Compute weights
        self.weights, self.total_budget_tokens = self._compute_weights(source_configs)

        # Bresenham accumulator for deterministic batch allocation
        self._accum = [0.0] * len(self.sources)

        # Supervision: hardcoded to NextTokenPrediction (internal implementation detail)
        self.supervision = NextTokenPrediction()

        self._current_state: Optional[Dict[str, Any]] = None

        # Resume
        if resume_state_dict is not None:
            self.set_state(resume_state_dict)

        # Log
        print(f"MixedDataLoader initialized:")
        print(f"  batch_size={self.batch_size}, sequence_len={self.sequence_len}")
        print(f"  supervision={self.supervision.__class__.__name__}")
        for i, (sc, w) in enumerate(zip(source_configs, self.weights)):
            budget = self.sources[i].budget_tokens()
            budget_str = f", budget={budget:,.0f} tokens" if budget is not None else ""
            print(f"  source[{i}] type={sc['type']}, weight={w:.4f}{budget_str}")
        if self.total_budget_tokens is not None:
            print(f"  total_budget={self.total_budget_tokens:,.0f} tokens")

    # ------------------------------------------------------------------
    # Weight computation & training budget
    # ------------------------------------------------------------------

    def _compute_weights(self, source_configs: List[Dict]) -> tuple:
        """
        Compute mixing weights from source configs.

        Three modes (auto-detected):
            All auto:     normalize by budget.
            All explicit: normalize by raw values.
            Mixed:        explicit weights are absolute fractions,
                          auto sources share (1 - sum(explicit)) by budget.

        Returns:
            (weights, total_budget_tokens):
                weights: list of floats summing to 1.0
                total_budget_tokens: estimated total training tokens, or None
        """
        n = len(source_configs)
        auto_entries = []   # [(index, budget)]
        explicit_entries = []  # [(index, weight)]

        for i, (source, sc) in enumerate(zip(self.sources, source_configs)):
            w = sc.get('weight', 'auto')
            if w == 'auto':
                budget = source.budget_tokens()
                if budget is None:
                    raise ValueError(
                        f"source[{i}] type={sc['type']} has weight=auto but no budget "
                        f"(set 'tokens' or 'epochs')"
                    )
                auto_entries.append((i, budget))
            else:
                w = float(w)
                if source.budget_tokens() is not None:
                    raise ValueError(
                        f"source[{i}] type={sc['type']} has explicit weight={w} "
                        f"and a token budget — this is ambiguous. "
                        f"Use weight=auto to derive weight from budget, "
                        f"or remove the budget (tokens/epochs) for explicit weight."
                    )
                explicit_entries.append((i, w))

        has_auto = len(auto_entries) > 0
        has_explicit = len(explicit_entries) > 0

        weights = [0.0] * n

        if has_auto and has_explicit:
            # Mixed mode: explicit are absolute fractions, auto share the rest
            explicit_sum = sum(w for _, w in explicit_entries)
            if explicit_sum >= 1.0:
                raise ValueError(
                    f"Sum of explicit weights ({explicit_sum:.4f}) must be < 1.0 "
                    f"to leave room for auto sources"
                )
            for i, w in explicit_entries:
                if w < 0:
                    raise ValueError(f"source[{i}] explicit weight must be >= 0, got {w}")
                weights[i] = w

            auto_remaining = 1.0 - explicit_sum
            auto_budget_sum = sum(b for _, b in auto_entries)
            if auto_budget_sum > 0:
                for i, budget in auto_entries:
                    weights[i] = auto_remaining * (budget / auto_budget_sum)
            # total_budget: tokens needed to exhaust all auto budgets at current weights
            total_budget = int(auto_budget_sum / auto_remaining) if auto_remaining > 0 else None

        elif has_auto:
            # All auto: proportional by budget
            auto_budget_sum = sum(b for _, b in auto_entries)
            if auto_budget_sum > 0:
                for i, budget in auto_entries:
                    weights[i] = budget / auto_budget_sum
            total_budget = auto_budget_sum

        else:
            # All explicit: normalize as relative proportions
            raw_sum = sum(w for _, w in explicit_entries)
            if raw_sum <= 0:
                raise ValueError("Sum of explicit weights must be > 0")
            for i, w in explicit_entries:
                weights[i] = w / raw_sum
            total_budget = None

        return weights, total_budget

    def estimate_max_steps(self, total_batch_size: int) -> Optional[int]:
        """Estimate training steps from total budget. Returns None if no budget info."""
        if self.total_budget_tokens is None:
            return None
        return int(self.total_budget_tokens // total_batch_size)

    # ------------------------------------------------------------------
    # Batch allocation
    # ------------------------------------------------------------------

    def _allocate_batch(self) -> List[int]:
        """
        Bresenham-style allocation: accumulate fractional slots, take integer part.

        Guarantees sum(counts) == batch_size and exact long-term proportions.
        """
        counts = []
        for i in range(len(self.sources)):
            self._accum[i] += self.weights[i] * self.batch_size
            c = int(self._accum[i])
            self._accum[i] -= c
            counts.append(c)

        # Distribute any deficit to sources with largest fractional remainders
        deficit = self.batch_size - sum(counts)
        if deficit > 0:
            order = sorted(range(len(self.sources)),
                           key=lambda i: self._accum[i], reverse=True)
            for i in order[:deficit]:
                counts[i] += 1
                self._accum[i] -= 1.0

        return counts

    # ------------------------------------------------------------------
    # Iteration (trainer interface)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        source_iters = [iter(s) for s in self.sources]

        while True:
            counts = self._allocate_batch()

            # Gather individual samples from each source
            samples = []
            for src_idx, count in enumerate(counts):
                for _ in range(count):
                    samples.append(next(source_iters[src_idx]))

            # Collate — all samples are [sequence_len], just stack
            batch_tokens = torch.stack([s['tokens'] for s in samples])
            batch_types = torch.stack([s['token_types'] for s in samples])
            batch_mask = torch.stack([s['attention_mask'] for s in samples])
            batch_loss_weights = torch.stack([s['loss_weights'] for s in samples])

            # loss_weights flows through for binary masking (0/1 from supervision).
            # Per-token-TYPE loss weighting is NOT a core concern — a project supplies a
            # weighted head variant.

            # Apply supervision (e.g., next-token-prediction shift)
            result = self.supervision.apply(
                batch_tokens, batch_types, batch_mask, batch_loss_weights
            )

            # Attach state for Trainer checkpointing
            self._current_state = {
                'source_states': [s.get_state() for s in self.sources],
                'accum': list(self._accum),
            }
            result['state_dict'] = self._current_state

            yield result

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get current state for checkpointing."""
        return self._current_state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Resume from checkpoint state."""
        for source, ss in zip(self.sources, state['source_states']):
            if ss is not None:
                source.set_state(ss)
        self._accum = list(state.get('accum', [0.0] * len(self.sources)))

    def __repr__(self) -> str:
        """Compact repr for checkpoint logging. Each source renders its own state
        (core stays modality-agnostic — it neither names nor sniffs a modality)."""
        parts = [repr(source) for source in self.sources]
        return f"MixedDataLoader state: {', '.join(parts)}"
