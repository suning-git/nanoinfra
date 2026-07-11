"""
Supervision Strategies.

Service layer for defining (input, target) pairs from sequences.
Dataloaders instantiate these internally — not user-configurable via YAML.

Two strategies:
- NextTokenPrediction: shift by 1, single sequence (standard LM)
- AlignedSupervision: no shift, supports separate input/target sequences
"""

import torch
from typing import Dict, Optional

from core.tokenization.vocab_layout import VocabLayout


class SupervisionStrategy:
    """
    Base class for supervision strategies.

    Defines how to generate (input, target) pairs from sequences.

    Responsibilities:
    - Input/target shifting (e.g., next token prediction) or alignment
    - Loss masking (which positions to supervise)
    - Loss weighting (different weights for different positions)
    """

    def apply(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        target_token_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply supervision strategy to generate training batch.

        Args:
            tokens: [B, L] Input token sequence
            token_types: [B, L] Input token types (0=text, 1=motion, 2=control)
            attention_mask: [B, L] Optional attention mask (1=valid, 0=padding)
            loss_weights: [B, L] Optional per-token loss weights (float).
                0.0 = ignore, >0 = supervise with given weight.
            target_tokens: [B, L] Optional separate target sequence.
                If None, targets are derived from tokens (e.g., via shift).
            target_token_types: [B, L] Optional target token types.
                If None, derived from token_types.

        Returns:
            dict with keys:
              input-aligned (position i = conditioning token):
                - idx: [B, T] Input token IDs
                - token_types: [B, T] Input token types
              target-aligned (position i = predicted token):
                - targets: [B, T] Target token IDs
                - target_types: [B, T] Target token types
                - attention_mask: [B, T] (optional) Attention mask
                - loss_weights: [B, T] (optional) Per-token loss weights
        """
        raise NotImplementedError


class NextTokenPrediction(SupervisionStrategy):
    """
    Standard causal language modeling: predict next token at every position.

    This is the default supervision strategy used in standard LM training.

    Behavior:
        Input:  [tok_0, tok_1, tok_2, ..., tok_T-1]
        Target: [tok_1, tok_2, tok_3, ..., tok_T]

    At each position i, the model sees tokens [0..i] and predicts token i+1.

    Note: target_tokens parameter is ignored — targets are always derived
    from tokens via shift.

    Example:
        >>> strategy = NextTokenPrediction()
        >>> sequence = torch.tensor([[1, 2, 3, 4, 5, 6]])  # [B=1, L=6]
        >>> token_types = torch.tensor([[0, 0, 0, 0, 0, 0]])
        >>> batch = strategy.apply(sequence, token_types)
        >>> batch['idx']      # [1, 2, 3, 4, 5]
        >>> batch['targets']  # [2, 3, 4, 5, 6]
    """

    def apply(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        target_token_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply next-token prediction shifting.

        Args:
            tokens: [B, L] Full sequence
            token_types: [B, L] Token types
            attention_mask: [B, L] Optional attention mask (1=valid, 0=padding)
            loss_weights: [B, L] Optional per-token loss weights (float).
                0.0 = ignore, >0 = supervise with given weight.
            target_tokens: Ignored (targets derived from tokens via shift)
            target_token_types: Ignored

        Returns:
            dict with shifted input/target pairs, plus loss_weights if provided
        """
        # Standard causal LM shifting: input is [:, :-1], target is [:, 1:]
        # After shift, two alignment groups exist in the result:
        #   input-aligned:  idx, token_types       — type/id of the conditioning token
        #   target-aligned: targets, target_types, loss_weights — type/id/weight of the predicted token
        idx = tokens[:, :-1]                # [B, L-1] positions 0 to L-2
        token_types_input = token_types[:, :-1]   # [B, L-1] input-aligned
        token_types_target = token_types[:, 1:]   # [B, L-1] target-aligned
        targets = tokens[:, 1:]             # [B, L-1] positions 1 to L-1 (shifted)

        result = {
            'idx': idx,
            'token_types': token_types_input,
            'target_types': token_types_target,
            'targets': targets,
        }

        # Handle attention mask if provided (for variable-length sequences)
        if attention_mask is not None:
            # For targets: we need to mask positions where the TARGET is padding
            # Targets are at positions [1, L-1], so use mask[:, 1:]
            target_mask = attention_mask[:, 1:]  # [B, L-1]

            # Use target_mask for attention_mask as well
            # This ensures positions with padding targets are masked out from training
            # (position i should only be trained if target[i] is valid)
            result['attention_mask'] = target_mask

            # Mask out padding positions in targets
            targets = torch.where(
                target_mask.bool(),
                targets,
                torch.full_like(targets, VocabLayout.IGNORE_INDEX)
            )
            result['targets'] = targets

        # Handle loss_weights (semantic: experimenter's supervision + weighting choice)
        # Composes with attention_mask: positions with weight==0 get targets=IGNORE_INDEX
        if loss_weights is not None:
            shifted_weights = loss_weights[:, 1:]  # align with targets
            targets = result['targets']
            result['targets'] = torch.where(
                shifted_weights > 0,
                targets,
                torch.full_like(targets, VocabLayout.IGNORE_INDEX),
            )
            result['loss_weights'] = shifted_weights

        return result


class AlignedSupervision(SupervisionStrategy):
    """
    Aligned supervision: input[i] → target[i], no shift.

    Used for reconstruction tasks where input and target are aligned:
    - Same sequence: input = target (e.g., masked prediction with causal attention)
    - Different sequences: input ≠ target (e.g., masked motion → full motion)

    Attention is still controlled by the model (causal or full), not by this class.

    Example (same sequence):
        >>> strategy = AlignedSupervision()
        >>> tokens = torch.tensor([[1, 2, 3, 4, 5]])
        >>> token_types = torch.tensor([[0, 0, 0, 0, 0]])
        >>> batch = strategy.apply(tokens, token_types)
        >>> batch['idx']      # [1, 2, 3, 4, 5]
        >>> batch['targets']  # [1, 2, 3, 4, 5]

    Example (different sequences):
        >>> strategy = AlignedSupervision()
        >>> input_tokens = torch.tensor([[1, 2, 3, 4, 5]])   # masked
        >>> target_tokens = torch.tensor([[6, 7, 8, 9, 10]]) # full
        >>> batch = strategy.apply(input_tokens, ..., target_tokens=target_tokens, ...)
        >>> batch['idx']      # [1, 2, 3, 4, 5]
        >>> batch['targets']  # [6, 7, 8, 9, 10]
    """

    def apply(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        target_token_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply aligned supervision (no shift).

        Args:
            tokens: [B, L] Input sequence
            token_types: [B, L] Input token types
            attention_mask: [B, L] Optional attention mask (1=valid, 0=padding)
            loss_weights: [B, L] Optional per-token loss weights (float).
                0.0 = ignore, >0 = supervise with given weight.
            target_tokens: [B, L] Optional separate target sequence.
                If None, uses tokens as targets.
            target_token_types: [B, L] Optional target token types.
                If None, uses token_types.

        Returns:
            dict with aligned input/target pairs
        """
        # No shift — input and target are aligned
        idx = tokens
        targets = target_tokens if target_tokens is not None else tokens
        target_types = target_token_types if target_token_types is not None else token_types

        result = {
            'idx': idx,
            'token_types': token_types,
            'target_types': target_types,
            'targets': targets,
        }

        # Handle attention mask if provided
        if attention_mask is not None:
            result['attention_mask'] = attention_mask

            # Mask out padding positions in targets
            targets = torch.where(
                attention_mask.bool(),
                result['targets'],
                torch.full_like(result['targets'], VocabLayout.IGNORE_INDEX)
            )
            result['targets'] = targets

        # Handle loss_weights
        if loss_weights is not None:
            targets = result['targets']
            result['targets'] = torch.where(
                loss_weights > 0,
                targets,
                torch.full_like(targets, VocabLayout.IGNORE_INDEX),
            )
            result['loss_weights'] = loss_weights

        return result
