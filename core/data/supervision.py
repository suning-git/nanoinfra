"""
Supervision: turn one raw sample into model inputs and targets.

The recipe (SequenceRecipe) assembles a sequence and selects what is supervised (`supervision_mask`); this module
turns that into (input, target) pairs ONE SAMPLE at a time, after the mixer has picked the sample's source and before
batching, so a packed row never carries "the last token of A predicts the first of B". The loader takes
build_NTP_supervised_sample (or a project function) as build_supervised_sample_fn; a project with already-aligned
targets (e.g. a two-stream video objective) yields prepared {data, meta} samples itself and passes None.
"""

import torch

from core.tokenization.vocab_layout import VocabLayout


IGNORE = VocabLayout.IGNORE_INDEX
LEGACY_FIELDS = ("loss_mask", "loss_weights", "attention_mask")
_INT_DTYPES = (torch.int64, torch.int32)


def build_NTP_supervised_sample(tokens, supervision_mask=None, *, token_types=None, meta=None):
    """Next-token prediction for one sample: raw tokens -> {"data": {idx, targets[, token_types, target_types]},
    "meta": {...}}. Two input forms, one shift:

        build_NTP_supervised_sample(tokens, supervision_mask=None, *, token_types=None, meta=None)
        build_NTP_supervised_sample(raw_sample)   # dict: tokens, optional supervision_mask / token_types / meta

    tokens: [L_raw] integer tensor, no batch padding, at least two tokens. The prepared sample has L_raw - 1
        positions: idx = tokens[:-1], targets = tokens[1:] (to fill a row of T positions, supply T + 1 tokens).
    supervision_mask: optional bool [L_raw] aligned with the RAW tokens; True = this token is a prediction target.
        Unsupervised targets become IGNORE (an IGNOREd target's input is still real context). None = every
        predictable token is a target. After this the selection lives only in targets; the mask is not passed on.
        A mask that leaves no target after the shift (nothing selected past the first token) is refused: every
        sample this builds has at least one supervised target, which is what LMSystem.loss assumes of a micro-batch.
    token_types: optional [L_raw]; shifted into token_types (inputs) and target_types (targets).
    meta: optional per-sample information, kept as a shallow copy.

    The raw form refuses the legacy pipeline fields (loss_mask, loss_weights, attention_mask) instead of silently
    reinterpreting them, and refuses mask/types/meta given twice. Inputs are never modified: idx, token_types and
    target_types are views of the raw tensors, and so is targets unless a mask rewrote it (then it is a fresh tensor);
    collate copies every field into the batch, so nothing downstream writes back into a source's tensors.
    """
    if isinstance(tokens, dict):
        raw = tokens
        if supervision_mask is not None or token_types is not None or meta is not None:
            raise ValueError("raw-sample form: supervision_mask/token_types/meta are read from the dict; "
                             "passing them again would give one field two sources")
        legacy = [k for k in LEGACY_FIELDS if k in raw]
        if legacy:
            raise ValueError(
                f"raw sample carries legacy pipeline fields {legacy}: rename loss_mask/loss_weights to a bool "
                f"supervision_mask, and strip mechanical padding (attention_mask) in the source before NTP; "
                f"the NTP step does not reinterpret them")
        if "tokens" not in raw:
            raise ValueError("raw sample has no 'tokens'")
        tokens = raw["tokens"]
        supervision_mask = raw.get("supervision_mask")
        token_types = raw.get("token_types")
        meta = raw.get("meta")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1 or tokens.numel() < 2:
        raise ValueError("NTP input must be one 1-D sequence of at least two tokens")
    if tokens.dtype not in _INT_DTYPES:
        raise ValueError(f"tokens must be an integer tensor, got {tokens.dtype}")
    targets = tokens[1:]
    if supervision_mask is not None:
        if (not isinstance(supervision_mask, torch.Tensor) or supervision_mask.dtype != torch.bool
                or supervision_mask.shape != tokens.shape):
            raise ValueError("supervision_mask must be a bool tensor aligned with the raw tokens")
        targets = targets.masked_fill(~supervision_mask[1:], IGNORE)      # a fresh tensor; the raw tokens untouched
        if not (targets != IGNORE).any():
            # Nothing to predict: the sample only costs compute, and a micro-batch made of such samples has a mean
            # loss of 0/0. No recipe here produces one, so it is refused where it is made, not handled downstream.
            raise ValueError("no supervised target after the shift: the supervision_mask selects no token past the "
                             "first. Check the recipe and the truncation, or filter such samples in the data source")
    data = {"idx": tokens[:-1], "targets": targets}
    if token_types is not None:
        if not isinstance(token_types, torch.Tensor) or token_types.shape != tokens.shape:
            raise ValueError("token_types must align with the raw tokens")
        data["token_types"] = token_types[:-1]
        data["target_types"] = token_types[1:]
    return {"data": data, "meta": dict(meta) if meta else {}}
