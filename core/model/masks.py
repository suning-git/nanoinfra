"""Building a BlockMask for packed rows: the shared pieces.

flex attention cuts query and key positions into blocks of FLEX_BLOCK = 128 and computes one 128 x 128 tile at a
time; a BlockMask labels each tile as fully hidden (skipped), fully visible, or partial, and for a partial tile flex
calls the visibility rule `rule(b, h, q, kv) -> bool` on every (q, kv) pair inside it. Two consequences every
caller meets, and that are easy to get wrong, live here once:

  - A row whose length T is not a multiple of 128 has a last tile reaching past T (T = 255: indices up to 255), so
    any per-position field a rule READS must be padded to the block multiple first. That is pad_to_flex_block; the
    padding is only so the rule can read those indices — the model's input stays T long.
  - "Same sample" is the rule that keeps packed samples apart: a query sees a key only when both carry the same
    doc_id and that doc_id is a real sample (>= 0), so padding neither sees nor is seen. same_sample builds it from
    a batch's doc_id, padding included. Combine it with a project's own rules by torch's and_masks.

default_mask_builder(device) is what core's packed AR preparation builds with unless a caller passes its own: on
CUDA one compiled create_block_mask, built once per process and reused (uncompiled, building a mask costs
milliseconds per batch, tens of milliseconds for long rows); each batch still builds its own mask from its own
doc_id, and the first call or a new shape may compile. On the CPU, torch's create_block_mask as it is.

    from torch.nn.attention.flex_attention import and_masks
    rule = and_masks(same_sample(doc_id), causal)
    mask = default_mask_builder(device)(rule, B=B, H=None, Q_LEN=T, KV_LEN=T, device=device)
"""
import torch
from torch.nn.attention.flex_attention import create_block_mask

FLEX_BLOCK = 128

_compiled_create_block_mask = None


def pad_to_flex_block(x, value):
    """A [B, T] per-position field padded on the right to the next multiple of FLEX_BLOCK, new positions = `value`
    (returned as it is when T already is a multiple). For the fields a visibility rule reads, never for the model's
    inputs."""
    T = x.shape[1]
    Tp = (T + FLEX_BLOCK - 1) // FLEX_BLOCK * FLEX_BLOCK
    return x if Tp == T else torch.nn.functional.pad(x, (0, Tp - T), value=value)


def same_sample(doc_id):
    """The rule "same sample": a query sees a key when both positions belong to the same real sample (doc_id >= 0
    and equal). doc_id is a batch's [B, T] int64 field; it is padded here, so the rule may be evaluated anywhere in
    the last tile."""
    doc = pad_to_flex_block(doc_id, -1)

    def rule(b, h, q, kv):
        return (doc[b, q] >= 0) & (doc[b, q] == doc[b, kv])
    return rule


def causal(b, h, q, kv):
    """The rule "not after me": a query sees itself and earlier positions."""
    return q >= kv


def default_mask_builder(device):
    """create_block_mask, compiled when `device` is CUDA (one compiled function per process, reused)."""
    global _compiled_create_block_mask
    if torch.device(device).type != "cuda":
        return create_block_mask
    if _compiled_create_block_mask is None:
        _compiled_create_block_mask = torch.compile(create_block_mask)
    return _compiled_create_block_mask
