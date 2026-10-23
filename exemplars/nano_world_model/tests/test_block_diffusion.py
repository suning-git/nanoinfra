"""The objective on core's pipeline: what one clip is prepared into, the batch's attention mask, and the loss.

  * A clip's sample is a function of the clip: the same (seed, epoch, row) gives the same sample bit for bit, another
    epoch other masks. Only cells of the predicted blocks are masked; the noisy copy holds [MASK] there and the target
    is the clean token, nowhere else; the weights of each block's masked cells add up to codes_per_frame (its mean CE
    over the loss's denominator); the token types are the layout's classification of the inputs.
  * The mask built from a batch's fields is the four two-stream rules written with fixed offsets into a row of length
    n (as the exemplar wrote them before it ran on core's pipeline), clip by clip, and padding sees and is seen by
    nothing — checked densely, every (query, key), on a batch with a padded row.
  * loss() is the mean over the batch's predicted blocks of each block's mean CE over its masked cells.

    pytest exemplars/nano_world_model/tests/test_block_diffusion.py
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask

from core.data.batchers import PaddingBatcher
from core.data.supervision import IGNORE
from core.model.heads import LMHead

from exemplars.nano_world_model.block_diffusion import BlockDiffusionSystem, TwoStreamPreparation, two_stream_mask
from exemplars.nano_world_model.tests.tiny import random_clips, tiny_assembly


def _prepared(n_clips=3, epoch=0, t_min=0.2, t_max=1.0):
    layout, resolver, rows, pads, mask_id = tiny_assembly()
    prep = TwoStreamPreparation(rows, layout, mask_id, t_min, t_max, seed=5)
    clips = random_clips(rows, n_clips)
    return layout, rows, pads, mask_id, prep, [prep({"tokens": c, "meta": {"row": i, "epoch": epoch}})
                                               for i, c in enumerate(clips)], clips


def test_a_clip_is_prepared_from_its_own_seed():
    layout, rows, pads, mask_id, prep, samples, clips = _prepared()
    again = [prep({"tokens": c, "meta": {"row": i, "epoch": 0}}) for i, c in enumerate(clips)]
    later = [prep({"tokens": c, "meta": {"row": i, "epoch": 1}}) for i, c in enumerate(clips)]
    for a, b, c in zip(samples, again, later):
        assert all(torch.equal(a["data"][k], b["data"][k]) for k in a["data"])
        assert not torch.equal(a["data"]["idx"], c["data"]["idx"])


def test_what_a_clip_is_prepared_into():
    layout, rows, pads, mask_id, prep, samples, clips = _prepared(n_clips=20)
    n, in_block = rows.row_len, rows.blk > 0
    for s, clean in zip(samples, clips):
        d = s["data"]
        masked = d["idx"][n:] == mask_id
        assert torch.equal(d["idx"][:n], clean)
        assert not (masked & ~in_block).any()                         # only predicted cells
        assert torch.equal(d["idx"][n:][~masked], clean[~masked])
        assert (d["targets"][:n] == IGNORE).all()
        assert torch.equal(d["targets"][n:][masked], clean[masked]) and (d["targets"][n:][~masked] == IGNORE).all()
        assert (d["weight"][:n] == 0).all() and (d["weight"][n:][~masked] == 0).all()
        for k in range(1, rows.n_blocks + 1):
            if masked[rows.blk == k].any():
                assert torch.isclose(d["weight"][n:][rows.blk == k].sum(), torch.tensor(float(rows.cpf)))
        assert torch.equal(d["token_types"], layout.classify_token_types(d["idx"]))
        assert s["meta"]["n_predicted"] == rows.n_blocks * rows.cpf


def _batch(samples, rows, pads, width):
    """The samples through core's padding batcher: one per row, rows of `width` positions."""
    return next(iter(PaddingBatcher(iter(samples), shape=(len(samples), width), pad_values=pads)))


def _rules_with_offsets(rows):
    """The two-stream rules over a fixed row of length n = row_len, [2n, 2n] bool."""
    n, blk, start = rows.row_len, rows.blk, rows.start_of
    q, kv = torch.arange(2 * n)[:, None], torch.arange(2 * n)[None, :]
    qs, qm, ks, km = q // n, q % n, kv // n, kv % n
    qb = blk[qm]
    return (((qs == 0) & (ks == 0) & (km <= qm)) | ((qs == 1) & (qb > 0) & (ks == 0) & (km < start[qm]))
            | ((qs == 1) & (qb > 0) & (ks == 1) & (blk[km] == qb)) | ((qs == 1) & (qb == 0) & (ks == 0) & (km <= qm)))


def test_the_mask_from_fields_is_the_two_stream_rules():
    layout, rows, pads, mask_id, prep, samples, clips = _prepared(n_clips=2)
    n = 2 * rows.row_len
    width = n + 128                                                    # a padded tail on every row
    batch = _batch(samples, rows, pads, width)
    mask_mod = two_stream_mask(batch, create_block_mask).mask_mod
    q, kv = torch.arange(width)[:, None].expand(width, width), torch.arange(width)[None, :].expand(width, width)
    ref = _rules_with_offsets(rows)
    for b in range(2):
        got = mask_mod(torch.tensor(b), torch.tensor(0), q, kv)
        assert torch.equal(got[:n, :n], ref)
        assert not got[n:, :].any() and not got[:, n:].any()          # padding: sees nothing, seen by nothing


class _Trunk(torch.nn.Module):
    """Stands in for the GPT: an embedding of the inputs (the loss's arithmetic does not depend on attention)."""

    def __init__(self, vocab):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, 8)
        self.config = None

    def forward(self, idx, token_types=None, attention=None, position_ids=None):
        return self.emb(idx)


def test_the_loss_is_the_mean_of_block_means():
    layout, rows, pads, mask_id, prep, samples, clips = _prepared(n_clips=3)
    batch = _batch(samples, rows, pads, 2 * rows.row_len)
    head = LMHead(8, layout.vocab_size)
    torch.nn.init.normal_(head.lm_head.weight, std=1.0)
    system = BlockDiffusionSystem(_Trunk(layout.vocab_size), head, rows, 0.2, 1.0, mask_builder=create_block_mask)
    got = system.loss(batch)
    n, block_means = rows.row_len, []
    for b in range(3):
        ce = torch.nn.functional.cross_entropy(head(system.trunk(batch["idx"][b:b + 1]))[0].float(),
                                               batch["targets"][b].clamp_min(0), reduction="none")
        for k in range(1, rows.n_blocks + 1):
            cells = torch.zeros(2 * n, dtype=torch.bool)
            cells[n:] = (rows.blk == k) & (batch["targets"][b, n:] != IGNORE)
            block_means.append(ce[cells].mean() if cells.any() else torch.tensor(0.0))
    want = torch.stack(block_means).mean()
    assert torch.allclose(got, want, rtol=1e-5), (got.item(), want.item())
