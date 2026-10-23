"""The sampler asks the model the question training scored.

Sampler.block_logits runs the current block over a KV cache holding the clean prefix, bidirectionally inside the block
(SDPA); training runs the noisy copy of the same block in a two-stream row under the batch's BlockMask (flex). Where
the two see the same thing — the clean prefix before the block's start and the block itself, the same rotary
positions, the same token types — the logits must agree. Checked at every block, fully masked and with a cell
committed, on a tiny real GPT in float32 on the CPU, where the two kernels agree to rounding. Also: block_logits
leaves the cache where it found it, and denoising commits every cell of the block.

    pytest exemplars/nano_world_model/tests/test_sampler.py
"""

import torch

from core.data.layout import collate_samples, padding_layout

from exemplars.nano_world_model import spec
from exemplars.nano_world_model.block_diffusion import BlockDiffusionSystem, two_stream_sample
from exemplars.nano_world_model.inference.sampler import Sampler
from exemplars.nano_world_model.rope3d import install_rope3d
from exemplars.nano_world_model.tests.tiny import N_VIDEO, tiny_assembly, tiny_gpt


def _training_logits(system, layout, rows, row, mask_id, pads, k):
    """Block k's video-band logits from the training forward: the sampler's row as the clean copy, its masked cells
    of block k (and only those) masked in the noisy copy."""
    s, e = rows.spans[k - 1]
    masked = torch.zeros(rows.row_len, dtype=torch.bool)
    masked[s:e] = row[s:e] == mask_id
    clean = row.clone()
    clean[masked] = rows.v_off                           # what the clean copy holds there is never seen by block k
    sample = two_stream_sample(clean, masked, masked.float(), rows, layout, mask_id, {})
    shape = (1, 2 * rows.row_len)
    batch = collate_samples([sample], padding_layout([shape[1]], shape), shape=shape, pad_values=pads)
    idx, kwargs = system.prepare_trunk_inputs(batch)
    h = system.trunk(idx, **kwargs)[0, rows.row_len + s:rows.row_len + e]
    lo, hi = layout.ranges[spec.VIDEO_TYPE_ID]
    return system.head(h).float()[:, lo:hi]


@torch.no_grad()
def test_block_logits_are_the_training_forwards():
    layout, resolver, rows, pads, mask_id = tiny_assembly()
    trunk, head = tiny_gpt(layout, rows)
    install_rope3d(trunk, rows)
    system = BlockDiffusionSystem(trunk, head, rows, 0.2, 1.0).eval()
    sampler = Sampler(system, layout, rows, mask_id, steps=3, temperature=1.0)
    g = torch.Generator().manual_seed(1)
    sampler.start(torch.randint(0, N_VIDEO, (rows.cpf,), generator=g), seed=0)
    checked = 0
    for k in range(1, rows.n_blocks + 1):
        actions = torch.randint(0, spec.N_ACTIONS, (rows.td,), generator=g)
        sampler.begin(actions)
        s, e = rows.spans[k - 1]
        # the actions driving block k sit right before it, as RowLayout lays a row out (both paths read this row)
        assert torch.equal(sampler.row[s - rows.td:s], rows.a_off + actions)
        slots = rows.action_slots[(k - 1) * rows.td:k * rows.td]
        assert torch.equal(torch.from_numpy(slots), torch.arange(s - rows.td, s))
        for r in range(2):                                  # fully masked, then with a committed cell
            got = sampler.block_logits()
            assert sampler.cache.pos == s                   # rewound: the block's K/V are gone
            want = _training_logits(system, layout, rows, sampler.row.cpu(), mask_id, pads, k)
            assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (k, r, (got - want).abs().max().item())
            checked += 1
            sampler.row[s] = rows.v_off + 1                 # a committed cell, as denoise writes one
        sampler.denoise()
        codes = sampler.commit()
        assert not (sampler.row[s:e] == mask_id).any() and ((codes >= 0) & (codes < N_VIDEO)).all()
        assert sampler.cache.pos == e
    assert checked == 2 * rows.n_blocks
