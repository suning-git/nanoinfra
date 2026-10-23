"""
block_diffusion.py — the objective: absorbing-state (masked) diffusion over blocks.

Each latent frame is one diffusion BLOCK. The forward process masks each token of a
block independently with probability t, where t is drawn per block from U[t_min,
t_max]; the model sees the clean prefix plus the partially masked block and predicts
the CLEAN token AT each masked position. Same trunk as an autoregressive model,
different head convention: predict-here, not predict-next.

THE TRAINING LOSS. Under MDLM's linear schedule the per-token NELBO is E_t[ mean CE
over the masked cells ], and the loss estimates it block by block from the cells the
draw actually masked:

    mean over predicted blocks of ( sum of CE over the block's masked cells / their number )

MDLM writes the estimate with a 1/t weight instead, (1/t) * sum(CE over masked) /
predicted tokens, which divides each block by the EXPECTED number of masked cells: also
an estimate of the NELBO, but each block's term then swings with how many cells this
draw happened to mask (in a 256-cell block about 12% at t = 0.2), with nothing about the
model behind the swing. The per-block mean has no such swing. Its expectation is not
exactly MDLM's: for a block of n cells the two differ by a term of weight 1/(n + 1) at
t_min (0.4% at n = 256). t is clipped away from 0 because the unclipped schedule has
enormous gradient variance (BD3-LM).

The validation number is the same quantity on a fixed grid of t with fixed masks
(evaluator.py), so it reads in nats per predicted token.

THE TWO-STREAM ROW (BD3-LM's vectorized trick). A clip is trained as

    [ clean row | noisy row ]        2 x row_len positions

so one forward pass gives every block both a clean prefix and its own noised content.
Who may see whom:
    clean -> clean      causal — an ordinary prefix-providing pass
    noisy_k -> clean    strictly before block k's start (its own clean block EXCLUDED,
                        which would otherwise hand the model its own labels)
    noisy_k -> noisy    its own block, bidirectionally — the masked cells co-denoise
    noisy_cond -> clean the noisy copy's other positions, causal over the clean copy;
                        their outputs are never read
Both copies of a position take the same rotary position (rope3d.py): the noisy copy
of a cell is rotated like the clean cell it stands for.

PER CLIP, ON CORE'S PIPELINE. A clip is noised when it is prepared, one at a time
before batching (TwoStreamPreparation, the build_supervised_sample_fn of core's loader,
core/data/mixed_dataloader.py), from a generator seeded by the clip itself — (seed, epoch, row) — so
a clip is noised the same way on any rank, in any batch and after a resume, and no two
clips share their noise. The prepared sample carries everything the step needs: its
inputs, targets, loss weights and token types, and the per-position facts the attention
rules read (stream, block, block start, position in the row). The mask is built from the
batch (two_stream_mask), not from offsets fixed in advance.

Two implementation notes that are load-bearing rather than stylistic:

  * Hidden states are sliced to the masked positions BEFORE the head, so its work and
    memory scale with the number of masked positions rather than the full row (in
    evaluation too: evaluator.py).
  * The per-token CE is the head's own `loss_per_token`, and train_wm.py builds the
    head with head_ce="compiled" to reduce CE peak memory. The saving depends on
    shape and software stack; compilation does not guarantee that no [M, V]
    tensor is materialized. core/tests/unit/test_heads_per_token.py checks a fixed
    shape's peak memory and agreement with eager under nonuniform token weights.
    Liger's current fused linear CE backward does not support those weights, so
    the liger arm refuses this objective while gradients are recorded.
"""

import hashlib

import torch
from torch.nn.attention.flex_attention import and_masks

from core.data.supervision import IGNORE
from core.model.masks import default_mask_builder, pad_to_flex_block, same_sample
from core.model.system import LMSystem


def masked_mean_weight(masked, blk, n_blocks, cpf):
    """The loss weight of each cell: cpf / (the masked cells of its block) where masked, 0 elsewhere. Summed against
    the loss's denominator (cpf per predicted block), each block then counts the mean CE over its masked cells.
    masked [n] bool; blk [n], each position's block (0 = outside the predicted blocks, never masked)."""
    count = torch.zeros(n_blocks + 1).scatter_add_(0, blk, masked.float())
    return torch.where(masked, cpf / count.clamp_min(1.0)[blk], 0.0)


def stream_fields(rows):
    """The per-position facts of one clip's two-stream row, which the attention rules read: stream (0 clean, 1 noisy),
    block (0 = outside the predicted blocks), block_start (where that block starts), pos (the position in the row,
    the same in both copies — also the rotary position). CPU int64 tensors of 2 x row_len."""
    n = rows.row_len
    pos = torch.arange(n)
    return {"stream": torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)]),
            "block": torch.cat([rows.blk, rows.blk]), "block_start": torch.cat([rows.start_of, rows.start_of]),
            "pos": torch.cat([pos, pos])}


def two_stream_sample(clean, masked, weight, rows, vocab, mask_id, meta):
    """One clip's prepared sample, its draws given: [clean | noisy] with the masked cells -> mask_id, the clean token
    as the target at each masked cell of the noisy copy (IGNORE elsewhere), `weight` there (0 elsewhere), the token
    types, and the stream fields. clean [row_len] int64, masked [row_len] bool, weight [row_len] float (read only where
    masked). meta gains n_predicted, the clip's predicted positions: what the loss divides by."""
    n = rows.row_len
    idx = torch.cat([clean, clean.masked_fill(masked, mask_id)])
    targets = torch.full((2 * n,), IGNORE, dtype=torch.long)
    targets[n:][masked] = clean[masked]
    w = torch.zeros(2 * n)
    w[n:][masked] = weight[masked]
    data = {"idx": idx, "targets": targets, "weight": w, "token_types": vocab.classify_token_types(idx),
            **stream_fields(rows)}
    return {"data": data, "meta": {**meta, "n_predicted": rows.n_blocks * rows.cpf}}


class TwoStreamPreparation:
    """core's build_supervised_sample_fn for block diffusion: one raw clip (dataset.ClipSource) -> one prepared
    two-stream sample. From a generator seeded by the clip: one t per predicted block from U[t_min, t_max], then an
    independent Bernoulli(t) per cell of the predicted blocks (the given frame, the actions and the delimiters are never
    masked); a masked cell weighs masked_mean_weight's share of its block.

    Args:
        rows: RowLayout; vocab: VocabLayout (token types); mask_id: global id of [MASK], the absorbing state;
        t_min, t_max: the noise-level range; seed: the run's seed, mixed into every clip's."""

    def __init__(self, rows, vocab, mask_id, t_min, t_max, seed=0):
        self.rows, self.vocab = rows, vocab
        self.mask_id, self.t_min, self.t_max, self.seed = int(mask_id), float(t_min), float(t_max), int(seed)

    def noise_seed(self, meta):
        """A 64-bit hash of (seed, epoch, row): stable across processes and ranks (python's hash() is salted per
        process), different for different clips and for one clip's different epochs."""
        key = f"{self.seed}/{meta['epoch']}/{meta['row']}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")

    def __call__(self, raw):
        rows, clean = self.rows, raw["tokens"]
        g = torch.Generator().manual_seed(self.noise_seed(raw["meta"]))
        t_blk = self.t_min + (self.t_max - self.t_min) * torch.rand(rows.n_blocks + 1, generator=g)
        t_pos = t_blk[rows.blk]                       # column 0 unused: no predicted block is block 0
        masked = (torch.rand(rows.row_len, generator=g) < t_pos) & (rows.blk > 0)
        weight = masked_mean_weight(masked, rows.blk, rows.n_blocks, rows.cpf)
        return two_stream_sample(clean, masked, weight, rows, self.vocab, self.mask_id, raw["meta"])


def pad_values(eos_id, eos_type):
    """One per data field (core's collate refuses to guess): a padding position is an eos that is never a target and
    weighs nothing; the batch's doc_id = -1 there keeps it out of every attention rule."""
    return {"idx": int(eos_id), "targets": IGNORE, "weight": 0.0, "token_types": int(eos_type),
            "stream": 0, "block": 0, "block_start": 0, "pos": 0}


def two_stream_mask(batch, mask_builder):
    """The batch's BlockMask: "same clip" (core's same_sample on doc_id) AND the four two-stream rules (module
    docstring), read from the batch's fields. A padding position (doc_id -1) sees and is seen by nothing. The fields
    the rules read are padded to the flex tile by core's pad_to_flex_block, so a rule may be evaluated anywhere in the
    last tile."""
    doc = batch["doc_id"]
    B, T = doc.shape
    st, bk, bs, ps = (pad_to_flex_block(batch[k], 0) for k in ("stream", "block", "block_start", "pos"))

    def two_streams(b, h, q, kv):
        qs, ks, qp, kp, qb = st[b, q], st[b, kv], ps[b, q], ps[b, kv], bk[b, q]
        clean_causal = (qs == 0) & (ks == 0) & (kp <= qp)
        noisy_prefix = (qs == 1) & (qb > 0) & (ks == 0) & (kp < bs[b, q])
        noisy_self = (qs == 1) & (qb > 0) & (ks == 1) & (bk[b, kv] == qb)
        noisy_cond = (qs == 1) & (qb == 0) & (ks == 0) & (kp <= qp)
        return clean_causal | noisy_prefix | noisy_self | noisy_cond

    return mask_builder(and_masks(same_sample(doc), two_streams), B=B, H=None, Q_LEN=T, KV_LEN=T, device=doc.device)


class BlockDiffusionSystem(LMSystem):
    """An LMSystem whose `loss(batch)` is the block-diffusion loss, not next-token CE.

    This is how core says to train a different objective: "Projects that want a
    different composition (e.g. a diffusion head, or multiple heads) write their own
    System satisfying the same `loss(batch)` contract — core does not change"
    (core/model/system.py). So core's Trainer runs this unmodified; there is no
    training-loop subclass anywhere in this project.

    It SUBCLASSES LMSystem rather than wrapping one, which matters: the same two
    submodules are registered, so `state_dict()` keys stay `trunk.*` / `head.*` and a
    checkpoint written here loads into any other System over the same trunk.

    prepare_trunk_inputs is the batch's translation into trunk arguments — the mask from
    its fields, the rotary positions from `pos`, its token types — and core's evaluation
    calls it too (evaluator.py), so training and evaluation run one translation.

    Args:
        trunk, head: from build_system; rows: RowLayout (the FLOPs estimate's mask);
        t_min, t_max: the noise-level range (the FLOPs estimate's expected masked share);
        mask_builder: builds the BlockMask, default core's (compiled create_block_mask on CUDA).
    """

    def __init__(self, trunk, head, rows, t_min, t_max, mask_builder=None):
        super().__init__(trunk, head)
        self.rows, self.t_min, self.t_max = rows, float(t_min), float(t_max)
        self.mask_builder = mask_builder

    def prepare_trunk_inputs(self, batch):
        build = self.mask_builder or default_mask_builder(batch["doc_id"].device)
        return batch["idx"], {"token_types": batch["token_types"], "attention": two_stream_mask(batch, build),
                              "position_ids": batch["pos"]}

    def loss(self, batch):
        """sum over masked cells of weight x CE / the batch's predicted positions: with masked_mean_weight, the mean
        over the batch's predicted blocks of each block's mean CE over its masked cells."""
        idx, kwargs = self.prepare_trunk_inputs(batch)
        hidden = self.trunk(idx, **kwargs)
        sel = batch["targets"] != IGNORE
        ce = self.head.loss_per_token(hidden[sel], batch["targets"][sel])
        n_predicted = sum(m["n_predicted"] for m in batch["sample_meta"])
        return (ce * batch["weight"][sel]).sum() / n_predicted

    def estimate_flops(self):
        """FLOPs/token, corrected — this objective breaks LMSystem's contract twice.

        The contract (core/model/system.py) is that every position counted in
        total_batch_size costs the same. Block diffusion computes on a DIFFERENT set
        of positions than it is charged for, in two independent ways:

          head       runs only where a token was masked. Charged for all 2*row_len
                     positions, it runs on E[t] * in-block of them — about 22% here.
          attention  runs under the two-stream mask, not dense causal over
                     sequence_len, which is what the trunk's formula assumes.

        Left uncorrected the two together read 1.78x high, and MFU printed 101.6% —
        above the card's physical ceiling, which is the only reason anyone looked.
        Corrected it read ~57% (about 51% on today's code, RESULTS.md), and that gap is
        the point: 101.6% says the card is saturated and there is nothing to win, 57%
        says a third of it is idle.

        The head correction shrinks as the trunk grows; the attention one does NOT.
        At a 129-frame window the head is down to ~15% of the uncorrected total while
        attention is ~67% (the mask keeps 27% of the tiles there), and the combined
        error is 2.5x. So this is not a small-model wart to outgrow.

        On the two denominators, because it is easy to get wrong (it was): the trunk's
        attention term counts a FULL sequence_len x sequence_len grid — the PaLM
        appendix-B / nanoGPT convention, which does not credit the causal half either.
        `BlockMask.sparsity()` also reports against the full grid, so the two match and
        `density` is the right multiplier. Measuring the mask against a CAUSAL grid
        instead gives 0.64 rather than 0.33 and understates the correction by ~2x.

        One consequence to state rather than hide: this MFU is physically true, while
        the repo's dense-causal numbers carry the convention's overcount. At text's
        sequence lengths that overcount is ~3% of the total and nobody cares; here it
        would have been 3x on the attention term. So do not read this number against
        the text exemplar's and conclude anything — MFU was never comparable across
        architectures anyway, and now it is not comparable across conventions either.

        Both factors are EXPECTATIONS, not per-step truths: t is redrawn for every
        block, so the exact figure moves batch to batch. MFU is a display metric and an
        expectation is the right resolution for it. The mask is one clip's, a full row
        of the batch; a row holding one clip, as train_wm.py batches, is exactly that.
        """
        cfg = self.trunk.config
        head = 6 * sum(p.numel() for p in self.head.parameters())
        # Mirrors the attention term in GPT.estimate_flops (core/model/gpt.py). It is
        # spelled out again because it has to be SUBTRACTED before being rescaled, and
        # the trunk hands back only the sum. If that formula changes, this changes.
        q = cfg.n_embd // cfg.n_head
        attn = 12 * cfg.n_layer * cfg.n_head * q * cfg.sequence_len

        e_t = 0.5 * (self.t_min + self.t_max)               # t ~ U[t_min, t_max]
        in_block = int((self.rows.blk > 0).sum())
        masked_frac = e_t * in_block / cfg.sequence_len     # sequence_len == 2*row_len
        device = self.trunk.cos.device
        clip = {k: v.unsqueeze(0).to(device) for k, v in stream_fields(self.rows).items()}
        clip["doc_id"] = torch.zeros_like(clip["pos"])
        density = 1.0 - two_stream_mask(clip, self.mask_builder or default_mask_builder(device)).sparsity() / 100.0

        dense = super().estimate_flops()                    # trunk matmul + attn + head
        return (dense - attn - head) + attn * density + head * masked_frac
