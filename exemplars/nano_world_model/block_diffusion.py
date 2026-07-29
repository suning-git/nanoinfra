"""
block_diffusion.py — the objective: absorbing-state (masked) diffusion over blocks.

Each latent frame is one diffusion BLOCK. The forward process masks each token of a
block independently with probability t, where t is drawn per block from U[t_min,
t_max]; the model sees the clean prefix plus the partially masked block and predicts
the CLEAN token AT each masked position. Same trunk as an AR model, different head
convention: predict-here, not predict-next.

Under MDLM's linear schedule the per-token NELBO is E_t[ mean CE over masked ], which
makes the training loss

    (1/t) * sum(CE over masked) / (rows * predicted tokens per row)

an unbiased NELBO estimate. The 1/t factor is MDLM's per-token weight; t is clipped
away from 0 because the unclipped schedule has enormous gradient variance (BD3-LM).

The validation number is the same quantity on a fixed t grid with fixed mask RNG, so
it reads in nats per token and is ONE-DIRECTIONALLY comparable to an AR nll: NELBO >=
NLL, so a lower BD number definitively beats AR, while a higher one is inconclusive.

Two implementation notes that are load-bearing rather than stylistic:

  * Hidden states are sliced to the masked positions BEFORE the head. Full-length
    logits are ~1GB per row at vocab 96786.
  * The head + cross-entropy run under torch.compile (`_head_ce`). Inductor tiles the
    vocab dimension so the [M, V] logits are never materialized: peak memory drops
    ~7x (26.7GB -> 4.0GB at M=16896), with gradients bitwise identical to eager. This
    is what made the long-window model trainable at all. Liger's fused CE is NOT a
    substitute: its reduction="none" backward drops the per-token upstream weight, so
    the loss looks right while the gradient is silently wrong (measured rel-err 0.82).
"""

import torch
import torch.nn.functional as F

from core.model.system import LMSystem


@torch.compile(dynamic=True)
def _head_ce(weight, hidden, tgt, softcap):
    """Per-token CE [M] at masked positions — LMHead.forward + reduction='none' CE,
    fused. Reproduces the head exactly (linear -> fp32 -> softcap tanh).

    Takes the head's WEIGHT rather than the head module because it assumes a full
    tensor. That holds for single-GPU and for replicated data parallel (NanoDDP,
    the measured-best setup here). Under FSDP the weight is a sharded DTensor and
    this would see one shard; that path would need the collective inside a
    registered head method. FSDP lost on both throughput and memory at this scale
    (2026-07-28 benchmarks), so it is deliberately unsupported.
    """
    logits = F.linear(hidden, weight).float()
    logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, tgt, reduction="none")


class BlockDiffusion:
    """The objective, bound to one row layout and one vocabulary.

    Args:
        rows: RowLayout — block spans and the two-stream attention mask
        vocab: VocabLayout — supplies classify_token_types for the type embeddings
        mask_id: global id of the [MASK] control token (the absorbing state)
        t_min, t_max: noise-level range
        device: where the (geometry-only, batch-independent) BlockMask is built
    """

    def __init__(self, rows, vocab, mask_id, t_min, t_max, device):
        self.rows = rows
        self.vocab = vocab
        self.mask_id = int(mask_id)
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.block_mask = rows.train_block_mask(device)
        self.blk = rows.blk.to(device)
        self.n_predicted = rows.n_blocks * rows.cpf   # supervised tokens per row

    def noise(self, idx, generator):
        """idx [B, n] clean rows -> (two [B, 2n], masked [B, n], weight [B, n]).

        One t per block per row; then an independent Bernoulli(t) draw per token,
        restricted to predicted blocks (conditioning positions are never masked)."""
        B, n = idx.shape
        t_blk = torch.rand(B, self.rows.n_blocks + 1, generator=generator, device=idx.device)
        t_blk = self.t_min + (self.t_max - self.t_min) * t_blk    # column 0 unused (blk==0)
        t_pos = t_blk.gather(1, self.blk.unsqueeze(0).expand(B, -1))
        in_block = (self.blk > 0).unsqueeze(0)
        masked = (torch.rand(B, n, generator=generator, device=idx.device) < t_pos) & in_block
        noisy = idx.masked_fill(masked, self.mask_id)
        weight = torch.where(masked, 1.0 / t_pos, torch.zeros_like(t_pos))
        return torch.cat([idx, noisy], dim=1), masked, weight

    def _hidden_at_masked(self, system, two, masked):
        """Run the trunk on [clean|noisy] and return the noisy-stream hidden states
        at masked positions only, [M, H]."""
        n = self.rows.row_len
        token_types = self.vocab.classify_token_types(two)
        h = system.trunk(two, token_types=token_types, block_mask=self.block_mask)
        return h[:, n:, :][masked]

    def loss(self, system, idx, noise_seed):
        """Scalar training loss (an unbiased NELBO estimate, nats per predicted token).

        The noise is seeded from the micro-batch's rows (see VideoRowLoader), so the
        masks are a function of the DATA, not of the step index or the rank."""
        generator = torch.Generator(device=idx.device).manual_seed(int(noise_seed))
        two, masked, weight = self.noise(idx, generator)
        flat_h = self._hidden_at_masked(system, two, masked)
        ce = _head_ce(system.head.lm_head.weight, flat_h, idx[masked], system.head.softcap)
        return (ce * weight[masked]).sum() / (idx.shape[0] * self.n_predicted)

    @torch.no_grad()
    def val_elbo(self, system, idx, t_grid, seed=0, batch=4):
        """Deterministic NELBO on held-out rows: masked CE averaged over a fixed t
        grid with fixed mask RNG (linear schedule => grid average ~ per-token NELBO).

        Returns (mean, {t: value}). Same rows, same masks, every time — so the number
        is comparable across runs and across checkpoints of one run."""
        device = idx.device
        in_block = (self.blk > 0).unsqueeze(0)
        per_t = {}
        for t in t_grid:
            gen = torch.Generator(device=device).manual_seed(seed + int(t * 1000))
            total, count = 0.0, 0
            for i in range(0, len(idx), batch):
                rows = idx[i:i + batch]
                B, n = rows.shape
                masked = (torch.rand(B, n, generator=gen, device=device) < t) & in_block
                two = torch.cat([rows, rows.masked_fill(masked, self.mask_id)], dim=1)
                flat_h = self._hidden_at_masked(system, two, masked)
                ce = _head_ce(system.head.lm_head.weight, flat_h, rows[masked],
                              system.head.softcap)
                total += ce.sum().item()
                count += int(masked.sum())
            per_t[t] = total / max(count, 1)
        return sum(per_t.values()) / len(per_t), per_t


class BlockDiffusionSystem(LMSystem):
    """An LMSystem whose `loss(batch)` is the block-diffusion NELBO, not next-token CE.

    This is how core says to train a different objective: "Projects that want a
    different composition (e.g. a diffusion head, or multiple heads) write their own
    System satisfying the same `loss(batch)` contract — core does not change"
    (core/model/system.py). So core's Trainer runs this unmodified; there is no
    training-loop subclass anywhere in this project.

    It SUBCLASSES LMSystem rather than wrapping one, which matters: the same two
    submodules are registered, so `state_dict()` keys stay `trunk.*` / `head.*` and a
    checkpoint written here loads into any other System over the same trunk. The
    objective is a plain object, not an nn.Module, so it does not enter state_dict.
    """

    def __init__(self, trunk, head, objective):
        super().__init__(trunk, head)
        self.objective = objective

    def loss(self, batch):
        # noise_seed comes from the rows in this micro-batch (see dataset.py), so the
        # masks are a function of the data rather than of the step index or the rank.
        return self.objective.loss(self, batch["idx"], batch["noise_seed"])
