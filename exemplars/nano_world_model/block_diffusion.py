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

  * Hidden states are sliced to the masked positions BEFORE the head, so its work
    and memory scale with the number of masked positions rather than the full row.
  * The per-token CE is the head's own `loss_per_token`, and train_wm.py builds the
    head with head_ce="compiled" to reduce CE peak memory. The saving depends on
    shape and software stack; compilation does not guarantee that no [M, V]
    tensor is materialized. core/tests/unit/test_heads_per_token.py checks a fixed
    shape's peak memory and agreement with eager under nonuniform token weights.
    Liger's current fused linear CE backward does not support those weights, so
    the liger arm refuses this entry. Masking, weights and normalization stay here.
"""

import torch

from core.model.system import LMSystem


class BlockDiffusion:
    """The objective, bound to one row layout and one vocabulary.

    Args:
        rows: RowLayout — block spans and the two-stream attention mask
        vocab: VocabLayout — supplies classify_token_types for the type embeddings
        mask_id: global id of the [MASK] control token (the absorbing state)
        t_min, t_max: noise-level range
        device: where the (contract-only, batch-independent) BlockMask is built
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
        ce = system.head.loss_per_token(flat_h, idx[masked])              # [M]
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
                ce = system.head.loss_per_token(flat_h, rows[masked])
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

    def estimate_flops(self):
        """FLOPs/token, corrected — this objective breaks LMSystem's contract twice.

        The contract (core/model/system.py) is that every position counted in
        total_batch_size costs the same. Block diffusion computes on a DIFFERENT set
        of positions than it is charged for, in two independent ways:

          head       runs only where a token was masked. Charged for all 2*row_len
                     positions, it runs on E[t] * in-block of them — about 22% here.
          attention  runs under `train_block_mask`, not dense causal over
                     sequence_len, which is what the trunk's formula assumes.

        Left uncorrected the two together read 1.78x high, and MFU printed 101.6% —
        above the card's physical ceiling, which is the only reason anyone looked.
        Corrected it reads ~57%, and that gap is the point: 101.6% says the card is
        saturated and there is nothing to win, 57% says a third of it is idle.

        The head correction shrinks as the trunk grows; the attention one does NOT.
        At a 129-frame window the head is down to ~14% of the total while attention
        is ~71%, and the combined error is still ~1.6x. So this is not a small-model
        wart to outgrow.

        On the two denominators, because it is easy to get wrong (it was): the trunk's
        attention term counts a FULL sequence_len x sequence_len grid — the PaLM
        appendix-B / nanoGPT convention, which does not credit the causal half either.
        `BlockMask.sparsity()` also reports against the full grid, so the two match and
        `density` is the right multiplier. Measuring the mask against a CAUSAL grid
        instead gives 0.64 rather than 0.33 and understates the correction by ~2x.

        One consequence to state rather than hide: this MFU is now physically true,
        while the repo's dense-causal numbers carry the convention's overcount. At
        text's sequence lengths that overcount is ~3% of the total and nobody cares;
        here it would have been 3x on the attention term. So do not read this number
        against the text exemplar's and conclude anything — MFU was never comparable
        across architectures anyway (a different architecture trades arithmetic
        intensity for capability), and now it is not comparable across conventions
        either.

        Both factors are EXPECTATIONS, not per-step truths: t is redrawn every step,
        so the exact figure moves batch to batch. MFU is a display metric and an
        expectation is the right resolution for it — nothing downstream consumes this.
        """
        cfg = self.trunk.config
        head = 6 * sum(p.numel() for p in self.head.parameters())
        # Mirrors the attention term in GPT.estimate_flops (core/model/gpt.py). It is
        # spelled out again because it has to be SUBTRACTED before being rescaled, and
        # the trunk hands back only the sum. If that formula changes, this changes.
        q = cfg.n_embd // cfg.n_head
        attn = 12 * cfg.n_layer * cfg.n_head * q * cfg.sequence_len
        obj = self.objective

        e_t = 0.5 * (obj.t_min + obj.t_max)                 # t ~ U[t_min, t_max]
        in_block = int((obj.blk > 0).sum())
        masked_frac = e_t * in_block / cfg.sequence_len     # sequence_len == 2*row_len
        density = 1.0 - obj.block_mask.sparsity() / 100.0   # fraction of blocks kept

        dense = super().estimate_flops()                    # trunk matmul + attn + head
        return (dense - attn - head) + attn * density + head * masked_frac
