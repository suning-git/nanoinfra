"""
autoregressive.py — the OTHER objective on the same rows: plain next-token prediction.

The exemplar trains two world models over one data pipeline, one row layout and one
trunk. They differ in exactly one thing, the objective:

    block_diffusion.py   predict a masked block from the clean prefix, bidirectionally
                         within the block. One forward covers every block, so training
                         is cheap; generating a frame takes several denoising steps.
    autoregressive.py    predict token n+1 from tokens <= n, causally. Training is one
                         forward too, but generation is one forward PER TOKEN — and
                         that is exactly what a KV cache and CUDA graphs make fast,
                         which is why the real-time engine decodes this one.

Keeping both is the point. They are the two honest options for discrete video, they
trade against each other rather than one dominating, and having them share everything
except `loss()` is what makes the comparison mean anything.

WHAT IS SUPERVISED. A row is

    [bos, vstart, L0(given), a a a a, L1, a a a a, L2, ..., vend, eos, pad...]

and the model is asked to predict the FUTURE, not to reproduce its conditioning. So
the loss covers the code positions of predicted latent frames plus the closing tag,
and nothing else: not the given frame (it is the observation), not the action tokens
(they are the control input, and learning to predict which button a player presses is
a different problem that would spend capacity), not the padding.

That mask is a property of the ROW, so it is computed once at construction rather than
per batch, and it is expressed as IGNORE_INDEX targets so core's fused cross-entropy
applies unchanged.

Comparing the two objectives: this reports nll in nats per predicted token, and block
diffusion reports a NELBO in the same units. NELBO >= NLL for the same model class, so
a diffusion number below an AR number is a real win and a diffusion number above one is
inconclusive — the comparison is one-directional, and RESULTS.md states it that way.
"""

import torch

from core.model.system import LMSystem
from core.tokenization.vocab_layout import VocabLayout


class Autoregressive:
    """Next-token prediction over the predicted part of a world-model row."""

    def __init__(self, rows):
        self.rows = rows

        # Position p predicts the token at p+1, so p is supervised when p+1 is one of
        # the positions we want predicted. Built once: it depends only on the layout.
        predicted = torch.zeros(rows.row_len, dtype=torch.bool)
        for start, end in rows.spans:
            predicted[start:end] = True
        predicted[rows.end_pos] = True            # the closing tag ends generation
        self.predicted = predicted
        self.n_supervised = int(predicted.sum())

    def targets(self, idx):
        """idx [B, row_len] -> (inputs, targets) shifted by one, masked to the future."""
        inp, tgt = idx[:, :-1], idx[:, 1:]
        keep = self.predicted[1:].to(idx.device)
        return inp, torch.where(keep, tgt, torch.full_like(tgt, VocabLayout.IGNORE_INDEX))

    def loss(self, system, idx, token_types=None):
        inp, tgt = self.targets(idx)
        types = None if token_types is None else token_types[:, :-1]
        hidden = system._run_trunk(inp, token_types=types)
        return system.head.loss(hidden, tgt)

    @torch.no_grad()
    def val_nll(self, system, idx, batch=8):
        """Mean nll per predicted token on a fixed set of rows.

        Averaged over TOKENS rather than over batches: the last batch is usually short,
        and a mean of means would silently weight its tokens more heavily.
        """
        system.eval()
        total, n = 0.0, 0
        for i in range(0, len(idx), batch):
            rows = idx[i:i + batch]
            inp, tgt = self.targets(rows)
            hidden = system._run_trunk(inp)
            per_token = torch.nn.functional.cross_entropy(
                system.head(hidden).reshape(-1, system.head.lm_head.weight.size(0)).float(),
                tgt.reshape(-1), ignore_index=VocabLayout.IGNORE_INDEX, reduction="sum")
            total += per_token.item()
            n += int((tgt != VocabLayout.IGNORE_INDEX).sum())
        system.train()
        return total / max(n, 1)


class AutoregressiveSystem(LMSystem):
    """An LMSystem whose loss covers the predicted future only.

    Subclasses rather than wraps, so `state_dict()` keys stay `trunk.*` / `head.*` —
    a checkpoint trained here loads into the diffusion System over the same trunk, and
    into the inference engine, without a key remap.
    """

    def __init__(self, trunk, head, objective):
        super().__init__(trunk, head)
        self.objective = objective

    def loss(self, batch):
        return self.objective.loss(self, batch["idx"], batch.get("token_types"))
