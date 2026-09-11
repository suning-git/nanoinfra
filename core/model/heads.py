"""
Pluggable output heads.

A head owns the un-embedding parameters and the loss/logits computation, as a
SEPARATE nn.Module from the trunk (the GPT body). This buys three things:
  - the orchestrator assembles the head (it is not baked into the model);
  - the head gets its own FSDP shard group — params are lazily all-gathered and
    every head call is a legitimate FSDP forward window (no more window-outside
    calls, which the old GPT.head_* methods relied on FSDP-root-no-reshard to
    survive);
  - torch.compile treats trunk and head as independent frames.

Behavior families (naive CE / Liger fused CE / future band-factorized) are chosen by
`__class__` INJECTION at assembly time (`XXXHead.setup(head)`), BEFORE `fully_shard`,
once. Runtime never changes `__class__`.

One family is NOT a class: head_ce="compiled" keeps the naive class and binds
compiled `loss` and `loss_per_token` as INSTANCE attributes instead — `torch.compile`
produces a callable, not a type, so there is nothing to inject. That asymmetry is why
`type_losses` below reaches for `type(self).loss` rather than `self.loss`.

FSDP handles `head(hidden)` through its standard forward hooks. The additional parameter-using
entry points (loss / loss_per_token / type_losses) are registered as FSDP forward
methods AFTER shard.

FSDP2 timing rule (why the order matters): the `__class__` injection MUST happen
BEFORE `fully_shard` — a post-shard `__class__` swap silently drops FSDP's dynamically
mixed-in class — and `register_fsdp_forward_method` MUST happen AFTER, since it no-ops
on a non-FSDPModule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.tokenization.vocab_layout import VocabLayout

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except Exception:                       # noqa: BLE001 — deliberately broad, see below
    # NOT just ImportError. `liger_kernel.transformers` runs @triton.autotune at
    # import time, which wants a live Triton driver; on a machine with none — a
    # cluster login node, say — it raises `RuntimeError: 0 active drivers`, which
    # sails straight through an ImportError guard and takes `import core.model.heads`
    # down with it. Reported from a CentOS 7 / A100 deployment, 2026-08-26.
    #
    # A broad except at an optional-import site is only safe because of what the
    # flag now means downstream: head_ce="liger" RAISES when this is False, it does
    # not quietly become another arm. Before that was true, swallowing everything
    # here would have meant a login-node quirk silently changing which kernel a
    # training run used.
    LigerFusedLinearCrossEntropyLoss = None

LIGER_AVAILABLE = LigerFusedLinearCrossEntropyLoss is not None


class LMHead(nn.Module):
    """Linear un-embedding + softcap; naive (F.cross_entropy) loss path.

    Tensor-level building block: methods take (hidden, targets), NOT a batch dict.
    Batch unpacking lives one level up in LMSystem. The head has its own FSDP root:
    head(hidden) uses the standard hooks; model_setup registers loss(), loss_per_token()
    and type_losses() as additional forward methods after sharding.
    """

    def __init__(self, n_embd, vocab_size, softcap=15.0):
        super().__init__()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.softcap = softcap

    def init_weights(self):
        # matches the old GPT.init_weights: the classifier starts at zero.
        torch.nn.init.zeros_(self.lm_head.weight)

    def forward(self, hidden):
        """hidden [B,T,H] -> softcapped logits [B,T,V] (fp32). Eval / inference entry."""
        logits = self.lm_head(hidden).float()
        return self.softcap * torch.tanh(logits / self.softcap)

    def loss(self, hidden, targets):
        """CE loss (scalar). Training entry. Naive path materializes logits."""
        logits = self.forward(hidden)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=VocabLayout.IGNORE_INDEX,
            reduction="mean",
        )

    def loss_per_token(self, hidden, targets):
        """Unreduced CE: hidden [..., H], targets [...] -> flattened [N].

        IGNORE_INDEX positions return zero. The caller selects tokens, applies
        weights and chooses the final reduction.

        The naive path materializes full logits. head_ce="compiled" compiles this
        method with dynamic N support and can reduce peak memory; the saving depends
        on tensor shapes and the PyTorch stack. Liger currently rejects this entry.
        """
        logits = self.forward(hidden)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=VocabLayout.IGNORE_INDEX,
            reduction="none",
        )

    def type_losses(self, hidden, targets, target_types, type_ids):
        """Per-type CE losses {type_id: scalar} via masked targets (eval).

        Calls `type(self).loss(self, ...)`, not `self.loss(...)` and not
        `LMHead.loss(self, ...)`. Both details are load-bearing:

          - `self.loss` would pick up whatever is bound as an INSTANCE attribute.
            Two things bind one: head_ce="compiled" (a dynamo wrapper) and, under
            FSDP, register_fsdp_forward_method. Reaching either from in here nests a
            frame or a forward window inside the one type_losses is already running
            in — untested, and this is an eval path not worth the risk.
          - `LMHead.loss` would hard-bind the naive implementation, silently
            downgrading a LigerLMHead's per-type eval to the unfused path (and
            materializing [B,T,V] logits once per type id, which is the exact cost
            liger exists to remove).

        `type(self)` threads that needle: it skips the instance dict but still
        resolves through the MRO, so a liger head gets liger — including after
        fully_shard rewrites __class__ to an FSDP-mixed subclass.
        """
        result = {}
        for tid in type_ids:
            masked = torch.where(
                target_types == tid, targets,
                torch.full_like(targets, VocabLayout.IGNORE_INDEX),
            )
            result[tid] = type(self).loss(self, hidden, masked)
        return result


class LigerLMHead(LMHead):
    """Liger fused linear CE computes loss in chunks to reduce peak memory.

    Selected by injection: `LigerLMHead.setup(head)` prepares `fused_ce` and swaps
    `__class__`. MUST run BEFORE `fully_shard(head)`. forward() keeps LMHead's logits
    path; the inherited type_losses() dispatches to this class's loss() per type.
    loss_per_token() is refused (below).
    """

    @classmethod
    def setup(cls, head):
        assert LigerFusedLinearCrossEntropyLoss is not None, "liger_kernel not installed"
        head.fused_ce = LigerFusedLinearCrossEntropyLoss(
            ignore_index=VocabLayout.IGNORE_INDEX, reduction="mean", softcap=head.softcap,
        )
        head.__class__ = cls  # inject: swap the method table; __dict__ (params, fused_ce) preserved

    def loss(self, hidden, targets):
        if hidden.device.type == "cuda":
            return self.fused_ce(
                self.lm_head.weight,
                hidden.reshape(-1, hidden.size(-1)),
                targets.reshape(-1),
            )
        return LMHead.loss(self, hidden, targets)  # cpu fallback

    def loss_per_token(self, hidden, targets):
        """Reject unreduced CE: the current fused backward cannot honor nonuniform
        per-token upstream gradients. A silent eager fallback would change the
        selected implementation and its memory cost. Use head_ce="compiled" for
        objectives that need this entry.
        """
        raise NotImplementedError(
            "LigerLMHead.loss_per_token: liger's reduction='none' backward drops the "
            "per-token upstream weights (loss right, gradient wrong); build with "
            "head_ce='compiled' for an unreduced CE")
