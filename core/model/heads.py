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
produces a callable, not a type, so there is nothing to inject. Callers therefore reach
these entries through the instance (`head.loss_per_token(...)`), never the class.

FSDP handles `head(hidden)` through its standard forward hooks. The additional parameter-using
entry points (loss / loss_per_token) are registered as FSDP forward methods AFTER shard.

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
    head(hidden) uses the standard hooks; model_setup registers loss() and loss_per_token()
    as additional forward methods after sharding.
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
        weights and chooses the final reduction. Evaluation reads every metric from
        this entry (core/evaluation/eval_loss.py).

        The naive path materializes full logits. head_ce="compiled" compiles this
        method with dynamic N support and can reduce peak memory; the saving depends
        on tensor shapes and the PyTorch stack. Liger serves it only without gradient
        recording (LigerLMHead.loss_per_token).
        """
        logits = self.forward(hidden)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=VocabLayout.IGNORE_INDEX,
            reduction="none",
        )


class LigerLMHead(LMHead):
    """Liger fused linear CE computes loss in chunks to reduce peak memory.

    Selected by injection: `LigerLMHead.setup(head)` prepares `fused_ce`, `fused_ce_fp32_accum` (see loss())
    and `fused_ce_per_token` (see loss_per_token()) and swaps `__class__`. MUST run BEFORE `fully_shard(head)`.
    forward() keeps LMHead's logits path.
    """

    @classmethod
    def setup(cls, head):
        assert LigerFusedLinearCrossEntropyLoss is not None, "liger_kernel not installed"
        kwargs = dict(ignore_index=VocabLayout.IGNORE_INDEX, softcap=head.softcap)
        head.fused_ce = LigerFusedLinearCrossEntropyLoss(**kwargs, reduction="mean")
        head.fused_ce_fp32_accum = LigerFusedLinearCrossEntropyLoss(**kwargs, reduction="mean",
                                                                    accum_dtype=torch.float32)
        head.fused_ce_per_token = LigerFusedLinearCrossEntropyLoss(**kwargs, reduction="none")
        head.__class__ = cls  # inject: swap the method table; __dict__ (params, fused_ce*) preserved

    def _bf16_weight_for_fp32_storage(self):
        """fp32 storage under bf16 autocast (FSDP hands this a bf16 copy instead): Liger cuts the tokens into chunks
        and multiplies each by weight.t(), a view, which autocast's weight-cast cache skips, so left to autocast the
        weight is cast to bf16 once per chunk. Returns the same bf16 values cast once, or None when that is not the
        case (bf16 storage, no or fp16 autocast)."""
        weight = self.lm_head.weight
        if (weight.dtype == torch.float32 and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.bfloat16):
            return weight.to(torch.bfloat16)
        return None

    def loss(self, hidden, targets):
        if hidden.device.type == "cuda":
            weight, fused_ce = self.lm_head.weight, self.fused_ce
            cast = self._bf16_weight_for_fp32_storage()
            if cast is not None:
                # The buffer Liger sums the chunks' weight gradients into takes the weight's dtype, so with the
                # weight cast once to bf16 that sum is kept in fp32 (fused_ce_fp32_accum). The head's weight
                # gradient then comes out rounded to bf16, as every trunk matrix's does, and Liger scales it by the
                # upstream gradient in bf16 (exact when that is a power of two, e.g. 1/grad_accum_steps with a
                # power-of-two accumulation). Not under fp16 autocast, whose range would underflow that gradient.
                weight, fused_ce = cast, self.fused_ce_fp32_accum
            return fused_ce(weight, hidden.reshape(-1, hidden.size(-1)), targets.reshape(-1))
        return LMHead.loss(self, hidden, targets)  # cpu fallback

    def loss_per_token(self, hidden, targets):
        """Unreduced CE, only WITHOUT gradient recording (evaluation runs under torch.no_grad).

        With gradients it is refused: Liger computes the gradient inside the forward and its reduction='none'
        backward scales it by one upstream value, so nonuniform per-token weights would get a wrong gradient
        (right loss). Use head_ce="compiled" for a training objective that weights tokens. The condition is grad
        mode, not hidden.requires_grad: with the trunk frozen and the head trained, hidden needs no gradient but
        the head's weight does. Not decorated with @torch.no_grad(), which would turn a training call into a
        silently gradient-free result instead of an error.
        """
        if torch.is_grad_enabled():
            raise NotImplementedError(
                "LigerLMHead.loss_per_token is served only without gradient recording (torch.no_grad, as in "
                "evaluation): liger's reduction='none' backward drops the per-token upstream weights (loss right, "
                "gradient wrong); build with head_ce='compiled' for an unreduced CE in training")
        if hidden.device.type != "cuda":
            return LMHead.loss_per_token(self, hidden, targets)  # cpu fallback
        cast = self._bf16_weight_for_fp32_storage()
        weight = self.lm_head.weight if cast is None else cast
        # detach: a hidden made with requires_grad=True keeps the flag inside no_grad, and Liger computes the
        # gradients in its forward whenever the input requires one: [N, H] for the input and, since the weight
        # requires grad even under no_grad, [V, H] for the weight, with two extra matmuls per chunk.
        return self.fused_ce_per_token(weight, hidden.detach().reshape(-1, hidden.size(-1)), targets.reshape(-1))
