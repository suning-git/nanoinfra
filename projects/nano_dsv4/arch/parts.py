"""Shared building blocks for the frontier-architecture minis (MoE router,
fine-grained experts, interleaved RoPE, RMSNorm).

One part per class, minimal_gpt-level readability. Training-mode only:
no KV cache, no incremental-decode state (training-mode only).

Conventions follow core/model/gpt.py: functional rmsnorm, no-bias linears,
init via each trunk's init_weights (meta-init: EVERY param must be set there).
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    # Purely functional rmsnorm with no learnable params (same as core gpt.py)
    return F.rms_norm(x, (x.size(-1),))


class RMSNorm(nn.Module):
    """Learnable-weight RMSNorm (both source models use weighted norms on the
    low-rank residuals; the trunk-level norms stay functional like gpt.py)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


def precompute_interleaved_rope(seq_len, rope_dim, base, device=None):
    """cos/sin tables for interleaved RoPE: one theta per consecutive PAIR of
    channels (rope_dim//2 entries), returned already repeat_interleave(2)-expanded
    to [1, T, 1, rope_dim] for broadcasting over [B, T, H, D]-shaped slices."""
    channel = torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel / rope_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # [T, rope_dim/2]
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos[None, :, None, :], sin[None, :, None, :]  # fp32; cast at apply time


def rotate_half_interleaved(x):
    # Interleaved pairing: (x0,x1),(x2,x3),... -> (-x1,x0),(-x3,x2),...
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_interleaved_rope(x, cos, sin):
    """Rotate the TRAILING cos.shape[-1] channels of x (partial RoPE, [nope|rope]
    layout, matching V4's reference); leading channels pass through untouched.
    x: [B, T, H, D] (or broadcastable); cos/sin: [1, T', 1, rope_dim] fp32."""
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (rope.float() * cos + rotate_half_interleaved(rope).float() * sin).to(x.dtype)
    if nope.shape[-1] == 0:
        return rotated
    return torch.cat([nope, rotated], dim=-1)


class FineGrainedExperts(nn.Module):
    """Routed experts, loop-over-experts implementation (fine at mini scale;
    the real models use grouped-GEMM kernels — same math, different schedule).

    Expert MLP is SwiGLU: down(silu(gate(x)) * up(x)), with optional clamping
    of gate/up (V4's gpt-oss-style swiglu_limit; GLM passes None = no clamp).
    """

    def __init__(self, n_experts, dim, expert_dim, swiglu_limit=None):
        super().__init__()
        self.n_experts = n_experts
        self.limit = swiglu_limit
        # Packed weights: [E, 2*expert_dim, dim] (gate|up) and [E, dim, expert_dim]
        self.gate_up = nn.Parameter(torch.empty(n_experts, 2 * expert_dim, dim))
        self.down = nn.Parameter(torch.empty(n_experts, dim, expert_dim))

    def forward(self, flat_x, topk_idx, topk_weight):
        """flat_x [N, dim]; topk_idx/topk_weight [N, k] -> [N, dim]."""
        out = torch.zeros_like(flat_x)
        one_hot = F.one_hot(topk_idx, num_classes=self.n_experts)  # [N, k, E]
        for e in range(self.n_experts):
            slot, kpos = torch.where(one_hot[:, :, e])
            if slot.numel() == 0:
                continue
            gate, up = F.linear(flat_x[slot], self.gate_up[e]).chunk(2, dim=-1)
            if self.limit is not None:
                gate = gate.clamp(max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
            y = F.linear(F.silu(gate) * up, self.down[e])
            out.index_add_(0, slot, (y * topk_weight[slot, kpos, None]).to(out.dtype))
        return out


class SwiGLU(nn.Module):
    """Dense SwiGLU MLP (shared expert / GLM's first-k dense layers)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NoAuxRouter(nn.Module):
    """Top-k router with aux-loss-free load balancing (DeepSeek-V3 lineage,
    `topk_method: noaux_tc`, used verbatim by GLM-5.2 and V4's learned layers).

    - scores = score_fn(x @ W) computed in fp32 (`moe_router_dtype: float32`).
    - SELECTION uses scores + e_score_correction_bias; the mixing WEIGHTS use
      the raw scores (bias steers who gets picked, never how much they weigh).
    - The bias is updated online from expert load: overloaded experts get
      bias -= u, underloaded get bias += u (V3 paper's b_i += u*sign(err)).
      The reference updates ONCE PER OPTIMIZER STEP, so forward() only
      ACCUMULATES the load; the update itself lives in `step_routers()`, which
      the training loop calls once per optimizer step. See that function for
      why the update cannot live in forward().

      What that cadence costs, measured (32 experts, top-4, sigmoid, u=1e-3,
      grad_accum=16, 200 steps), against the earlier variant that updated once
      per micro-batch (i.e. grad_accum times too fast). The bias does NOT grow
      without bound — it is a closed loop and settles at a magnitude set by how
      much steering the balance needs:

          per micro-batch (16x too fast):  |bias|max 0.039  load imbalance 0.052
          per optimizer step (reference):  |bias|max 0.015  load imbalance 0.19

      The number that matters is the bias measured against the SPREAD of the
      content scores it competes with (per-token std across experts ~0.040 at
      init): the old cadence puts the bias at ~1.0x that spread, the reference
      cadence at ~0.4x. So the fast controller does not "swamp" the scores by
      orders of magnitude; it becomes comparable to them, which is enough to
      let load, not content, pick the experts early in training.

      Note this is a TRADE-OFF, not a free fix: the reference cadence balances
      load visibly worse (0.19 vs 0.052). It is done because it is what the
      reference does — a faithfulness fix, nothing more.

      Do NOT read a quality claim into it. The wide validation hump seen in the
      nano-glm52 reproduction was chased to this controller and then shown, on
      2026-07-28, to be a MEASUREMENT ARTIFACT of a train/eval sequence-format
      mismatch, not a model pathology: three arms differing only in controller
      cadence (16x / 1x / frozen) reached identical training loss (3.946 /
      3.948 / 3.941) while the mismatched validation metric read 4.79 / 14.12 /
      4.24. The one arm measured with a format-matched validation stream scored
      3.948 — equal to its training loss, with no hump at all. The controller
      changes how badly a broken ruler misreads the model; it does not change
      the model.
    - score_fn: 'sigmoid' (GLM-5.2 / V3) or 'sqrtsoftplus' (V4).
    """

    def __init__(self, n_experts, dim, top_k, score_fn, routed_scaling,
                 bias_update_speed=1e-3):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.score_fn = score_fn
        self.routed_scaling = routed_scaling
        self.bias_update_speed = bias_update_speed
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        self.register_buffer("e_score_correction_bias", torch.zeros(n_experts))
        self.register_buffer("_load_accum", torch.zeros(n_experts), persistent=False)  # transient: never in state_dict

    def _scores(self, logits):
        if self.score_fn == "sigmoid":
            return logits.sigmoid()
        if self.score_fn == "sqrtsoftplus":
            return F.softplus(logits).sqrt()
        raise ValueError(self.score_fn)

    def forward(self, flat_x):
        # build_system casts the whole trunk to bf16 AFTER init; promote the
        # bias buffer back to fp32 lazily so 1e-3 updates don't quantize away
        if self.e_score_correction_bias.dtype != torch.float32:
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.float()
        logits = F.linear(flat_x.float(), self.weight.float())
        scores = self._scores(logits)
        topk_idx = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        topk_weight = scores.gather(1, topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        if self.training:
            with torch.no_grad():
                # ONLY accumulate here. forward() sees one micro-batch on one rank;
                # the controller is defined over the whole optimizer step on the whole
                # global batch. step_routers() is where those shards are summed.
                if self._load_accum.dtype != torch.float32:
                    self._load_accum.data = self._load_accum.data.float()
                self._load_accum += torch.bincount(
                    topk_idx.flatten(), minlength=self.n_experts).float()
                # NOTHING ELSE GOES HERE. In particular, no Python int counter:
                # torch.compile treats an nn.Module's integer attributes as STATIC and
                # guards on their value, so `self._micro += 1` recompiles the graph on
                # every forward, blows past cache_size_limit, and silently drops the
                # whole frame back to eager. Dynamo names it outright:
                # "___stack1._micro == 0 ... torch.compile considers integer attributes
                # of the nn.Module to be static" (13 recompiles in 8 forwards, plus a
                # graph break). This cost the fast arm most of its throughput and NONE
                # of the correctness gates caught it — they all ran uncompiled.
                # Accumulate into the tensor buffer only; anything that needs a count
                # reads it from there (routers_pending_load).
        return topk_idx, (topk_weight * self.routed_scaling).to(flat_x.dtype)


def step_routers(model) -> int:
    """Apply ONE load-balancing update to every NoAuxRouter. Call once per
    OPTIMIZER STEP (not per micro-batch), after the last micro-backward.

    Why this is not inside forward(). The controller's input is the expert load
    over the whole optimizer step's global batch, but forward() only ever sees
    ONE SHARD of that batch. The global batch is split two ways at once:

        global batch  =  grad_accum micro-batches  x  world_size ranks

    forward() is blind to both. Updating inside it therefore gets the cadence
    wrong (it fires grad_accum times too often) AND, under data parallelism, the
    load wrong (each rank sees only its own shard). The second one is the nastier
    of the two: it is a BUFFER, so DDP's gradient all-reduce does not touch it,
    and the replicas' routers silently drift apart while training looks healthy.
    Measured on 2 GPUs before this was moved here: all 9 routers had diverged by
    the end of step 0, while every parameter stayed bit-identical.

    Summing both shards is the same one reduction, so both defects close here.
    Returns the number of routers stepped, so callers can log it.
    """
    routers = [m for m in model.modules() if isinstance(m, NoAuxRouter)]
    if not routers:
        return 0
    with torch.no_grad():
        for m in routers:
            if m._load_accum.dtype != torch.float32:
                m._load_accum.data = m._load_accum.data.float()
            if m.e_score_correction_bias.dtype != torch.float32:
                m.e_score_correction_bias.data = m.e_score_correction_bias.data.float()
        if dist.is_available() and dist.is_initialized():
            # One collective for all routers, not one each. It runs outside forward
            # and outside the backward's comm stream, so it cannot race the gradient
            # all-reduces; and every rank reaches it in the same place, so the
            # collective ordering NCCL requires is satisfied by construction.
            flat = torch.cat([m._load_accum for m in routers])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            off = 0
            for m in routers:
                n = m._load_accum.numel()
                m._load_accum.copy_(flat[off:off + n])
                off += n
        for m in routers:
            err = m._load_accum.mean() - m._load_accum   # positive = underloaded
            m.e_score_correction_bias += m.bias_update_speed * err.sign()
            m._load_accum.zero_()
    return len(routers)


def routers_pending_load(model) -> float:
    """Total expert-load still sitting in the routers, un-applied.

    The safety net for "the training loop forgot to call step_routers()". It is a
    FUNCTION rather than a counter inside forward() on purpose: a Python int
    attribute mutated in forward is guarded by torch.compile and recompiles the
    graph every step (see the note in NoAuxRouter.forward). Call it from a test or
    once at the end of a run; zero cost on the hot path.

    Returns 0.0 right after a step_routers() call, and grows with every micro-batch
    that is never stepped.
    """
    return float(sum(m._load_accum.sum().item()
                     for m in model.modules() if isinstance(m, NoAuxRouter)))


class HashRouter(nn.Module):
    """V4's hash routing for the first `num_hash_layers` MoE layers: WHICH
    experts is a frozen token-id -> expert-ids table (`tid2eid`); the learned
    gate still produces the mixing weights for those experts. The real table
    ships in the checkpoint; we freeze a random balanced assignment (same
    spirit: static, uniform load by construction)."""

    def __init__(self, n_experts, dim, top_k, score_fn, routed_scaling, vocab_size):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.score_fn = score_fn
        self.routed_scaling = routed_scaling
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        self.register_buffer("tid2eid", torch.zeros(vocab_size, top_k, dtype=torch.long))

    def init_table(self, generator=None):
        # Balanced random assignment: each token id draws top_k distinct experts.
        v = self.tid2eid.shape[0]
        table = torch.argsort(torch.rand(v, self.n_experts, generator=generator), dim=-1)[:, : self.top_k]
        self.tid2eid.copy_(table.to(self.tid2eid.device))

    def _scores(self, logits):
        if self.score_fn == "sigmoid":
            return logits.sigmoid()
        return F.softplus(logits).sqrt()

    def forward(self, flat_x, flat_ids):
        logits = F.linear(flat_x.float(), self.weight.float())
        scores = self._scores(logits)
        topk_idx = self.tid2eid[flat_ids]  # [N, k] frozen selection
        topk_weight = scores.gather(1, topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, (topk_weight * self.routed_scaling).to(flat_x.dtype)


class MoEBlock(nn.Module):
    """Fine-grained MoE block: routed experts (top-k) + one always-on shared
    expert. `router_kind`: 'noaux' or 'hash' (V4 first layers)."""

    def __init__(self, dim, n_experts, expert_dim, top_k, score_fn, routed_scaling,
                 n_shared, router_kind="noaux", vocab_size=None, swiglu_limit=None):
        super().__init__()
        if router_kind == "hash":
            self.gate = HashRouter(n_experts, dim, top_k, score_fn, routed_scaling, vocab_size)
        else:
            self.gate = NoAuxRouter(n_experts, dim, top_k, score_fn, routed_scaling)
        self.router_kind = router_kind
        self.experts = FineGrainedExperts(n_experts, dim, expert_dim, swiglu_limit)
        self.shared_experts = SwiGLU(dim, n_shared * expert_dim)

    def forward(self, x, flat_ids=None):
        B, T, D = x.shape
        flat = x.view(-1, D)
        if self.router_kind == "hash":
            topk_idx, topk_weight = self.gate(flat, flat_ids)
        else:
            topk_idx, topk_weight = self.gate(flat)
        routed = self.experts(flat, topk_idx, topk_weight).view(B, T, D)
        return routed + self.shared_experts(x)


def init_linear_(weight):
    """gpt.py's linear init: normal, std = 1/sqrt(fan_in) * min(1, sqrt(fan_out/fan_in))."""
    import math
    fan_out, fan_in = weight.shape[0], weight.shape[1]
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    torch.nn.init.normal_(weight, mean=0.0, std=std)
