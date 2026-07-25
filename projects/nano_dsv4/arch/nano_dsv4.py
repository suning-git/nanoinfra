"""DeepSeek-V4 mini — a ~100M-active faithful miniature of
deepseek-ai/DeepSeek-V4-Flash (transformers `deepseek_v4`), on the nanoinfra
trunk seam.

What is kept from the original (see docs/deepseek_v4.md for the delta list):
  - Shared-KV MQA: ONE kv head, K and V are the same tensor; low-rank Q with
    unweighted per-head RMSNorm; partial interleaved RoPE on the trailing
    slice; per-head learnable attention sink; INVERSE rope on the attention
    output (V carries rope because K=V — the conjugate rotation at the query
    position removes the absolute phase, leaving relative-only dependence);
    grouped low-rank output projection (o_a block-diagonal -> o_b mix).
  - Three layer types via compress_ratios: 0 = sliding-window only;
    4 = CSA (compress every 4 tokens with the overlapping Ca/Cb two-series
    scheme, lightning indexer picks top-k compressed entries per query);
    hca_rate = HCA (compress every hca_rate tokens, non-overlapping, no
    indexer — every entry attended). Every layer keeps the sliding branch;
    long-range entries are CONCATENATED onto the KV axis.
  - mHC: hc_mult parallel residual streams; per-site dynamic pre/post/comb
    weights, comb Sinkhorn-projected onto doubly-stochastic matrices; final
    HyperHead collapse.
  - MoE every layer: 32 routed experts top-4 + 1 shared, sqrtsoftplus scoring,
    noaux_tc bias balancing, swiglu clamp (limit 10); the first
    num_hash_layers layers hash-route (frozen token-id -> expert table).
  - CSA indexer trains via a KL auxiliary against the real attention
    distribution over compressed entries (V3.2-recipe; selection itself is
    non-differentiable). Driver adds trunk.pop_aux_loss() to the LM loss.

Training-mode only (stateless compressors, whole sequence per forward).
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import (MoEBlock, RMSNorm, apply_interleaved_rope, init_linear_, norm,
                    precompute_interleaved_rope, rotate_half_interleaved)


@dataclass
class NanoDSV4Config:
    sequence_len: int = 512
    vocab_size: int = 65536
    n_layer: int = 12
    n_embd: int = 768
    n_token_types: int = 3
    # attention (original Flash: 64 heads x 512, 1 kv head, q_lora 1024,
    # o_groups 8 x o_lora 1024, rope 64, sliding 128, sinks)
    # Distinct-values discipline: unrelated quantities must not share a number,
    # so window 96 != head_dim 128, rope 40, topk 48, o_lora 320 != group_in 384,
    # expert_dim 448 != seq 512 (two residual coincidences remain: head_dim 128 =
    # CSA entry count 512/4, and hca_rate 32 = n_routed_experts 32).
    n_head: int = 12
    head_dim: int = 128
    q_lora_rank: int = 192
    o_groups: int = 4
    o_lora_rank: int = 320
    rope_dim: int = 40
    sliding_window: int = 96
    rope_base_main: float = 10000.0
    rope_base_compress: float = 160000.0
    # layer types: 0 = sliding-only, 4 = CSA, hca_rate = HCA
    # (Flash pattern: [0, 0, 4, 128, 4, 128, ...]; mini uses hca_rate=32 so
    #  seq 512 still yields 16 HCA entries)
    csa_rate: int = 4
    hca_rate: int = 32
    compress_ratios: tuple = (0, 0, 4, 32, 4, 32, 4, 32, 4, 32, 4, 32)
    index_topk: int = 48          # of seq/csa_rate = 128 compressed entries
    index_n_heads: int = 4
    index_head_dim: int = 64
    # mHC (original: hc_mult 4, sinkhorn 20 iters)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # MoE (original Flash: 256 experts top-6 + 1 shared, sqrtsoftplus,
    # swiglu_limit 10, first 3 layers hash-routed, every layer MoE)
    n_routed_experts: int = 32
    num_experts_per_tok: int = 4
    moe_expert_dim: int = 448
    n_shared_experts: int = 1
    routed_scaling: float = 1.5
    swiglu_limit: float = 10.0
    num_hash_layers: int = 3
    aux_loss_weight: float = 0.01


def gated_compress(kv, gate, rate, overlap):
    """Softmax-gated window pooling shared by HCA / CSA / the CSA indexer.

    kv, gate: [B, n_win, rate, D] (already window-shaped, position_bias added
    to gate by the caller). `overlap=False` (HCA): one softmax over the rate
    slots per window. `overlap=True` (CSA): kv/gate carry 2D channels = two
    series Ca|Cb; entry w pools window w-1's Ca slice with window w's Cb slice
    (width 2*rate, stride rate); window 0's Ca half is zero-kv / -inf-gate.
    Returns [B, n_win, D]."""
    B, W, R, D2 = kv.shape
    if not overlap:
        return (kv * gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)).sum(dim=2)
    D = D2 // 2
    new_kv = kv.new_zeros(B, W, 2 * R, D)
    new_gate = gate.new_full((B, W, 2 * R, D), float("-inf"))
    new_kv[:, :, R:] = kv[..., D:]        # Cb: current window
    new_gate[:, :, R:] = gate[..., D:]
    if W > 1:
        new_kv[:, 1:, :R] = kv[:, :-1, :, :D]   # Ca: previous window
        new_gate[:, 1:, :R] = gate[:, :-1, :, :D]
    return (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)


class V4Compressor(nn.Module):
    """Long-range channel: compress every `rate` source tokens into one
    head_dim entry (HCA: non-overlapping; CSA: overlapping Ca/Cb series).
    Entries get RoPE at their window-start position (compress rope family)."""

    def __init__(self, config, rate, overlap, out_dim=None):
        super().__init__()
        self.rate = rate
        self.overlap = overlap
        self.dim = out_dim or config.head_dim
        width = 2 * self.dim if overlap else self.dim
        self.kv_proj = nn.Linear(config.n_embd, width, bias=False)
        self.gate_proj = nn.Linear(config.n_embd, width, bias=False)
        self.position_bias = nn.Parameter(torch.empty(rate, width))
        self.kv_norm = RMSNorm(self.dim)

    def forward(self, x, cos_sin_c):
        B, T, _ = x.shape
        W = T // self.rate
        usable = W * self.rate
        kv = self.kv_proj(x[:, :usable]).view(B, W, self.rate, -1)
        gate = self.gate_proj(x[:, :usable]).view(B, W, self.rate, -1) + self.position_bias
        entries = self.kv_norm(gated_compress(kv, gate, self.rate, self.overlap))
        cos, sin = cos_sin_c
        win_pos = torch.arange(W, device=x.device) * self.rate
        entries = apply_interleaved_rope(entries.unsqueeze(2), cos[:, win_pos], sin[:, win_pos]).squeeze(2)
        return entries  # [B, W, dim]


class V4Indexer(nn.Module):
    """CSA's lightning indexer: its own scaled-down compressor (index_head_dim)
    over the same windows, then score(q_t, entry_w) = sum_h w_h ReLU(q.k)."""

    def __init__(self, config):
        super().__init__()
        c = config
        self.compressor = V4Compressor(c, c.csa_rate, overlap=True, out_dim=c.index_head_dim)
        self.q_b_proj = nn.Linear(c.q_lora_rank, c.index_n_heads * c.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(c.n_embd, c.index_n_heads, bias=False)
        self.n_heads = c.index_n_heads
        self.head_dim = c.index_head_dim
        self.softmax_scale = c.index_head_dim ** -0.5

    def forward(self, x, q_resid, cos_sin_c):
        B, T, _ = x.shape
        entries = self.compressor(x, cos_sin_c)  # [B, W, D]
        cos, sin = cos_sin_c
        q = self.q_b_proj(q_resid).view(B, T, self.n_heads, self.head_dim)
        q = apply_interleaved_rope(q, cos[:, :T], sin[:, :T])
        scores = torch.einsum("bthd,bwd->bthw", q.float(), entries.float()) * self.softmax_scale
        scores = F.relu(scores)
        w = self.weights_proj(x).float() * (self.n_heads ** -0.5)
        return torch.einsum("bth,bthw->btw", w, scores)  # [B, T, W] raw scores


class V4Attention(nn.Module):
    """Sliding-window shared-KV MQA + optional long-range compressed channel.
    Returns (out, aux_kl or None)."""

    def __init__(self, config, layer_idx):
        super().__init__()
        c = config
        self.layer_type = c.compress_ratios[layer_idx]  # 0 / csa_rate / hca_rate
        self.n_head = c.n_head
        self.head_dim = c.head_dim
        self.rope_dim = c.rope_dim
        self.sliding = c.sliding_window
        self.topk = c.index_topk
        self.scaling = c.head_dim ** -0.5

        self.q_a_proj = nn.Linear(c.n_embd, c.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(c.q_lora_rank)
        self.q_b_proj = nn.Linear(c.q_lora_rank, c.n_head * c.head_dim, bias=False)
        self.kv_proj = nn.Linear(c.n_embd, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim)
        # grouped low-rank output projection: block-diagonal o_a, then o_b mix
        assert (c.n_head * c.head_dim) % c.o_groups == 0
        self.group_in = c.n_head * c.head_dim // c.o_groups
        self.o_a_proj = nn.Parameter(torch.empty(c.o_groups, self.group_in, c.o_lora_rank))
        self.o_b_proj = nn.Linear(c.o_groups * c.o_lora_rank, c.n_embd, bias=False)
        self.sinks = nn.Parameter(torch.empty(c.n_head))

        if self.layer_type == c.csa_rate and self.layer_type != 0:
            self.compressor = V4Compressor(c, c.csa_rate, overlap=True)
            self.indexer = V4Indexer(c)
        elif self.layer_type != 0:
            self.compressor = V4Compressor(c, c.hca_rate, overlap=False)
            self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

    def forward(self, x, q_resid, cos_sin_main, cos_sin_c, sliding_mask):
        B, T, _ = x.shape
        cos_sin = cos_sin_main if self.compressor is None else cos_sin_c
        cos, sin = cos_sin

        q = self.q_b_proj(q_resid).view(B, T, self.n_head, self.head_dim)
        q = norm(q)  # unweighted per-head RMSNorm (q_b_norm)
        q = apply_interleaved_rope(q, cos[:, :T], sin[:, :T])
        kv = self.kv_norm(self.kv_proj(x)).unsqueeze(2)  # [B, T, 1, D] single head
        kv = apply_interleaved_rope(kv, cos[:, :T], sin[:, :T]).squeeze(2)

        # local branch logits: [B, H, T, T] with sliding-causal mask
        logits = torch.einsum("bthd,bsd->bhts", q, kv).float() * self.scaling
        logits = logits.masked_fill(~sliding_mask, float("-inf"))

        aux_kl = None
        if self.compressor is not None:
            entries = self.compressor(x, cos_sin_c)  # [B, W, D]
            W = entries.shape[1]
            rate = self.layer_type
            e_logits = torch.einsum("bthd,bwd->bhtw", q, entries).float() * self.scaling
            # entry w (source tokens [w*rate, (w+1)*rate)) visible to query t
            # iff w < (t+1) // rate
            tpos = torch.arange(T, device=x.device)
            visible = torch.arange(W, device=x.device)[None, :] < ((tpos[:, None] + 1) // rate)
            e_logits = e_logits.masked_fill(~visible[None, None], float("-inf"))
            if self.indexer is not None:
                index_scores = self.indexer(x, q_resid, cos_sin_c)  # [B, T, W]
                index_scores = index_scores.masked_fill(~visible[None], float("-inf"))
                topk = min(self.topk, W)
                topk_idx = index_scores.topk(topk, dim=-1).indices
                keep = torch.zeros(B, T, W, dtype=torch.bool, device=x.device)
                keep.scatter_(-1, topk_idx, True)
                # KL(attention-over-entries || indexer) — teaches the scout
                with torch.no_grad():
                    tgt = e_logits.softmax(dim=-1).mean(dim=1)  # [B, T, W]
                    tgt = torch.nan_to_num(tgt)  # early queries see no entry -> all -inf row
                log_pred = index_scores.log_softmax(dim=-1).masked_fill(~visible[None], 0.0)
                log_pred = torch.nan_to_num(log_pred, neginf=0.0)
                aux_kl = (tgt * (tgt.clamp_min(1e-9).log() - log_pred)).sum(-1).mean()
                e_logits = e_logits.masked_fill(~(keep.unsqueeze(1) & visible[None, None]), float("-inf"))
            logits = torch.cat([logits, e_logits], dim=-1)
        # per-head learnable sink: one extra logit column that absorbs prob mass
        sink = self.sinks.float().view(1, -1, 1, 1).expand(B, -1, T, 1)
        probs = torch.cat([logits, sink], dim=-1).softmax(dim=-1)[..., :-1].to(x.dtype)

        if self.compressor is not None:
            v_all = torch.cat([kv, entries], dim=1)  # [B, T+W, D] (K=V!)
        else:
            v_all = kv
        out = torch.einsum("bhts,bsd->bthd", probs, v_all)

        # inverse rope on the output's rope slice (undo the phase V carried, at
        # the query position: rotation by -theta*t)
        out = apply_interleaved_rope(out, cos[:, :T], -sin[:, :T])

        grouped = out.reshape(B, T, -1).view(B, T, len(self.o_a_proj), self.group_in)
        grouped = torch.einsum("btgi,gio->btgo", grouped, self.o_a_proj).flatten(2)
        return self.o_b_proj(grouped), aux_kl


class HyperConnection(nn.Module):
    """mHC: turn hc_mult residual streams into (pre, post, comb) weights.
    pre collapses streams into the sublayer input; post places the sublayer
    output back onto streams; comb (Sinkhorn doubly-stochastic) mixes streams."""

    def __init__(self, config):
        super().__init__()
        self.hc = config.hc_mult
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        mix = (2 + self.hc) * self.hc
        self.fn = nn.Parameter(torch.empty(mix, self.hc * config.n_embd))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, streams):  # [B, T, hc, D]
        hc = self.hc
        flat = norm(streams.flatten(2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        s = self.scale.float()
        pre = torch.sigmoid(pre_w * s[0] + pre_b) + self.eps
        post = 2 * torch.sigmoid(post_w * s[1] + post_b)
        comb = torch.softmax(comb_w.view(*comb_w.shape[:-1], hc, hc) * s[2] + comb_b.view(hc, hc), dim=-1) + self.eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=2).to(streams.dtype)
        return post, comb, collapsed


class HyperHead(nn.Module):
    """Final stream collapse before the trunk's last norm."""

    def __init__(self, config):
        super().__init__()
        self.eps = config.hc_eps
        self.fn = nn.Parameter(torch.empty(config.hc_mult, config.hc_mult * config.n_embd))
        self.base = nn.Parameter(torch.empty(config.hc_mult))
        self.scale = nn.Parameter(torch.empty(1))

    def forward(self, streams):
        flat = norm(streams.flatten(2).float())
        pre = torch.sigmoid(F.linear(flat, self.fn.float()) * self.scale.float() + self.base.float()) + self.eps
        return (pre.unsqueeze(-1) * streams.float()).sum(dim=2).to(streams.dtype)


class NanoDSV4Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        c = config
        self.attn = V4Attention(c, layer_idx)
        router = "hash" if layer_idx < c.num_hash_layers else "noaux"
        self.mlp = MoEBlock(
            dim=c.n_embd, n_experts=c.n_routed_experts, expert_dim=c.moe_expert_dim,
            top_k=c.num_experts_per_tok, score_fn="sqrtsoftplus",
            routed_scaling=c.routed_scaling, n_shared=c.n_shared_experts,
            router_kind=router, vocab_size=c.vocab_size, swiglu_limit=c.swiglu_limit)
        self.attn_hc = HyperConnection(c)
        self.ffn_hc = HyperConnection(c)

    def forward(self, streams, flat_ids, cos_sin_main, cos_sin_c, sliding_mask):
        dtype = streams.dtype
        post, comb, collapsed = self.attn_hc(streams)
        xin = norm(collapsed)
        q_resid = self.attn.q_a_norm(self.attn.q_a_proj(xin))
        attn_out, aux = self.attn(xin, q_resid, cos_sin_main, cos_sin_c, sliding_mask)
        streams = post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), streams)

        post, comb, collapsed = self.ffn_hc(streams)
        mlp_out = self.mlp(norm(collapsed), flat_ids=flat_ids)
        streams = post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), streams)
        return streams, aux


class NanoDSV4(nn.Module):
    Config = NanoDSV4Config

    def __init__(self, config):
        super().__init__()
        assert len(config.compress_ratios) == config.n_layer
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(config.n_token_types, config.n_embd)
        self.h = nn.ModuleList([NanoDSV4Block(config, i) for i in range(config.n_layer)])
        self.hc_head = HyperHead(config)
        for kind, base in (("main", config.rope_base_main), ("compress", config.rope_base_compress)):
            cos, sin = precompute_interleaved_rope(config.sequence_len * 4, config.rope_dim, base)
            self.register_buffer(f"cos_{kind}", cos, persistent=False)
            self.register_buffer(f"sin_{kind}", sin, persistent=False)
        self._aux_loss = None

    @property
    def blocks(self):
        return self.h

    def get_device(self):
        return self.wte.weight.device

    def init_weights(self):
        c = self.config
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)
        gen = torch.Generator().manual_seed(1234)
        for block in self.h:
            a = block.attn
            init_linear_(a.o_a_proj.view(-1, a.o_a_proj.shape[-1]))
            nn.init.zeros_(a.o_b_proj.weight)                 # zero the final output proj
            nn.init.zeros_(a.sinks)
            for mod in (a.compressor, getattr(a.indexer, "compressor", None)):
                if mod is not None:
                    nn.init.zeros_(mod.position_bias)
            gu, dn = block.mlp.experts.gate_up, block.mlp.experts.down
            init_linear_(gu.view(-1, gu.shape[-1]))
            nn.init.zeros_(dn)
            nn.init.normal_(block.mlp.gate.weight, mean=0.0, std=0.02)
            if hasattr(block.mlp.gate, "e_score_correction_bias"):
                nn.init.zeros_(block.mlp.gate.e_score_correction_bias)
            if hasattr(block.mlp.gate, "tid2eid"):
                block.mlp.gate.init_table(generator=gen)
            nn.init.zeros_(block.mlp.shared_experts.down_proj.weight)
            for hc in (block.attn_hc, block.ffn_hc):
                nn.init.normal_(hc.fn, mean=0.0, std=0.02)
                nn.init.zeros_(hc.base)
                nn.init.ones_(hc.scale)
        nn.init.normal_(self.hc_head.fn, mean=0.0, std=0.02)
        nn.init.zeros_(self.hc_head.base)
        nn.init.ones_(self.hc_head.scale)
        nn.init.zeros_(self.type_emb.weight)
        for kind, base in (("main", c.rope_base_main), ("compress", c.rope_base_compress)):
            cos, sin = precompute_interleaved_rope(c.sequence_len * 4, c.rope_dim, base,
                                                   device=self.get_device())
            setattr(self, f"cos_{kind}", cos)
            setattr(self, f"sin_{kind}", sin)
        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)
            self.type_emb.to(dtype=torch.bfloat16)

    def estimate_flops(self):
        c = self.config
        total = sum(p.numel() for p in self.parameters())
        emb = self.wte.weight.numel() + self.type_emb.weight.numel()
        inactive = c.n_layer * (c.n_routed_experts - c.num_experts_per_tok) * 3 * c.n_embd * c.moe_expert_dim
        active_matmul = total - emb - inactive
        # attention: score+value against (T local logits + compressed entries)
        t = c.sequence_len
        attended = t + t // c.csa_rate  # dense logits materialized before masking
        attn = 6 * c.n_layer * c.n_head * attended * c.head_dim
        return 6 * active_matmul + attn

    def pop_aux_loss(self):
        aux, self._aux_loss = self._aux_loss, None
        return aux

    def forward(self, idx, token_types=None, kv_cache=None, block_mask=None):
        assert kv_cache is None and block_mask is None, "NanoDSV4 is training-only"
        B, T = idx.shape
        c = self.config
        x = self.wte(idx)
        if token_types is not None:
            x = x + self.type_emb(token_types)

        cs_main = (self.cos_main[:, :T].float(), self.sin_main[:, :T].float())
        cs_c = (self.cos_compress.float(), self.sin_compress.float())
        tpos = torch.arange(T, device=idx.device)
        sliding_mask = (tpos[:, None] >= tpos[None, :]) & (tpos[:, None] - tpos[None, :] < c.sliding_window)

        streams = norm(x).unsqueeze(2).expand(-1, -1, c.hc_mult, -1).contiguous()
        flat_ids = idx.reshape(-1)
        aux_sum, n_aux = None, 0
        for block in self.h:
            streams, aux = block(streams, flat_ids, cs_main, cs_c, sliding_mask)
            if aux is not None:
                aux_sum = aux if aux_sum is None else aux_sum + aux
                n_aux += 1
        if self.training and aux_sum is not None:
            self._aux_loss = c.aux_loss_weight * aux_sum / n_aux
        return norm(self.hc_head(streams))
