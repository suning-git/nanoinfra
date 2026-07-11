# text_pretrain — capability log

What this project produces, through the blessed Orchestrator
`modalities.text.train_text`. Three stages: (1) the champion, (2) the scaling
law, (3) inference. How the recipe's `lr_max=3e-4` was chosen is
[`provenance.md`](provenance.md).

**Fixed facts:** dim = depth×64, vocab 32768, seed 42, use_compile=true, liger
fused-CE head, Chinchilla `target_param_data_ratio=20`. GPU: RTX 5090 (32 GB,
500 W-capped). FineWeb sample-10BT.

## 1 · The champion (best recipe, trained once)

**d12 · dim 768 · 135,268,608 params · Chinchilla 2.705B tokens · 20,640 steps ·
lr 3e-4 · ~59 % MFU · final val CE 3.8101** — the shipped checkpoint
(step_020000, ≈97 % through warmdown) re-measures **1.236 bits/byte**
(CE 3.8165) on the standard 2 M val window. (An earlier "bpb 5.497" figure
was bits-per-token: `token_bytes.pt` was missing and the byte table silently
fell back to ones — since installed.) Checkpoint:
`models/exemplars/text_pretrain_d12_lr3e-4/step_020000`. This is the model
`inference.py` samples from; the scaling law (§2) characterises the same
architecture family (dim = depth×64) at smaller sizes.

## 2 · The compute-optimal scaling law

Per-model **training curves → the compute-optimal frontier.** Five sizes (d2–d8)
are each trained once, through the SAME Orchestrator, to **2 B tokens each** —
every size gets the same budget ON PURPOSE: small models pancake on their floor
(d2 ends at ~5100 tokens/param) while the largest has just cleared its bend
(d8: ~79, 4× Chinchilla), which is what makes the curves cross, the frontier
exist, and every curve's bend AND flattening visible. CONSTANT
LR (scheduler warmdown off, so each curve is genuine loss-vs-compute) with a
**fixed 200-step warmup** (absolute, not a ratio — the recipe must not drift when
the budget changes); validation evaluated at **40 log-spaced steps** from step 20
(~0.66 M tokens) — the schedule is computed by `scaling.py` and injected through
the core evaluator's `eval_at` (core never computes schedules). In
[`scaling_law.png`](scaling_law.png), plotted as loss vs compute (C = 6ND), each
size drops → bends → flattens to its floor; the **lower envelope is the
compute-optimal frontier** — at each budget several sizes compete and one (★) is
optimal.

> **N_opt ∝ C^a,  a ≈ 0.52**  (Chinchilla ≈ 0.5)

Cross-checked against the pre-refactor pipeline at ITS budget (500 M): the same
fitter on both worlds' raw curves gives **0.478 (new) vs 0.474 (old)** — two
entirely different data pipelines, agreement to 0.004 (the old report's own
quote, 0.54, uses a stricter overlap-range convention). The estimate grows with
the covered budget — **0.449 / 0.478 / 0.506 / 0.519 at 300 M / 500 M / 1 B /
2 B** — as the envelope's high-compute end fills in; report the budget with the
number. The
refactor preserves the scaling law. One deliberate simplification: all sizes
share the champion's LR (3e-4); a precision study would tune LR per size (the
pre-refactor study lists the same caveat).

**A measurement note worth keeping** (→ vision's measurement-culture list): the
frontier exponent is acutely sensitive to how EARLY each curve is sampled. A first
attempt evaluated at a fixed LINEAR interval whose first point landed at ~12 M
tokens — the early low-compute segment (where small models are compute-optimal over
a wide range) was missing, the smallest and largest sizes never shared a compute
budget, and the exponent inflated to **0.79**. Sampling from step 20 restores the
full range. (The first fix densified linear eval and thinned points post-hoc; the
log-spaced `eval_at` schedule in core has since retired that workaround — the
figure plots the actual evaluation points.)

`eval_tokens` = 512 K per point — **measured as sufficient**: the same trajectory
read through 128 K and 512 K windows shows the same ~0.009 per-point jitter, i.e.
the wiggle is genuine model-state fluctuation under constant LR (trajectory-
dominated), so a larger window buys no smoothness. A scaling study needs RELATIVE
CE and the fixed val prefix keeps the bias consistent across curves. Single
curves also keep a run-to-run character — two d8 runs differing only in warmup
length wiggle in different places — while the frontier is robust to it (see the
budget progression above). Raw curves + fit:
[`results/scaling.json`](results/scaling.json).

## 3 · Inference

Continuations from the champion through the **core KV-cache engine**
(`core.model.inference.autoregressive_generate`), temp 0.8 / top-k 40 — coherent,
grammatical English at 135M params. Four samples in
[`results/samples.md`](results/samples.md); e.g. *"The history of the Roman
Empire"* → *"…and the birthplace of the Roman Empire. It was founded by the
monarchy of the 3rd century B.C. …"*. The trained model runs end-to-end as an LM,
not just a loss number.

**The budget ladder** ([`inference_compare.md`](inference_compare.md)): the same
spec at four budgets — 20M / 131M / 1.1B / 2.7B tokens (2 min → 4.4 h) =
**1.759 / 1.411 / 1.251 / 1.236 bits/byte** (shipped checkpoints, common 2 M
window), samples side by side. Big gaps are
visible in generation (prompt-following, then coherence, emerge rung by rung);
the last 0.015 is not — the loss metric resolves what the eye cannot. All
checkpoints self-describe and load via `load_system`.
