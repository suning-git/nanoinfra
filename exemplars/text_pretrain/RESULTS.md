# text_pretrain — capability log

What this project produces, through the blessed Orchestrator
`modalities.text.train_text`. Three stages: (1) the champion, (2) the scaling
law, (3) inference. How the recipe's `lr_max=3e-4` was chosen is
[`provenance.md`](provenance.md).

**Fixed facts:** dim = depth×64, vocab 32768, seed 42, compile_trunk=true (the key was named `use_compile` when these ran),
`head_ce=liger`, Chinchilla `target_param_data_ratio=20`. GPU: RTX 5090 (32 GB,
500 W-capped). FineWeb sample-10BT.

> **`head_ce` is part of the recipe, not the environment.** It used to be neither
> stated nor pinned — the head took liger whenever the package happened to be
> installed. That is why the MFU below is the one number here nobody can source: the
> package was uninstalled 2026-07-05, the champion trained 07-09, and it was
> reinstalled 07-10, with no record either way. Every number on this page is now
> stamped with the arm that produced it, and `spec.py` pins it.

## 1 · The champion (best recipe, trained once)

**d12 · dim 768 · 135,268,608 params · Chinchilla 2.705B tokens · 20,640 steps ·
lr 3e-4 · ~59 % MFU (⚠ arm unrecorded — see note above) · final val CE 3.8101** — the shipped checkpoint
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
[`scaling_law.png`](example_results/scaling_law.png), plotted as loss vs compute (C = 6ND), each
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
[`example_results/scaling.json`](example_results/scaling.json).

**Reproducing this on your own tokenizer: compare the exponent, not the CE.**
Every CE here is per-TOKEN, so it is only comparable inside one tokenizer
artifact — retrain the BPE and the merges change, each token carries a different
number of bytes, and the absolute level moves even on identical data. (§1 already
shows how badly a byte table can be misread: `token_bytes.pt` was once missing
and "bpb 5.497" was silently bits-per-token.) The exponent is a log-log slope and
does not care: an independent reproduction on a re-trained BPE recovered
**a = 0.508** against the 0.519 here. For a level you can compare across
tokenizers, read `val/bpb` — normalised per UTF-8 byte — which `scaling.py` now
records alongside CE in each trajectory point when a byte table is present. The
curves shipped above predate that, so they carry no bpb to compare against; the
exponent is what to check a reproduction by until a reference run records it.

## 3 · Inference

Continuations from the champion through the **core KV-cache engine**
(`core.model.inference.autoregressive_generate`), temp 0.8 / top-k 40 — coherent,
grammatical English at 135M params. Four samples in
[`example_results/samples.md`](example_results/samples.md); e.g. *"The history of the Roman
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

## 4 · The head's three arms

`head.loss` has three implementations, and they are three answers to one question —
that `[B,T,V]` logits tensor: **pay** for it (`naive`), dodge it with a hand-written
triton kernel (`liger`), or dodge it with inductor tiling the vocab dimension
(`compiled`). Switch with one word:

```bash
.venv/bin/python -m exemplars.text_pretrain.pretrain head_ce=compiled
```

Measured on this recipe's family, one RTX 5090, seq 512, vocab 32768, audit batch
sizes; each arm run twice on one RTX 5090, spread ≤1.1 %:

| depth | naive | liger | compiled |
|---|---|---|---|
| d6 (dim 384) | 24.4 % | 47.5 % | **60.8 %** |
| **d12 (this project)** | 53.5 % | 63.6 % | **73.1 %** |
| d20 (dim 1280) | 72.8 % | 75.3 % | **80.1 %** |

**And on §2's own ladder** — the five sizes trained for the scaling law, at that
stage's fixed shape (seq 1024, dbs 32, tbs 32768), single GPU:

| | params | head's share | naive | liger | compiled |
|---|---|---|---|---|---|
| d2 | 8.8 M | **48 %** | 6.6 % | 16.3 % | **33.9 %** |
| d3 | 13.9 M | 45 % | 10.7 % | 27.5 % | **45.7 %** |
| d4 | 19.9 M | 42 % | 15.5 % | 34.1 % | **53.5 %** |
| d6 | 35.8 M | 35 % | 26.5 % | 50.6 % | **64.3 %** |
| d8 | 58.7 M | 29 % | 37.5 % | 58.3 % | **70.0 %** |

The ratio tracks the head's share of the parameters exactly, because it is the same
fact stated twice. At d2 the un-embedding is *half the model*: the naive arm spends
93 % of the card on a `[32, 1024, 32768]` fp32 tensor that two of the three arms never
build. Worth knowing before you re-run the ladder — the curves in §2 cost what they
cost partly because of this.

Peak memory runs the other way — liger 3.32 / 6.79 / 10.99 GB against compiled
5.17 / 7.54 / 11.16 at d6/d12/d20 — so liger stays the answer when memory, not
throughput, is what binds. The head is a *large* fraction of a shallow model's step (a 384-wide trunk is
outweighed by its own un-embedding), which is why the spread collapses with depth.

Two things worth taking from this beyond the numbers:

**The arms differ in numerics, not only speed.** At step 8 of a fixed run:
naive 8.356687, compiled 8.356690, liger 8.358035. Compiled is the same math
reassociated; liger's fused kernel reduces in a different order, and that 1.3e-3 is
the size that once broke a 2-GPU loss-trajectory regression check at 5e-3. So this is a
recipe knob, not a deployment detail — which is why there is deliberately no
"auto": what a run computes must not depend on what is pip-installed.

**This project pins `liger` because its numbers were measured there, not because it
is the best arm.** `compiled` is faster at every depth and needs no optional package.
Adopting it means re-pinning §1–§2, which is a measured decision, not a config edit.
