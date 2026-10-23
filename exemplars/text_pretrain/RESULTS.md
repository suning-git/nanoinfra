# text_pretrain — capability log

What this project produces, through the blessed Orchestrator
`modalities.text.train_text`. Three stages: (1) the champion, (2) the scaling
law, (3) inference. How the recipe's `lr_max` was chosen is
[`provenance.md`](provenance.md).

**Fixed facts:** dim = depth×64, vocab 32768, seed 42, compile_trunk=true (the key was named `use_compile` in
July), Chinchilla `target_param_data_ratio=20`, FineWeb sample-10BT. The head arm, the storage and the GPU differ
between sections and are stated in each.

> **Storage precision.** §1's champion and §2's curves and exponent were measured with the parameters, their
> gradients and the AdamW moments stored in fp32, core's default since 2026-10 (`param_dtype: fp32`; the arithmetic
> stays under bf16 autocast). §2's measurement notes, §3's budget ladder and §4's throughput tables are earlier
> measurements (July–August 2026) in bf16 storage, core's only option then, kept as measured. In bf16 storage an
> update smaller than half a bf16 step of the weight is rounded away, so compare against those only in bf16 storage
> (`param_dtype=bf16`).

> **`head_ce` is part of the recipe, not the environment.** It used to be neither
> stated nor pinned — the head took liger whenever the package happened to be
> installed, and the July champion's arm went unrecorded (the package was uninstalled
> 2026-07-05, the champion trained 07-09, and it was reinstalled 07-10). Every number
> on this page is now stamped with the arm that produced it. `spec.py` and `scaling.py`
> both pin `compiled`; `scaling.py` pinned `liger` until 2026-10-22, and §2's curves
> were measured with it.

## 1 · The champion (best recipe, trained once)

**d12 · dim 768 · 135,268,608 params · Chinchilla 2.705B tokens · 20,640 steps · lr 6e-4 · `head_ce=compiled` ·
fp32 storage · final val CE 3.4068, 1.1112 bits/byte** — the last step's checkpoint read on the standard 2 M val
window on one GPU. Trained 2026-10-13 on 8 × A100 80 GB with `pretrain.py --nproc 8 --parallel ddp`, 44.6 min of
training. Checkpoint: `models/exemplars/text_pretrain_d12_lr6e-4/step_020639` (a `--parallel ddp` run saves under
`…_lr6e-4_ddp/` instead, which `inference.py --parallel ddp` reads), the model `inference.py` samples from; the scaling law (§2) characterises the same architecture family (dim = depth×64) at smaller sizes. How
`lr_max` was chosen: [`provenance.md`](provenance.md).

- **Read the window on one GPU.** The run's own last evaluation printed 3.4162. With 8 processes, each reads its
  own row groups of the validation shard for an eighth of the 2 M tokens, so the window is a different set of
  documents from the one a single process reads; the same checkpoint read on one GPU gives 3.4068. Validation CE
  taken with different numbers of processes is not directly comparable.
- **The same recipe elsewhere** (each read on one GPU, one run each): one RTX 5090, 3.4120 (3.8 h, about 70 % MFU);
  one A100, 3.4026. Both ran the code of early 2026-10, whose rows held 511 positions, with the whole trunk compiled
  on one process; the 8-A100 run is the current code (rows of 512, blocks compiled one by one under ddp). They also
  differ in the batch sequence (8 ranks read different documents at each step) and the GPU type, and single runs
  cannot separate these; all three agree within 0.01.
- **Throughput.** 8 A100s took 44.6 min against 230 min for one A100 (the early-2026-10 code): 5.2×, about 0.64 of
  linear. The step times fall into bands that interleave through the whole run — about a fifth of the steps at
  85–99 ms, most at 120–140 ms, about a sixth at 150–170 ms, against 79 ms for linear scaling — so most of the loss
  comes from the four steps in five that land in the slower bands, not from a floor every step pays. The cause was
  not isolated. An earlier 8-A100 test of this recipe's previous version measured 0.77 of linear at this batch and
  0.96 with an 8× larger batch, which is a different recipe.
- **July's champion** (lr 3e-4, bf16 storage, head arm unrecorded) ended at val CE 3.8101 on the July validation
  protocol, which has since changed (validation sequences now go through the same assembly as training); its
  step-20000 checkpoint reads 3.5170 on the current one (on its own 511-position rows). It is kept as the top rung
  of §3's budget ladder.

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

> **N_opt ∝ C^a,  a ≈ 0.50**  (0.501; Chinchilla ≈ 0.5)

These curves: one A100 per size, fp32 storage, lr 3e-4 shared by all sizes, `head_ce=liger`, the code of early
2026-10; `scaling.py` pins this recipe except the head, `compiled` since 2026-10-22. Checked on the A100 over each
size's first 200M tokens (2026-10-22): compiled against these liger curves at the same 28 evaluation steps differs
by 0.004–0.005 in mean absolute val CE per size, at most 0.026, with no consistent sign; the 2B-token exponent was
not re-measured. The curves this page shipped before (July: one RTX 5090 per size, bf16 storage) gave 0.519, and the
same ladder in bf16 storage on the A100 gives 0.508. Moving to fp32 storage changed the exponent by −0.007, which
comes from the compute budgets where the optimal size switches moving by up to about 12 %; what it changed most is
the level — each size ends 0.07–0.13 lower in val CE at 2 B tokens.

On the July curves: cross-checked against the pre-refactor pipeline at ITS budget (500 M), the same
fitter on both worlds' raw curves gives **0.478 (new) vs 0.474 (old)** — two
entirely different data pipelines, agreement to 0.004 (the old report's own
quote, 0.54, uses a stricter overlap-range convention). The estimate grows with
the covered budget — **0.449 / 0.478 / 0.506 / 0.519 at 300 M / 500 M / 1 B /
2 B** — as the envelope's high-compute end fills in; report the budget with the
number. The
refactor preserves the scaling law. One deliberate simplification: all sizes
share one LR (3e-4, the champion's when these curves were measured); a precision study would tune LR per size (the
pre-refactor study lists the same caveat).

**A measurement note worth keeping**: the
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
number of bytes, and the absolute level moves even on identical data. (A byte
table can also be misread: `token_bytes.pt` was once missing, and an early
"bpb 5.497" was silently bits-per-token.) The exponent is a log-log slope and
does not care: an independent reproduction on a re-trained BPE recovered
**a = 0.508** against the July curves' 0.519 (a different measurement from the A100
bf16 ladder's 0.508 above). For a level you can compare across
tokenizers, read `val/bpb` — normalised per UTF-8 byte — which `scaling.py` now
records alongside CE in each trajectory point when a byte table is present. The
curves shipped above record it at every point, so a reproduction on another
tokenizer can compare levels in bits per byte as well as the exponent.

## 3 · Inference

Continuations from the champion through the **core KV-cache engine**
(`core.model.inference.autoregressive_generate`), temp 0.8 / top-k 40, each prompt
framed as training opens a row (`bos`, `text_start`, then the text) and stopped at the
next `bos` (where the training stream ends a document) — coherent,
grammatical English at 135M params. Four samples in
[`example_results/samples.md`](example_results/samples.md); e.g. *"The best way to learn
a new language is"* → *"to take a course in it. I have learned an entirely new
language and learning to speak it is very different from learning a new language. …"*.
The trained model runs end-to-end as an LM, not just a loss number.

**The budget ladder** ([`inference_compare.md`](inference_compare.md); July: bf16
storage, lr 3e-4, kept as measured): the same spec at four budgets — 20M / 131M / 1.1B / 2.7B tokens (2 min → 4.4 h) =
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
.venv/bin/python -m exemplars.text_pretrain.pretrain head_ce=liger   # needs the liger extra
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

**The champion pins `compiled`** — faster at every depth, and no optional package. The scaling study
(`scaling.py`) pinned `liger`, the arm its shipped curves were measured on, until 2026-10-22, and pins `compiled`
now too; the curves in §2 are still the liger ones.
