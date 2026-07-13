# example: GPT-2 vs a modern architecture

A **generational A/B** — not an ablation. It trains the classic 2019 GPT-2 /
minGPT trunk against the current core GPT on the same data, budget, and recipe,
and reads the two curves. Where an ablation removes one thing to show it matters,
this measures what the *accumulated* modern changes buy, together.

This is the `projects/` tier: a fork of an exemplar with something changed. Point
a newcomer here to see *"has the architecture actually improved since GPT-2, and
by how much?"*

## The two architectures

Both are pre-norm decoder-only transformers with the SAME width/depth and the
SAME (untied) `LMHead`; only the trunk body differs:

| | GPT-2 style (`gpt2.py`) | modern core GPT |
|---|---|---|
| position | learned absolute embedding (`wpe`) | RoPE (no position embedding) |
| normalization | LayerNorm (learnable weight + bias) | RMSNorm (no learnable params) |
| MLP activation | tanh-GELU | ReLU² |
| linear layers | with biases | no biases |
| attention | plain multi-head | + QK-norm, GQA-capable |

The GPT-2 trunk borrows its architecture from the course's `minimal_gpt.py`
(minGPT gpt-nano). Note minGPT-nano itself does **not** tie its head, so the
shared untied `LMHead` is faithful to both — the curves isolate the *trunk*.

## How it's wired

No forked training loop: `GPT2Trunk` (`gpt2.py`) satisfies the trunk contract
(`forward -> hidden`, `blocks`, `estimate_flops`, `Config`) and drives through the
**same** orchestrator (`modalities.text.train_text`), selected by one config knob,
`model.trunk_class`. That is the framework's pluggable-trunk seam — a genuinely
different architecture, not a tweak of GPT.

> **Gotcha worth knowing if you write your own trunk.** The framework builds the
> trunk on the `meta` device and calls `init_weights()` *after* `to_empty(device)`,
> so **nothing is initialized by default** — `init_weights` must set *every*
> parameter, LayerNorm included. (The modern GPT sidesteps this by using a
> parameter-free `F.rms_norm`; a GPT-2-style LayerNorm has real params that must be
> initialized explicitly, or the block outputs collapse and the model never trains.)

| file | what |
|------|------|
| `gpt2.py` | `GPT2Trunk` — the classic architecture as a trunk |
| `gpt2_rope.py` | `GPT2RoPETrunk` — `gpt2.py` with learned-pos → RoPE, the single-change ablation rung |
| `spec.py`  | the recipe (depth, budget, the two arms) — the one knob |
| `run.py`   | trains both arms through the orchestrator, collects the val curves |
| `plot.py`  | the two curves → `gpt2_vs_modern.png` |

## Run it

```bash
# once: fetch a FineWeb shard (shared with the text exemplar)
python download_data.py

python run.py     # trains modern + gpt2 (d6, minutes)
python plot.py    # -> gpt2_vs_modern.png
```

## Result

<!-- filled from an actual d6 run (~20M tokens, one FineWeb shard, constant LR) -->
![gpt2 vs modern](gpt2_vs_modern.png)

| arm | val CE @ end (step 1219) |
|-----|-------------------------:|
| modern GPT | **5.43** |
| GPT-2 + RoPE | 5.55 |
| GPT-2 style | 5.84 |

The modern architecture ends **~0.41 CE lower** — a real win, but an **incremental**
one. (Contrast the [residual ablation](https://github.com/suning-git/understand_transformer_by_ablation/tree/main/suning/example_residual_ablation): removing the
residual connection costs +2.4 CE — *load-bearing*; the accumulation of RoPE /
RMSNorm / ReLU² / no-bias is worth ~0.4 here.) The field moved for good reasons,
but no single 2019→2025 change is make-or-break the way the residual is.

**The middle arm is the ablation — it attributes that 0.41.** GPT-2 + RoPE
(`gpt2_rope.py`) is the classic trunk with the learned absolute-position embedding
swapped for RoPE and **nothing else changed**. That single swap buys **0.28 of the
0.41** (5.84 → 5.55); everything else the modern trunk piles on — RMSNorm, ReLU²,
no-bias, QK-norm — buys the remaining **0.12** (5.55 → 5.43). At this scale **RoPE
is ~70% of the whole 2019→2025 architecture gain**: not make-or-break like the
residual, but by far the most load-bearing of the incremental changes. On the plot
the purple curve hugs GPT-2 early, breaks away mid-run, and settles between the two
— closer to modern's floor.

**And note the crossover.** For the first ~100 steps GPT-2 is actually *ahead* — it
descends a touch faster early. The modern trunk overtakes around step ~110 and pulls
away, winning the **floor**, not the opening. A good habit to take from this: don't
rank architectures (or recipes) by their first hundred steps — read the curve out to
where it flattens.

**Extension (a great Hackathon project):** turn this A/B into a *ladder* —
GPT-2 → +RoPE → +RMSNorm → +ReLU² → … → modern — to see which single change
contributes what. The **+RoPE** rung is already built (`gpt2_rope.py`, the middle
arm above); add `+RMSNorm`, `+ReLU²`, `+no-bias` trunks the same way — each a
one-line delta from the rung before it — to attribute the rest of the 0.12.
