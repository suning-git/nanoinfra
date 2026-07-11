# example: residual-connection ablation

A **worked negative ablation** — the shape of a Hackathon experiment start to
finish. It removes one thing from the transformer, the **residual (identity)
connection**, and measures what that costs: same model, same data, same budget,
one architectural change, two training curves.

This is the `projects/` tier: a fork of an exemplar with something changed. Point
a newcomer here to see *"how do I ablate an architecture and read the result?"*

## The ablation

A pre-norm transformer block normally keeps an identity path around each
sub-layer:

```
x = x + attn(norm(x))
x = x + mlp(norm(x))
```

`trunk.py` removes the `x +` — nothing else:

```
x = attn(norm(x))
x = mlp(norm(x))
```

That is the whole change. Residual connections are what let gradients flow
cleanly through a deep stack (the ResNet insight); remove them and the same
model, on the same data, converges to a much worse loss — and the gap opens in
the first handful of steps.

### One honest subtlety (in `trunk.py`)

The reference GPT zero-inits each block's output projection so a block *starts* as
the identity `x + 0 = x`. That trick only makes sense with a residual path —
without it a zeroed projection makes every block output 0 (a dead, gradient-free
network). So `NoResidualGPT` re-inits those projections normally. That keeps the
comparison a clean single variable: the only difference from the baseline is the
residual connection itself.

## How it's wired

No core edit, no forked training loop. `NoResidualGPT` (`trunk.py`) subclasses the
reference GPT in ~a dozen lines and satisfies the same trunk contract, so it drives
through the **same** blessed orchestrator (`modalities.text.train_text`) — selected
by one config knob, `model.trunk_class`. That is the framework's pluggable-trunk
seam: to change the architecture you provide a trunk, you do not patch the shared
core.

| file | what |
|------|------|
| `trunk.py` | `NoResidualGPT` — the architecture change |
| `spec.py`  | the recipe (depth, budget, the two arms) — the one knob |
| `run.py`   | trains both arms through the orchestrator, collects the val curves |
| `plot.py`  | the two curves → `residual_ablation.png` |

## Run it

```bash
# once: fetch a FineWeb shard (shared with the text exemplar)
python exemplars/text_pretrain/data/download_shards.py

python projects/example_residual_ablation/run.py     # trains baseline + no_residual (d6, minutes)
python projects/example_residual_ablation/plot.py    # -> residual_ablation.png
```

Defaults to a tiny **d6** smoke scale (a couple of minutes on one GPU). Raise
`DEPTH` in `spec.py` and the gap only grows — residuals matter more the deeper
the stack.

## Result

<!-- filled from an actual d6 run (~20M tokens, one FineWeb shard, constant LR) -->
![residual ablation](residual_ablation.png)

The two arms start together at the trivial init loss (~10.2 CE), descend together
for the first ~20 steps, and then split hard:

| arm | val CE @ step 5 | val CE @ end (step 1219) |
|-----|----------------:|-------------------------:|
| baseline (GPT) | 10.26 | **5.43** |
| no residual    | 10.12 | **7.86** |

The residual-free model trains — it is not dead — but it **flattens at ~7.9 and
cannot use its depth**, while the baseline keeps descending to 5.43. That is a
**+2.43 CE (≈2.4 nats)** gap, and it opens within the first ~60 steps (it clears
0.5 CE by step 59 of 1220). At d6 this is already stark; deeper stacks only widen it.

**Conclusion — residual connections are load-bearing, not optional.** Remove the
identity path and the same model, on the same data and budget, gives up ~2.4 nats
of cross-entropy, and it shows almost immediately.
