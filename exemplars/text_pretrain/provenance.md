# Provenance — how `lr_max = 3e-4` was chosen

This is **not** one of the project's demos — it is the audit that fixed the LR in
`spec.py`, kept here so the recipe is justified rather than magic. (Re-tuning any
*other* knob would be a separate audit; the project just uses the settled values.)

## The LR bracket (production horizon)

Five learning rates, everything else the blessed d12 Chinchilla recipe (2.705B
tokens, seed 42, ~4.4 h each on one 5090). Final val cross-entropy:

| lr_max | final val CE | val bits/byte* | mean MFU | wall |
|--------|-------------:|--------:|---------:|-----:|
| 1.5e-4   | 3.8436 | 1.248 | 58.8% | 4.46 h |
| **3e-4** | **3.8101** | **1.236** | 59.1% | 4.43 h |
| 6e-4     | 3.8150 | 1.238 | 58.3% | 4.51 h |
| 1.2e-3   | 3.9307 | 1.275 | 57.5% | 4.57 h |
| 2.4e-3   | 3.8643 | 1.255 | 58.3% | 4.51 h |

\* bits/byte re-measured on each run's kept checkpoint (step 20000, ≈97 %
through warmdown) with the TRUE token-byte table, standard 2 M val window;
the U's ordering is unchanged under this protocol. The CE column remains each
run's final-step eval — checkpoints read ~0.005–0.010 higher (the missing
warmdown tail). The originally logged "bpb" column was bits-per-token
(missing `token_bytes.pt` → silent ones fallback), since fixed.

Full per-run metrics + val-CE trajectories: [`results/bracket.json`](results/bracket.json).

## Verdict: `3e-4` wins — a clean U

```
1.5e-4 (3.8436)  >  3e-4 (3.8101)  <  6e-4 (3.8150)  <  2.4e-3 (3.8643)  <  1.2e-3 (3.9307)
```

The minimum sits at **3e-4**, both edges probed. 3e-4 and 6e-4 are a near-tie
(gap 0.0049), both clearly beating the higher LRs. Above the tie the shape is
non-monotone — 1.2e-3 is *worse* than 2.4e-3 — because 1.2e-3 sits just past
stable and **regresses mid-run** (val rises while train falls), while 2.4e-3's
larger warmdown rescues it harder. Neither is usable.

## The lesson (why this bracket exists)

A short-horizon audit (1500 steps) had said *"raise the LR"* — 6e-4 looked −0.049
better, 1.2e-3 −0.074. At the **full Chinchilla horizon that inverts**: raising LR
above 3e-4 does not pay off. Short probes overestimate the optimum; the inherited
conservative 3e-4 was right all along.

Corollary worth internalizing: **never rank an LR before its warmdown tail
finishes.** 3e-4 trailed 6e-4 at 96.5% of the run, then its final linear warmdown
dropped it into the lead. Read only completed runs.

## Re-tuning for YOUR model (if you fork this)

Change the model in `spec.py`, then sweep the LR with the project's own
machinery — set `spec.LR_MAX` to each candidate and run stage 1:

```bash
# for each candidate: edit LR_MAX = "<lr>" in spec.py, then
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exemplars/text_pretrain/pretrain.py
```

`spec.ckpt_dir()` tags checkpoints by (depth, lr), so candidate runs never
collide. Compare only **completed** runs (warmdown included) and pick the U's
minimum.
