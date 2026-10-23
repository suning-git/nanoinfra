# Provenance — how `lr_max` was chosen

This is **not** one of the project's demos — it is the audit that fixed the LR in
`spec.py`, kept here so the recipe is justified rather than magic. (Re-tuning any
*other* knob would be a separate audit; the project just uses the settled values.)

## 2026-10: `lr_max = 6e-4` (fp32 storage, compiled head)

Three things have changed since the July bracket below. Training now stores the parameters, their gradients and
the AdamW moments in fp32 (the arithmetic stays under bf16 autocast); the champion uses the compiled head; and the
validation protocol changed — validation sequences now go through the same assembly as training — so the July CE
values cannot be compared with these. The protocol change alone overturns the July ranking: re-read on the current
protocol, the bracket's own bf16 checkpoints (step 20000 of each run) give 6e-4 3.4419 < 1.2e-3 3.4478 <
3e-4 3.5170 < 2.4e-3 3.5402 < 1.5e-4 3.6520. The LR was then re-chosen in the current setting — the blessed d12
Chinchilla recipe, seed 42, fp32 storage, `head_ce=compiled`, one GPU per run. Val cross-entropy at the last step
(20639), standard 2 M val window:

| lr_max | GPU | final val CE |
|--------|-----|-------------:|
| 3e-4     | RTX 5090 | 3.4373 |
| **6e-4** | RTX 5090 | **3.4120** |
| **6e-4** | A100     | **3.4026** |
| 1.2e-3   | A100     | 3.4105 |

The rule was written before the A100 runs: of 6e-4 and 1.2e-3, take the one with the lower val CE, and 6e-4
if the two are within 0.01 of each other. 6e-4 is lower by 0.008, so `lr_max = 6e-4`. That is a selection rule, not a proof that the
two are equivalent. Compare only within one GPU type: the same recipe reads 0.009 apart on the two. On the 5090,
3e-4 trails 6e-4 by 0.025. The upper side was probed only up to 1.2e-3, which did not win.

These runs used the code of early 2026-10, before rows were redefined: a row then held 511 positions
(`sequence_len` counted the window's tokens), where the current code trains rows of 512.

## 2026-07: the LR bracket in bf16 storage (history)

Everything from here through "The lesson" was measured with bf16 storage, a head arm that was not recorded
(see the `head_ce` note in RESULTS.md) and the July validation protocol; it is kept as the record of how 3e-4
was chosen then.

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
through the run, ≈85 % through its warmdown) with the TRUE token-byte table, standard 2 M val window;
the U's ordering is unchanged under this protocol. The CE column remains each
run's final-step eval — checkpoints read ~0.005–0.010 higher (the missing
warmdown tail). The originally logged "bpb" column was bits-per-token
(missing `token_bytes.pt` → silent ones fallback), since fixed.

Full per-run metrics + val-CE trajectories: [`example_results/bracket.json`](example_results/bracket.json).

**Byte-table note (2026-09-11).** Every bits/byte figure measured before 2026-09-11
used a `token_bytes.pt` (sha256 `481782d8…`) that overstated the byte length of 216
of the 32768 ids: the byte-fallback ids 128–255 and 88 merges ending mid-character
were counted through U+FFFD (3 bytes) instead of their real 1–5 bytes. The corrected
table (sha256 `27f38b8f…`, built by `build_token_bytes` from the unchanged
`tokenizer.pkl`) makes the denominator smaller, so those bpb values read LOW by a
corpus-dependent fraction — 0.0825% on a 1.4M-token FineWeb shard-0 sample, 0.0904%
on shard 5. The figures above are kept as measured, under the old table; CE, tokens,
and checkpoints are unaffected. A re-measurement under the new table would be listed
separately with its date, not patched in here.

## Verdict (July, bf16 storage): `3e-4` wins — a clean U

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
better, 1.2e-3 −0.074. On the July validation protocol, at the **full Chinchilla
horizon that inverted**: raising LR above 3e-4 did not pay off, and the inherited
conservative 3e-4 looked right all along. (Step 20000 already gave the July order —
3e-4 3.8165 < 6e-4 3.8209 in `bracket.json` — so the 2026-10 re-read of the same
step-20000 checkpoints on the current protocol, which ranks 6e-4 first by 0.075, is a
protocol-only comparison: the July verdict came from the protocol. The corollary below
stands.)

Corollary worth internalizing: **never rank an LR before its warmdown tail
finishes.** 3e-4 trailed 6e-4 at 96.5% of the run, then its final linear warmdown
dropped it into the lead. Read only completed runs.

## Re-tuning for YOUR model (if you fork this)

Change the model in `spec.py`, then sweep the LR with the project's own
machinery — set `spec.LR_MAX` to each candidate and run stage 1:

```bash
# for each candidate: edit LR_MAX = "<lr>" in spec.py, then
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m exemplars.text_pretrain.pretrain
```

`spec.ckpt_dir()` tags checkpoints by (depth, lr), so candidate runs never
collide. Compare only **completed** runs (warmdown included) and pick the U's
minimum.

## tokenizer.pkl provenance

The exemplar's world is a rustbpe BPE, vocab 32768, with the 18 control tokens
bundled as the last ids. The original artifact was trained on FineWeb during the
project's earliest bring-up. The maintained rebuild entry is now
`python -m modalities.text.train_tokenizer` (same recipe: FineWeb shards from
base_data/, control registry from modalities/control) — retraining yields a
same-shaped but not id-identical vocab, so checkpoints bind to the exact artifact
they were trained with.
