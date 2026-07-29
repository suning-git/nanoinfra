# rot139_vqvae — birth certificate

**What it is:** rot139 features → 512 codes via VQ-EMA codebook, conv-AE family
(convae), width 512, code_dim 512, downsample 4 (4 frames/code @30fps). The
tokenizer behind the eyeball-verified AMASS text-to-motion result.

**Objective: PLAIN feature MSE + commitment (NO kinematic loss).** Verified from
the checkpoint's training `log` (pure feature-MSE curve; the `result` field is
`{val_feature_mse 0.40, codes_used 512}` — no kin lambdas). Kinematic loss was a
LATER lever found during tokenizer research; this predates it. (This corrected an
earlier mis-label as "rot139_kin_vqvae" — the artifact says plain.)

**Weights:** `models/motion/vqvae_amass_512.pt` (promoted from a research
checkpoint; moving-in smoke passed — loads and round-trips, |err| 0.079).

**Trained:** AMASS. Exact steps and lr are not fully pinned by the checkpoint —
its log shows feature MSE over at least 5k steps at a 500-step cadence — so
recipe.yaml records what is recoverable and is marked best-effort. An artifact
whose recipe cannot be fully recovered says so on its card.

**Verified:**
- ✅ **Generation** — eyeball-verified: AMASS text→motion (d6),
  5/6 prompts clearly good. The only shelf tokenizer whose GENERATION was
  checked by eye, which is the harder bar.
- ⚠️ **Reconstruction** — weaker: AMASS fair global-canonical recon ~16.4 cm, and
  val_feature_mse 0.40 at train time. Not the recon champion; rot139_kin_fsq2 is.

**Why on the shelf despite weaker recon:** it and rot139_kin_fsq2 bracket the two
validation axes — this one has verified GENERATION, fsq2 has verified
RECONSTRUCTION. Neither dominates; the shelf holds both, honestly.
