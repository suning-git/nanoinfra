# rot139_kin_fsq2 — birth certificate

**What it is:** rot139 features → 512 codes via FSQ2 [8,8,8], conv-AE family
(convae), width 512, code_dim 512, downsample 4 (4 frames/code @30fps).

**Weights:** `models/motion/codec_rot139_kin_fsq2_bones_seed_30k.pt`

**Trained:** matched-budget comparison against the VQ arm, 30k steps, AdamW 2e-4,
kinematic loss (pos+vel+foot, λ 100/100/100), win 64 / stride 32, 500k-window
subsample of Bones-SEED train (128,679 clips / 28.6M frames = 6.7× AMASS,
actor-split val), seed 0, best-val (root-rel) selection. 2026-07-06.

**Verified:**
- ✅ **Reconstruction** — fair global-canonical MPJPE **10.09 cm / 4.2°**
  (N=300, stride-64 windows, rng(0)); the arbitration WINNER over VQ (10.13/4.7°).
- ❌ **Generation** — NOT eyeball-verified. A t2m model (d12) on this codec looked
  poor by eye against an earlier one on `rot139_vqvae`; prompt-style mismatch is a
  confirmed factor and the rest is unseparated. See REGISTRY.md.

**Provenance:** the search that found this recipe is not shipped; what is here is
the final recipe and a `train()` that reproduces the artifact from it.
