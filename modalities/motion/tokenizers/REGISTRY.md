# motion tokenizers — the shelf (REGISTRY)

Trained tokenizer artifacts on the shelf, and the tokenizer-dependent data
caches they produced. **On-shelf = has a birth certificate**, not "looks good".

Projects USE these as a user and PROMOTE mature work back here as a developer;
day-to-day experimentation runs on copies inside the project, never here.

**Layout:** each tokenizer is a folder = a self-serve facade (`load()`) +
recipe + card. Open the folder, call `load()`, use it — you never touch the
architecture machinery in `_convae/`.

```python
from modalities.motion.tokenizers import rot139_kin_fsq2
codec = rot139_kin_fsq2.load(device="cuda")      # -> the shelf contract
codes = codec.encode(features139)                # .vocab_size / .downsample / encode / decode
```

(`_convae/` = the conv-autoencoder architecture family: nets, quantizers,
MotionCodec, the training harness. A different architecture would be a sibling
family folder, not an edit inside _convae.)

---

## 1. Tokenizer artifacts

Each has a folder (façade + card.md) here; weights live under `models/motion/`
(the promoted-artifact root, gitignored). Every codec is rot139 → 512 codes,
downsample 4 (4 frames/code @30fps), width 512.

| shelf name | file (`models/motion/`) | quantizer | recon (fair global) | generation | trained on | date |
|---|---|---|---|---|---|---|
| **fsq2_bones_512** | `codec_rot139_kin_fsq2_bones_seed_30k.pt` | FSQ2 [8,8,8] | **10.09 cm / 4.2°** (N=300 fair) — matched-budget comparison winner | **not eyeball-verified**; a t2m model on it looked poor, with prompt-style mismatch one confirmed factor | Bones-SEED 30k steps | 2026-07-06 |
| **vqvae_amass_512** | `vqvae_amass_512.pt` | VQ-EMA | AMASS fair recon 16.4 cm (weaker) | **eyeball-verified** — an AMASS t2m d6 model, 5/6 prompts good | AMASS+HumanML3D | 2026-06 |
| _(bench-only)_ vq_bones_512 | `codec_rot139_kin_vq_bones_seed_30k.pt` | VQ-EMA | 10.13 cm / 4.7° | — | Bones-SEED 30k | 2026-07-06 |

**Certificate honesty is the point of the shelf:** the two headline artifacts are
strong on DIFFERENT axes — FSQ2-bones is the reconstruction champion but its
generation is unverified; vqvae_amass is the only one whose generation was
checked by looking at it, but it reconstructs worse. There is no single "best"
codec; pick by what your task needs, and read the certificate. The vq_bones arm is
kept only as the arbitration counterpart (reproduces the research 10.13/4.7°).

Reproduce: each tokenizer folder's `train()` runs its `recipe.yaml` through the
shared harness in `_convae/train.py` and rebuilds the artifact.

## 2. Tokenizer-dependent caches

Code streams and t2m pairings are DERIVED from a specific tokenizer, so they live
in `outputs/motion_caches/` (regenerable) rather than in `datasets/`, which holds
only tokenizer-INDEPENDENT format versions. The tag is the tokenizer that made them.

| cache pattern | tokenizer (tag) | producer | notes |
|---|---|---|---|
| `<dataset>_{split}_codes_k512.npz` | the tokenizer that encoded it | `exemplars/nano_motion/data/encode.py` | motion code streams |
| `t2m_<dataset>_{split}.npz` | same | `exemplars/nano_motion/data/encode.py` | (codes, caption) pairs |

Rule: a cache without a known tokenizer tag is untrustworthy. Codes are integers;
codes from two different codecs are the same integers and look identical on disk,
so a mislabelled cache trains a model against the wrong decoder and nothing
complains. That happened once, which is why the tag is recorded here.
