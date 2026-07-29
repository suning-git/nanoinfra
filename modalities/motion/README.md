# modalities/motion — the motion modality

Human motion as tokens, in the same vocabulary and the same GPT as text. This
package owns the three things a modality has to own: how it declares itself to the
vocabulary, how it turns motion into integers, and how raw capture data becomes the
features the tokenizer eats.

```
modalities/motion/
├── __init__.py     manifest(codec) + FakeMotionCodec (CPU tests) + TYPE_ID=1
├── tokenizers/     the tokenizer SHELF
│   ├── _convae/       architecture FAMILY (conv autoencoder): nets, quantizers,
│   │                    models, MotionCodec, train harness. `_` = machinery, not
│   │                    a tokenizer; a future arch is a sibling (e.g. _transformer/)
│   ├── rot139_kin_fsq2/  a tokenizer = FACADE (load()/train()) + recipe.yaml + card.md
│   ├── rot139_vqvae/     the same shape, a different quantizer
│   └── REGISTRY.md    the shelf index: what each artifact is and what it proved
└── data/           the reusable data leg
    ├── converters/    SMPL/BVH <-> rot139, geometry
    ├── loaders/       amass, lafan
    ├── dataset.py     load_or_build / Normalizer / make_windows
    ├── fk_torch.py    differentiable FK (rot139 -> root-relative positions)
    ├── sources.py     MotionDataSource / T2MDataSource (core DataSource contract)
    └── paths.py       the data-layout authority (datasets/ vs cache roots)
```

`exemplars/nano_motion` is the worked example that uses all of it end to end —
data preparation, training a tokenizer, training a text→motion model, sampling.

## Usage

```python
from modalities.motion.tokenizers import rot139_kin_fsq2     # a shelf tokenizer
codec = rot139_kin_fsq2.load(device="cuda")                  # facade — never touch _convae/
from modalities.motion.data import dataset, fk_torch         # the data leg
from modalities.motion.data.sources import T2MDataSource     # core DataSource
```

A tokenizer folder is a facade: open it, call `load()`, use it. The architecture
machinery lives in `_convae/` and you should not need to read it to train or use a
codec — `train()` takes the folder's `recipe.yaml` and reproduces the artifact.

## The representation

- **rot139** — absolute joint rotations, root displacement and height, foot
  contacts. 139 numbers per frame.
- **kinematic loss** — FK position + velocity + foot contact. The single biggest
  lever on reconstruction quality, which is why the training harness carries it
  rather than leaving it to each caller.
- **Two quantizers on the shelf** — `MotionVQVAE` (EMA codebook) and `MotionFSQ2`
  (bijective FSQ). Which one an artifact uses is checkpoint DATA, not a call-site
  choice; `MotionCodec` dispatches at load. The contract either way is
  `.vocab_size .downsample .d_feat .rep`, `encode([T,139]) -> list[int]`,
  `decode(codes) -> [T',139]`.

## Two artifacts, strong on different axes

There is no single "best" codec here, and the shelf is built to say so rather than
to hide it:

| artifact | strength | what is not established |
|---|---|---|
| `fsq2_bones_512` | reconstruction champion — 10.09 cm / 4.2° on the fair global metric, from a matched-budget comparison | generation quality not eyeball-verified |
| `vqvae_amass_512` | the only one whose generation was checked by looking at it (5/6 prompts good) | reconstructs worse, 16.4 cm |

Pick by what your task needs and read the card. Full details in
[tokenizers/REGISTRY.md](tokenizers/REGISTRY.md).

**A measurement worth carrying:** an earlier round of this work crowned a different
representation (`hml263`) as the reconstruction champion. That was a metric
artifact — the metric was heading-blind, so a representation that lost heading
information scored well. The "fair global" numbers above are measured in a global
canonical frame precisely because of it. When a representation comparison produces
a surprising winner, suspect the ruler before the result.

## Generation quality is open

The reconstruction numbers are measured and solid. Generation quality is not a
settled fact of this package: a text→motion model trained on one of these codecs
looked clearly worse than an earlier one trained on the other, with several
confounds unseparated — codec, training data, caption style, and how much of the
apparent quality was memorization. One factor is confirmed: models trained on
full-sentence captions answer full-sentence prompts and go out of distribution on
terse ones. The rest is open.
