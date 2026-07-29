# RESULTS — what a working run looks like

Reference numbers, so you can tell whether your run is training or broken.
RTX 5090, 12 layers / 768 dim / 12 heads (0.14B), sequence length 256.

## Is the pipeline alive? (a minute)

After `data/encode.py`, on a small LAFAN1 cache:

```bash
python -m exemplars.nano_motion.train_t2m \
    max_steps=6 optimizer.scheduler.warmup_steps=2 evaluation.interval_steps=3 \
    use_compile=false checkpoint.enabled=false device_batch_size=4
```

```
nano_motion — motion on lafan1, codec rot139_kin_fsq2
  vocab 33280 across 3 bands; motion band at 32768 (512 codes)

Step 00000  loss 10.41   val/motion_ce 10.33
Step 00003               val/motion_ce  9.11
Step 00005  loss  9.55   val/motion_ce  8.31
```

Two things to check before the loss values. **The vocabulary line**: the motion band
must be 512 codes wide and start where the text and control bands end — a band at the
wrong offset trains perfectly and decodes into nonsense. **The starting loss**: about
ln(512) ≈ 6.2 if the model is only ever choosing among motion codes, higher when the
sequence includes text positions, but never zero and never flat.

The same command under `torchrun --nproc_per_node=2 ... parallel=ddp` reaches the
same place — 8.30 against 8.31 at step 5, the difference being which rows each rank
happened to see.

## Training the codec

```bash
python -m exemplars.nano_motion.train_codec --steps 120     # ~6 s, is it wired
```

```
[fsq2] train (12616, 64, 139) val (2788, 64, 139) | params 18.27M
[fsq2] step 120  loss 3.374  ppl 29 | root-rel 17.59cm *best*
[fsq2] FAIR global 9262.57 cm | heading 18.7 deg
```

**Read `root-rel`, not the global number, at this step count.** The global metric
integrates root displacement across the window, so a model that has not yet learned
to stand still walks tens of metres off and the number looks broken when it is merely
early. `root-rel` is scored per frame against the root and means something at any
step count; the global metric becomes readable only once displacement is learned.

`ppl 29` out of 512 codes at step 120 says the codebook is being used rather than
collapsing to a handful of entries — which is the other thing that can quietly go
wrong in a quantizer and does not show up in the loss.

## Codec reconstruction

The tokenizer sets the ceiling: the AR model can only produce motion the codec can
represent. Reconstruction of the shelf artifacts, MPJPE in a **global canonical**
frame (N=300, stride-64 windows):

| codec | MPJPE | rotation | generation |
|---|---:|---:|---|
| `rot139_kin_fsq2` | **10.09 cm** | 4.2° | not eyeball-verified |
| `rot139_vqvae` | 16.4 cm | — | eyeball-verified, 5/6 prompts |

Before blaming the AR model for mushy samples, decode a real clip's own codes and
look at that — it is the best the AR model could possibly do.

These numbers are only comparable to numbers from the same metric. An earlier round
of this work crowned a different representation under a heading-blind metric, which
flattered anything that discarded heading; the winner changed when the metric was
fixed.

## What is not established

Generation quality. A model trained on the reconstruction champion looked clearly
worse by eye than an earlier one trained on the weaker codec, and the confounds —
codec, training data, caption style, memorisation — were never separated. One factor
is confirmed: a model trained on full-sentence captions goes out of distribution on
terse ones.

No number in this file should be read as a generation-quality claim.
