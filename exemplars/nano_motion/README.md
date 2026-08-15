# nano_motion — human motion as tokens, in the same GPT as text

Train a transformer to generate human motion, in the same vocabulary the text
models use. A caption goes in as text tokens, motion comes out as motion tokens,
and the same trunk that predicts the next word predicts the next quarter-second of
movement. That is the point of the exercise: motion is not a special case, it is
another band in one vocabulary.

Everything from raw motion capture to a rendered stick figure is here — preparing
the data, training the tokenizer, training the model, sampling, rendering.

```
data/            capture files -> rot139 features -> code streams
train_codec.py   rot139 -> 512 discrete codes (a conv autoencoder + quantizer)
train_t2m.py     THE ORCHESTRATOR — read this first
generate.py      prompt -> band-masked sampling -> codes -> features
render.py        features -> SMPL forward kinematics -> stick-figure GIF
spec.py          the knob panel
```

Unlike `nano_world_model` next door, this exemplar trains its own tokenizer. Video
codecs are large and borrowing one is the sane default; a motion codec is small
enough to train in half an hour, and training it is worth seeing — it is the step
that decides the ceiling on everything after it.

## Run it

```bash
# 0. the text tokenizer, which sizes the shared vocabulary (seconds)
python -m modalities.text.train_tokenizer

# 1. motion capture data (LAFAN1: free, no account, no captions)
python -m exemplars.nano_motion.data.download

# 2. raw -> rot139 features (minutes)
python -m exemplars.nano_motion.data.prepare

# 3. train the tokenizer (~30 min on one GPU; --steps 200 to check the wiring)
python -m exemplars.nano_motion.train_codec

# 4. features -> code streams
python -m exemplars.nano_motion.data.encode --codec models/motion/<what step 3 wrote>

# 5. train the model, on one GPU or several
python -m exemplars.nano_motion.train_t2m
torchrun --nproc_per_node=2 --standalone -m exemplars.nano_motion.train_t2m parallel=ddp

# 6. sample and render
python -m exemplars.nano_motion.generate --ckpt exemplars/nano_motion/models/<run>/step_XXXXX
```

Text→motion needs captions, which means AMASS + HumanML3D and accepting their
licenses; LAFAN1 alone trains an unconditional motion model through the identical
path. [data/README.md](data/README.md) covers both and what each costs.

---

## Three things worth knowing

### 1. The vocabulary is assembled, not configured

```
[  text band  |  control  |  motion band  ]
   32768          18           512 codes
```

`train_t2m.py` builds this from three manifests and derives every offset. The motion
band's size is read **off the codec**, not written in a config — the codec is what
decides how many distinct codes exist, and a band that reserves the wrong number
puts motion tokens on top of whatever follows. `vocab_size` and `n_token_types` are
therefore facts of the assembly, and deliberately not settable in the YAML: a config
that can disagree with the bands eventually will.

### 2. Supervision is a property of the batch, not of the loop

In text→motion the model should learn to produce motion given a caption, not to
produce captions. So the loss covers the motion half only. That does not need a
custom trainer or a mask argument threaded through core: the data source marks each
token's weight, and the loader turns unsupervised positions into `IGNORE_INDEX`
targets. Core's fused cross-entropy already honours that, so the caption half costs
nothing and core learns nothing about what "supervised" means here.

### 3. The codec sets the ceiling, so it is trained with kinematics in the loss

Reconstruction error on the 139 features is not the same as error you can see. A
small rotation mistake at the hip moves the foot a long way, and a codec tuned on
feature MSE alone will happily trade a visibly sliding foot for a better number.
`train_codec.py` therefore runs the decoded rotations through forward kinematics and
penalises joint POSITION, velocity and foot contact as well. That term was the
single biggest lever on reconstruction quality in the work this comes from.

---

## Tests

The multi-GPU checks live next door in `exemplars/nano_world_model/tests/test_ddp.py`
and cover `core.parallel.NanoDDP` itself, which this exemplar uses through the same
`Trainer(ddp=...)` seam. Nothing here re-tests core.

## What is open

**Generation quality is not a settled fact.** The reconstruction numbers for the
shelf codecs are measured and solid. Generation is not: a model trained on the
reconstruction-champion codec looked clearly worse than an earlier one trained on
the weaker codec, and the confounds — codec, training data, caption style, and how
much of the apparent quality was memorisation — were never separated. One factor is
confirmed: a model trained on full-sentence captions goes out of distribution on
terse ones, which is why `spec.PROMPTS` are full sentences.

This exemplar exists to make that question reproducible, not to claim it is answered.
