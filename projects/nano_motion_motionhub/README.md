# Training `nano_motion` on the MotionHub/HumanML3D route

This directory contains the data preparation and model-selection utilities used
to train a NanoInfra Text2Motion model for the Motion Cerebellum demo. It does not
redistribute motion capture, captions, SMPL assets, codec weights, or checkpoints.

## Data preparation

`prepare.py` resolves the fixed MotionHub revision, downloads only the requested
HumanML3D/AMASS SMPL-H subset, converts MotionHub Y-up coordinates to the course
Z-up convention, encodes the motion with a supplied rot139 codec, and writes the
captioned caches expected by `exemplars.nano_motion.train_t2m`.

```bash
python projects/nano_motion_motionhub/prepare.py \
  --starter /path/to/humanml3d_course_starter \
  --cache-dir /path/to/huggingface-cache \
  --out-dir /path/to/nanoinfra/outputs/motion_caches \
  --codec /path/to/codec.pt \
  --report outputs/nano_motion_motionhub/prepare.json \
  --train-limit 10000 --val-limit 650 --workers 24
```

The reported run pinned `ZeyuLing/MotionHub` revision
`c3f6c8eb8a4ba9e5ca521cdc0af9264756b66726` and produced:

| split | clips | caption-motion pairs |
| --- | ---: | ---: |
| train | 9,968 | 29,760 |
| validation | 649 | 1,937 |

All geometry, floor, translation-spike, usable-fraction, and codec-domain gates
passed. Median root-relative codec reconstruction MPJPE was 6.34 cm on train and
7.07 cm on validation.

## Training

The model uses the repository's existing `exemplars/nano_motion` orchestrator:
12 layers, width 768, 12 attention heads, sequence length 256, and a 33,280-token
assembled vocabulary (about 140M parameters). The current configuration expresses
the recorded motion-only loss directly as `supervise: [motion_tokens, motion_end]`;
the equivalent current command is shown below.

```bash
torchrun --nproc_per_node=2 --standalone \
  -m exemplars.nano_motion.train_t2m \
  source=motionhub codec_ckpt=/path/to/codec.pt parallel=ddp \
  model.depth=12 model.dim=768 model.n_head=12 model.n_kv_head=12 \
  sequence_len=256 max_steps=20000 \
  evaluation.interval_steps=500 checkpoint.save_every=500
```

Training was stopped after the 5,000-step review rather than continued blindly:
validation CE reached 2.8259 at step 3,000 and worsened to 3.1004 by step 5,000.
The frozen checkpoint rule ranked balanced left/right semantic pass rate first,
overall pass rate second, and validation CE third. It selected step 2,500:

| fixed 16-seed prompt | pass rate |
| --- | ---: |
| walk forward | 12/16 |
| turn left | 2/16 |
| turn right | 4/16 |

`compare_checkpoints.py` and `analyze_turn_sweep_v3.py` implement these recorded
rules. Tracker outcomes are not inputs to checkpoint selection.
