# Data, model, and license boundary

| dependency | use in this project | included here? |
| --- | --- | --- |
| BONES-SEED / SEED license | trains the supplied rot139 motion codec and tracker corpus | No raw captures, derived caches, codec weights, or tracker shards |
| MotionHub HumanML3D/AMASS subset | captioned motions for the self-trained Text2Motion model | No dataset files or captions |
| HumanML3D and AMASS | source annotations/motions exposed through the prepared route | No; users must obtain them under their own terms |
| SMPL-H body model | rot139 forward kinematics and retarget handoff | No model assets |
| GMR | SMPL-H to Unitree G1 inverse-kinematics retargeting | No vendored checkout |
| `suning-git/motion_tracking` | G1 quality gates, frozen policies, and MuJoCo evaluation | No policies or generated reference shards |

The checked-in JSON files contain compact aggregate measurements only. The MP4s
are synthetic model/controller rollouts and do not contain source capture frames.
They are qualitative artifacts, not replacements for any gated dataset or model.

To reproduce the reported numbers, users must independently obtain each gated or
licensed dependency and provide compatible codec/tracker checkpoints. No weight
hash is published because the weights themselves are intentionally outside the PR.
