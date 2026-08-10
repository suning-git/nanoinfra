# Text2Motion → Motion Cerebellum — capability log

This project connects a pretrained text-to-motion generator to a separately
trained whole-body tracking policy for the Unitree G1:

```text
text → OMG 50M → G1 reference → adapter → motion_tracking actor → MuJoCo G1
```

The final controller is the upstream actor architecture, trained from random
initialization. It is not a post-processing filter and no residual, preview, or
oracle controller is used in the reported path.

## Pinned main result

Three policies were trained independently with seeds 0, 1, and 2. Each used
1,500 references, 4,500 PPO iterations, 3,840 parallel environments, 80 workers,
the upstream `g1/deployable` recipe, and observation noise. Evaluation used the
same 60 native held-out clips and three fixed OMG prompts for all policies, with
four noise repeats per item.

| Domain | Success | Completion | Empjpe | Foot slide | Jerk |
|---|---:|---:|---:|---:|---:|
| Native held-out | **91.67 ± 0.83%** | 96.01 ± 0.38% | 29.31 ± 0.58 mm | 5.246 ± 0.047 | 5.101 ± 0.076 |
| Three OMG prompts | **94.44 ± 4.81%** | 99.75 ± 0.22% | 28.52 ± 1.77 mm | 4.748 ± 0.166 | 4.467 ± 0.193 |

The `±` values are sample standard deviations across the three independently
trained policies. The corresponding df=2 t intervals are stored in
[`results/main_results.json`](results/main_results.json). Every seed passed the
frozen native floor (80% success, 90% completion) and the three-prompt demo floor
(two-thirds success, 85% completion).

## Preregistered stress test

The three-prompt demo was not treated as evidence for open-vocabulary success.
A 12-prompt suite was frozen before generation: the original three plus nine new
prompts, one generation attempt per new prompt, with no rerolls.

| Stage | Result |
|---|---:|
| New references passing the upstream physical-quality gate | **3 / 9 (33.3%)** |
| Tracker success on all six quality-passing prompts | **81.94 ± 2.41%** |
| End-to-end success over all 12 preregistered prompts | **40.97 ± 1.20%** |

The expanded demo therefore fails its preregistered claim. Six new references
were rejected before tracking: continuity (2), excessive root speed (2), joint
speed (1), and foot slide (1). The tracker also struggled with the accepted
right-sidestep reference. The result localizes two separate limitations: high-level
reference generation is the main end-to-end bottleneck, while lateral tracking is
a specific low-level weakness.

Machine-readable summaries are in
[`results/expanded_prompt_results.json`](results/expanded_prompt_results.json)
and
[`results/prompt_quality_diagnostics.json`](results/prompt_quality_diagnostics.json).

## Post-hoc frozen-reference repair

After the preregistered stress test failed, two diagnostic repair passes reused
the exact same nine generated motions. Neither pass regenerated, rerolled, or
tuned a rule for an individual prompt. These are post-hoc follow-ups and do not
replace the failed preregistered result above.

| Result | Original | Repair v1 | Repair v2 |
|---|---:|---:|---:|
| New references passing the quality gate | 3/9 | 6/9 | **6/9** |
| Tracking success among quality-passing references | 81.94% (6 refs) | 73.15% (9 refs) | **75.93% (9 refs)** |
| End-to-end success over all 12 prompts | 40.97% | 54.86% | **56.94%** |

V1 obtained gate compliance partly by stretching the two discontinuous clips to
16.96–19.80 seconds. V2 instead repaired local joint jumps and capped time stretch
at 2×, shortening them to 6.84–7.28 seconds. This raised mean end-to-end success
by another 2.08 percentage points over v1, but the final acceptance decision still
failed: seed 1 had 72.22% tracking success and 85.39% completion, and the kick
reference failed all 12 noisy episodes. V2 also had slightly worse mean Empjpe,
foot slide, and jerk than v1, so it is a useful localization result rather than a
new main claim. Compact results and source hashes are stored in
[`results/reference_repair_results.json`](results/reference_repair_results.json).

## Post-hoc generator-stage attribution

The adapter code audit found that OMG already outputs Unitree G1 `qpos_36`.
There is no SMPL-to-G1 retargeting in this bridge: it validates the joint order,
normalizes quaternion signs, and resamples the motion from 30 to 50 Hz. Replaying
the quality measurements on both sides of that bridge showed that all six frozen
rejections already violated the corresponding constraint in the raw 30 Hz OMG
motion. This rules out the adapter as the cause of those six failures.

Five of the six rejected motions had their largest joint step at transition
59→60. We therefore ran an exploratory comparison using the documented OMG
condition-sequence interface, with the same nine prompt texts, one generation per
prompt/variant, and no rerolls:

| OMG generation mode | Quality pass | Largest step at 59→60 |
|---|---:|---:|
| One 60-frame chunk (`text: prompt`) | **6/9** | 0/9 |
| Two 60-frame chunks (`text[2]: prompt`) | **3/9** | 8/9 |

The one-chunk path recovered backward walk, both-arms raise, and right-leg kick.
Left sidestep and jog still exceeded the root-speed gate, while squat exceeded
the hover gate. This is a post-hoc, single-generation comparison whose variants
differ in duration, so it is not a statistical estimate of general model quality.
It nevertheless gives a concrete next action: repair or avoid the multi-chunk
boundary and repeat across generation seeds before retraining the whole generator.
For action classes that still fail inside one chunk, a different pretrained
generator or targeted Text2Motion fine-tuning remains justified. This diagnostic
does not indicate that the tracker itself should be retrained. Compact evidence
and source hashes are stored in
[`results/generator_diagnosis_results.json`](results/generator_diagnosis_results.json).

## Multi-seed short-horizon follow-up

We first tested the continuation mechanism already present in the pinned OMG
planner on jog, both-arms raise, and right-leg kick, with generation seeds 0–2.
All three continuation settings reduced some boundary jumps, but each still
passed 0/9 quality gates. The strongest setting reduced the mean worst seam step
from 1.91 to 0.96 rad per source frame; that remained insufficient, and jog also
contained excessive speed inside the first chunk. Existing continuation is
therefore not a complete long-horizon repair.

We then repeated the full nine-prompt comparison across generation seeds 0–2.
The first 60 frames were paired exactly within each prompt and seed:

| Variant | Quality pass | 59→60 is largest step |
|---|---:|---:|
| Single 60-frame action | **21/27 (77.8%)** | 0/27 |
| Baseline two chunks | **9/27 (33.3%)** | 22/27 |

The paired cells were 9 both-pass, 12 single-only-pass, 0 double-only-pass, and
6 both-fail. All 21 quality-passing single-chunk references were then evaluated
on all three frozen trackers with four noise repeats; the other six remained
end-to-end failures rather than being rerolled.

| Metric | tracker seed 0 | tracker seed 1 | tracker seed 2 | mean ± sample SD |
|---|---:|---:|---:|---:|
| Tracking success on 21 accepted references | 82.14% | 80.95% | 78.57% | **80.56 ± 1.82%** |
| Completion | 94.62% | 94.56% | 93.74% | **94.31 ± 0.49%** |
| End-to-end success over all 27 cells | 63.89% | 62.96% | 61.11% | **62.65 ± 1.41%** |

Every tracker passed the frozen 75% tracking, 90% completion, and 60% overall
end-to-end floors. This establishes a post-hoc **short-horizon** multi-generation-
seed demo, not a repair of long-context OMG. It is also generation-seed sensitive:
seed 0's end-to-end result was below 60% for each tracker. Bow, backward walk,
wave, and both-arms raise tracked perfectly across their accepted generations;
kick (12.5%) and lateral motion (45.8–50.0%) remain clear low-level weaknesses.
Machine-readable results and source hashes are in
[`results/short_horizon_results.json`](results/short_horizon_results.json).

## Long-horizon repair and adaptation follow-up

The frozen 27 two-chunk cells were next evaluated with no rerolls. A direct C1
residual decay improved quality from 9/27 to 14/27 but lost one previously
passing cell. Treating horizontal translation and yaw as a rigid planar symmetry,
then correcting only unsafe joint channels, preserved all 9 original passes and
reached 15/27. A separately labeled adapter sanitizer recovered two foot-slide
and one joint-velocity failure, reaching the preregistered 18/27 quality floor:

| Stage | Quality pass | Preserved prior passes | Eligible for tracking |
|---|---:|---:|---:|
| Raw two-chunk OMG | 9/27 | — | no |
| Unconstrained C1 residual | 14/27 | 8/9 | no |
| Planar-space-aligned selective C1 | 15/27 | 9/9 | no |
| Reason-specific adapter sanitizer | **18/27** | 15/15 | **yes** |

The sanitizer is not reported as a generator improvement: raw OMG remains
9/27, and seven final rejections still contain excessive root speed. All 18
sanitized references were evaluated on the three frozen trackers with four
noise repeats (216 episodes):

| Metric | tracker seed 0 | tracker seed 1 | tracker seed 2 | mean ± sample SD |
|---|---:|---:|---:|---:|
| Tracking success | 65.28% | 72.22% | 73.61% | **70.37 ± 4.46%** |
| Completion | 84.09% | 87.04% | 87.25% | **86.13 ± 1.76%** |
| End-to-end success over 27 cells | 43.52% | 48.15% | 49.07% | **46.91 ± 2.98%** |

No tracker passed both the frozen 75% success and 90% completion floors, and
none passed the 60% end-to-end floor. The failures were concentrated: bow and
right-hand wave were 100%, backward walk was 97.22%, and both-arms raise was
80.56%, while kick was 0%, right sidestep 20.83%, and squat 16.67%.

Two held-out-generation-seed adaptation smokes were then run for tracker seed 0.
Training on 12 long references for 300 iterations raised held-out success only
58.33→62.50% while reducing native success 91.67→73.33%. Adding 120 native
replay references did not fix the route: held-out success fell to 50.00% and
native success to 76.25%. Both preregistered expansion decisions were false, so
the other two trackers were not fine-tuned. The evidence rejects small-corpus
adaptation, not all tracker retraining; a credible next attempt needs a much
larger mixed long-action corpus and a held-out semantic split. Compact evidence
and hashes are in
[`results/long_horizon_results.json`](results/long_horizon_results.json).

## What this establishes

The evidence supports a reproducible Text2Motion-to-tracker demo, a three-seed
official-rollout-scale tracking baseline, and a post-hoc short-horizon extension
in simulation. The long-horizon follow-up supplies a negative result rather than
upgrading that claim. It does **not** establish reliable long-context generation,
generation-seed invariance, open-vocabulary reliability, real-robot deployment,
a universal 10% reduction in foot sliding or jerk, or an exact reproduction of
every upstream paper result.

## Qualitative demo

Three side-by-side videos are included under [`assets/demo/`](assets/demo/):
forward walking, left turn, and right turn. The left/blue skeleton is the OMG
reference and the right/orange skeleton is the closed-loop G1 rollout from the
preferred clean-start seed-0 policy.

## Quick verification

From a clean repository checkout, without datasets, checkpoints, or MuJoCo:

```bash
python3 -m unittest discover \
  -s projects/text2motion_cerebellum -p 'test_*.py' -v
```

The dependency-light tests cover the adapter contract, quaternion continuity,
30→50 Hz resampling, tracker reference serialization, fixed-protocol aggregation,
preregistered prompt accounting, source-stage attribution, and consistency
between this report and the tracked result files. Full physics reruns require the
external artifacts described in [`DATA_AND_LICENSES.md`](DATA_AND_LICENSES.md).
