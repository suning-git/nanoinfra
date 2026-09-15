# Data, models, and licenses

Source code, reports, preregistered prompt definitions, compact aggregate
results, and three skeleton demo videos belong in this repository. Raw captures,
derived reference shards, checkpoints, episode logs, and full render sets remain
external.

## External components

| Component | Use | Distribution in this repository |
|---|---|---|
| [motion_tracking](https://github.com/suning-git/motion_tracking) | G1 reference format, quality gate, PPO tracker, evaluator | Not vendored; pinned to commit `a3b8d0c092684f1307c53175d94260c4ff323306`. Upstream code is MIT-licensed. |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Unitree G1 model | Not vendored; pinned to commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`. |
| OMG 50M | Pretrained text-to-G1 motion generator | Weights are not redistributed. Users must obtain them under the upstream terms. |
| BONES-SEED | Training motion capture | Raw and converted data are not redistributed. Access requires accepting the SEED license. |

The repository-level MIT license covers this project's original source code. It
does not relicense external datasets, pretrained weights, robot assets, or outputs
derived from them. Anyone reproducing the experiment is responsible for obtaining
those components and complying with their terms.

## Qualitative renders

`assets/demo/` contains forward-walk, left-turn, and right-turn comparisons. They
show projected skeleton coordinates: the OMG reference beside the simulated G1
rollout. They are not used to compute the reported metrics.

## External artifact contract

The frozen experiment used:

- three policy files, one for each training seed;
- 28 training-reference shards containing the 1,500-reference samples;
- two native held-out reference shards;
- three original OMG prompt reference shards.

The policy files are not distributed as part of this project submission. The
checked-in aggregate results and skeleton-only demo videos are sufficient to
review the submitted project; rerunning checkpoint-dependent evaluation requires
separately obtained external artifacts.

The complete local evidence bundle remains under the ignored `outputs/` root.
The checked-in result JSON files record source-artifact provenance so a
maintainer with that bundle can confirm that the compact summaries came from the
frozen evidence.

## Expected private layout

No absolute machine path is part of the interface. Reproduction scripts should be
given explicit paths to:

```text
<external-data>/bones_seed/...
<external-models>/omg/...
<external-models>/motion_tracking/policy_*.pt
<repo>/outputs/remote_text2motion_mainline/...
```

This preserves NanoInfra's separation of source, `datasets/`, `models/`, and
`outputs/` and makes the submission portable between local and remote machines.
