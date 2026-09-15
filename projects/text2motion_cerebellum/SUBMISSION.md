# Submission scope

This project follows the NanoInfra convention: project source lives under
`projects/`, while data, checkpoints, and generated outputs remain outside Git.

## Include

- `README.md`, `RESULTS.md`, `FINAL_REPORT_CN.md`, and `DATA_AND_LICENSES.md`;
- the OMG-to-tracker adapter and the final evaluation/review utilities;
- dependency-light tests;
- `expanded_prompts_v1.json` (the preregistered protocol);
- the explicitly labeled post-hoc frozen-reference repair protocols and tests;
- the post-hoc generator-stage attribution and chunk-boundary diagnostic;
- the multi-generation-seed short-horizon protocol, tracker review, and explicit
  short-reference loader override;
- the long-horizon planar seam, adapter-sanitization, three-tracker evaluation,
  and held-out domain-adaptation negative result;
- three compact skeleton demo videos and their poster images;
- compact JSON summaries under `results/`.
- the self-trained NanoInfra generator extension under
  `projects/nano_motion_motionhub/` and `projects/nano_motion_cerebellum/`,
  including compact aggregate results and three additional demo videos.
- the `nano_motion` supervision-config fix required for the documented training
  command to construct a valid sequence recipe.

## Exclude

- `datasets/`, `models/`, and `outputs/`;
- BONES-SEED BVH files and all derived reference shards;
- OMG and tracker checkpoints;
- all other MP4/WebM renders and raw episode logs;
- AutoDL credentials, SSH material, fleet registries, raw provider logs, and
  machine-specific task state;
- abandoned residual/oracle experiments and one-off remote queue wrappers from
  the final project change set.

## Pre-commit checks

```bash
python3 -m unittest discover \
  -s projects/text2motion_cerebellum -p 'test_*.py' -v
python3 -m unittest discover \
  -s projects/nano_motion_cerebellum -p 'test_*.py' -v
git diff --check
git status --short
```

Review the staged set explicitly before committing. The intended change set is the
three project directories, the repository-root README discovery link, and the
single `exemplars/nano_motion/configs/train_t2m.yaml` supervision fix. Edits under
`remote/autodl/` are infrastructure work and must be submitted separately, if at
all.

The exact allow-list is [`submission_files.txt`](submission_files.txt). It omits
the abandoned residual/oracle experiments and machine-specific training wrappers
that remain in the working directory for research history. Use this allow-list
instead of staging the whole untracked project directory.
