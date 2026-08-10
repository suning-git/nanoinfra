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
- compact JSON summaries under `results/`.

## Exclude

- `datasets/`, `models/`, and `outputs/`;
- BONES-SEED BVH files and all derived reference shards;
- OMG and tracker checkpoints;
- all MP4/WebM renders, poster images, and raw episode logs;
- AutoDL credentials, SSH material, fleet registries, raw provider logs, and
  machine-specific task state;
- abandoned residual/oracle experiments and one-off remote queue wrappers from
  the final project change set.

## Pre-commit checks

```bash
python3 -m unittest discover \
  -s projects/text2motion_cerebellum -p 'test_*.py' -v
git diff --check
git status --short
```

Review the staged set explicitly before committing. The intended change set is the
project directory plus the repository-root README discovery link; edits under
`remote/autodl/` are infrastructure work and must be submitted separately, if at
all.

The exact allow-list is [`submission_files.txt`](submission_files.txt). It omits
all rendered media, the abandoned residual/oracle experiments, and
machine-specific training wrappers that remain in the working directory for
research history. Use this allow-list instead of staging the whole untracked
project directory.
