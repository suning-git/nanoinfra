# Checkpoints

A checkpoint is a directory, `step_NNNNNN` under a run's save directory, holding the model's and the optimizers' state
in PyTorch's distributed checkpoint format (DCP), plus a `meta.json` that describes it and records where the data and
the Trainer stood. The Trainer writes one when the config asks; a run starts from one to resume (the model, the optimizers, the step
and the data position) or to start from trained weights (the weights only); inference loads one.

```text
<save_dir>/step_002000/
  .metadata                    DCP's index of the tensors: names, shapes, dtypes
  __0_0.distcp, __1_0.distcp   the tensor data, one file per rank: model weights and optimizer state
  meta.json                    written last, by rank 0: step, trainer and data state, the model's description
```

| Piece | Code |
|---|---|
| saving: `save_checkpoint_if_needed`, `save_checkpoint_dcp`, `should_save_checkpoint`, `cleanup_old_checkpoints`, `merge_data_snapshots` | [checkpoint_manager.py](checkpoint_manager.py) |
| starting from one: `load_checkpoint_if_needed`, `resolve_checkpoint_path`, `find_latest_checkpoint`, `load_checkpoint_dcp`, `load_model_only` | [checkpoint_manager.py](checkpoint_manager.py) |
| reading one: `load_metadata`, `get_model_config`, `config_from_meta` | [checkpoint_manager.py](checkpoint_manager.py) |
| `load_system` (inference) | [../training/model_setup.py](../training/model_setup.py), described in [../training/training.md](../training/training.md) |
| the optimizer hyperparameters on a resume | [../training/optim_resume.py](../training/optim_resume.py), described in [../training/training.md](../training/training.md) |
| the data snapshot | [../data/data_pipeline.md](../data/data_pipeline.md), "Resume" |

The examples on this page run on the CPU, in order, share their names, and write under
`/tmp/core_docs_example/`.

## When the Trainer saves

With `checkpoint.enabled: true` (the Trainer's default is false; the sample configs' `train_base.yaml` sets true),
the Trainer calls `save_checkpoint_if_needed` after every step, and it saves when `should_save_checkpoint` says so:

- at the launch's last step (`max_steps - 1`, or where `num_iterations` stops it), always;
- at every step that is a multiple of `checkpoint.save_every`; with `save_every: -1`, only at the last step.

Each save writes `<checkpoint.save_dir>/step_<step, six digits>` (an existing directory of that name is replaced), then
keeps the newest `checkpoint.keep_last_n` of the run's `step_*` directories (-1: all; 0 and other negatives are
refused) and deletes the older ones. Directories with a larger step than the current one (left by an earlier, longer
run in the same place) are reported and kept. Here is the Trainer's call for a 12-step run saving every 5 steps and
keeping 2:

```python run
import contextlib, io, json, os, shutil
import torch
from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.system import LMSystem, prepare_unpacked_AR_trunk_inputs
from core.training.optim import build_optimizers
from core.model import checkpoint_manager

def fresh(param_dtype=torch.float32):
    """A small System and its optimizers, built as build_system would (on the CPU, with no GPU at hand)."""
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=8, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, n_token_types=1)
    trunk, head = GPT(cfg), LMHead(cfg.n_embd, cfg.vocab_size)
    trunk.init_weights()
    head.init_weights()
    for p in list(trunk.parameters()) + list(head.parameters()):
        p.data = p.data.to(torch.bfloat16).to(param_dtype)          # build_system's param_dtype
    system = LMSystem(trunk, head, prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)
    system.arch = "GPT"                                             # build_system records the trunk class
    with contextlib.redirect_stdout(io.StringIO()):                 # (build_optimizers prints its learning rates)
        optimizers = build_optimizers(system, {"type": "adamw"})
    return system, optimizers

run_dir = "/tmp/core_docs_example/run"
shutil.rmtree(run_dir, ignore_errors=True)
system, optimizers = fresh()
config = {"checkpoint": {"save_dir": run_dir, "save_every": 5, "keep_last_n": 2}}
for step in range(12):
    checkpoint_manager.save_checkpoint_if_needed(
        step, system, optimizers, config, rank=0, world_size=1,
        dataloader_state={"source_states": {"toy": {"next": step + 1}}},    # the batch's snapshot
        trainer_state={"smooth_train_loss": 4.1}, max_steps=12)
print(sorted(os.listdir(run_dir)))
```
```text output
✓ Checkpoint saved: /tmp/core_docs_example/run/step_000000
✓ Checkpoint saved: /tmp/core_docs_example/run/step_000005
✓ Checkpoint saved: /tmp/core_docs_example/run/step_000010
  Removing old checkpoint: /tmp/core_docs_example/run/step_000000
✓ Checkpoint saved: /tmp/core_docs_example/run/step_000011
  Removing old checkpoint: /tmp/core_docs_example/run/step_000005
['step_000010', 'step_000011']
```

On several GPUs every rank takes part: each writes its own shard of the tensors, and before that the data snapshots of
all ranks are gathered and merged into the run's (`merge_data_snapshots`; why, in [../data/data_pipeline.md](../data/data_pipeline.md)).
Rank 0 writes `meta.json` last. A `step_*` directory without `meta.json` is therefore a save that was interrupted, by
a preemption say: finding the newest checkpoint skips it with a warning, and if every `step_*` directory is
incomplete it raises rather than start the run afresh over them. That is a rule about which directory to resume
from, not a check that the files are intact (only the presence of `meta.json` is looked at), and not a promise to
keep the directory: the cleanup after a later save counts every `step_*` directory, complete or not, and may delete
an old incomplete one.

## What `meta.json` holds

```python run
print(json.dumps(checkpoint_manager.load_metadata(f"{run_dir}/step_000011"), indent=1))
```
```text output
{
 "step": 11,
 "trainer_state": {
  "smooth_train_loss": 4.1
 },
 "dataloader_state": {
  "source_states": {
   "toy": {
    "next": 12
   }
  }
 },
 "model_config": {
  "sequence_len": 8,
  "vocab_size": 64,
  "n_layer": 2,
  "n_head": 2,
  "n_kv_head": 2,
  "n_embd": 32,
  "n_token_types": 1
 },
 "model_arch": "GPT",
 "param_dtype": "float32"
}
```

| Key | Written by | |
|---|---|---|
| `step` | the save | the step just trained; a resume starts at step + 1 |
| `trainer_state` | the Trainer | its loss average (`smooth_train_loss`, `ema_factor`), `total_training_time`, and for the record the precision convention (`training_precision`) and the float32 matmul setting (`cuda_matmul_fp32_precision`) |
| `dataloader_state` | the save, from the loader | the run's data snapshot, merged over the ranks |
| `model_config` | the save, from `model.config` | the trunk's config, complete; `config_from_meta` rebuilds it |
| `model_arch` | the save, from `model.arch` | the trunk class `build_system` was given |
| `param_dtype` | the save, from the live parameters | their storage dtype (a list of names when mixed) |
| `param_dtype_history` | a resume that changed the storage | when, from which checkpoint, from what to what (absent if never) |

Everything in it goes through JSON and must come back unchanged, which is why the data sources' states keep to plain
JSON types (see [../data/data_pipeline.md](../data/data_pipeline.md), "Data sources").

## Starting from a checkpoint

Two config keys, read when the Trainer is built (`load_checkpoint_if_needed`):

- **`checkpoint.resume_from`** continues a run: the model weights, the optimizers' moments, step counts and
  hyperparameters, the step (the next one), the data position (when the loader has `set_state`) and the Trainer's
  own averages. Its value is `null` (train from scratch, the default), `auto`, or a path.
- **`checkpoint.init_model_from`** starts a new run from trained weights: the weights only; optimizers, step and data
  start fresh. It takes a path, not `auto`.

When both are set, `resume_from` wins; if it finds nothing to resume (`auto` in an empty save directory), the run
starts from `init_model_from`'s weights. So one command serves the first launch, which starts from pretrained
weights, and every relaunch after a preemption, which resumes.

How a value becomes a checkpoint (`resolve_checkpoint_path`; `$VARS` are expanded):

| `resume_from` | |
|---|---|
| `null` | no checkpoint: train from scratch |
| `auto` | the newest complete `step_*` in `checkpoint.save_dir`; none, or no such directory yet: train from scratch |
| a checkpoint directory (one with `meta.json`) | exactly that one |
| a run directory (one with `step_*` inside) | its newest complete `step_*` |
| a path that does not exist, a file, or a directory with neither | an error: a typo must not start a fresh run |

```python run
print(checkpoint_manager.resolve_checkpoint_path("auto", save_dir=run_dir))
os.makedirs(f"{run_dir}/step_000012")                       # a save that died before its meta.json
print(checkpoint_manager.resolve_checkpoint_path(run_dir))
```
```text output
🔄 Auto-resume: Found checkpoint in /tmp/core_docs_example/run
   Resuming from: step_000011
/tmp/core_docs_example/run/step_000011
  ⚠️  Skipping 1 incomplete checkpoint dir(s) without meta.json (an interrupted save?): ['/tmp/core_docs_example/run/step_000012']. Not deleted.
📁 Auto-detected latest checkpoint: /tmp/core_docs_example/run/step_000011
   (from directory: /tmp/core_docs_example/run, latest: step_000011)
/tmp/core_docs_example/run/step_000011
```

A resume refuses to build on a model that does not match. The first two checks run before anything is loaded; the
last two after the weights and the optimizers are in, and a refusal there stops the run without undoing the load:

- **The architecture**: `model_arch`, and the fields that shape the weights (`n_layer`, `n_embd`, `n_head`,
  `n_kv_head`, `vocab_size`, `n_token_types`), must equal the live model's. A different `sequence_len` is allowed and
  noted: the rotary table is computed, not stored.
- **The storage dtype**: weights and moments stored in another dtype than the live model's (bf16 against fp32
  storage) are refused, since DCP would convert them without a word. `checkpoint.allow_param_dtype_change: true`
  converts deliberately (bf16 to fp32 is exact, the other way rounds) and records the change in
  `param_dtype_history`, which every later checkpoint carries. Buffers are converted with a note either way.
- **The optimizer hyperparameters** must match unless `checkpoint.override_optimizer_hparams: true` adopts this
  launch's ([../training/training.md](../training/training.md), "Resuming").
- **The data sources**: the snapshot's source names must be the loader's.

What a resume does not restore: the random generators (the Trainer saves none; a data source that shuffles keeps its
own seed and position), and anything decided by this launch's code and config rather than stored. The learning-rate
schedule is rebuilt from this launch's config (`max_steps` and `optimizer.scheduler`), and the System is built with
this launch's `build_system` arguments: the architecture (the trunk class and the config fields above) and the
storage dtype are checked against the checkpoint, the other assembly choices (`head_ce`, `head_softcap`, ...) are
neither restored nor checked. A head with
a non-default `head_softcap`, for instance, gets it again only if the caller passes it again (to `load_system` too).

```python run
resumed, resumed_optimizers = fresh()
meta = checkpoint_manager.load_checkpoint_dcp(f"{run_dir}/step_000011", resumed, resumed_optimizers)
print("resumed at step", meta["step"] + 1)

bf16_model, bf16_optimizers = fresh(param_dtype=torch.bfloat16)
try:
    checkpoint_manager.load_checkpoint_dcp(f"{run_dir}/step_000011", bf16_model, bf16_optimizers)
except ValueError as e:
    print("ValueError:", str(e).split(";")[0].split(": ", 1)[1])
```
```text output
resumed at step 12
ValueError: stored in another dtype than this run's model (checkpoint float32 -> this model bfloat16 (rounded)
```

## Loading weights

- **`load_system(checkpoint_dir, ...)`** is the inference entry: it rebuilds the config from `meta.json`, assembles the
  System through `build_system`, loads the weights only and puts it in eval mode
  ([../training/training.md](../training/training.md)). It needs a GPU.
- **`load_model_only(checkpoint_dir, model)`** loads the weights into a model you built: what `init_model_from` does.
  It checks the architecture like a resume, but converts a different storage dtype, saying so: an initialization
  starts no optimizer state of the checkpoint's.
- **`config_from_meta(checkpoint_dir, config_cls, sequence_len=None)`** rebuilds the config a checkpoint was trained
  with (keys the class does not know are dropped; fields the record lacks take the class's defaults);
  `load_metadata` and `get_model_config` read `meta.json` without touching the tensors.

```python run
bf16_model, _ = fresh(param_dtype=torch.bfloat16)
checkpoint_manager.load_model_only(f"{run_dir}/step_000011", bf16_model)
print(bf16_model.trunk.transformer.wte.weight.dtype)
```
```text output
  note: converting on load, checkpoint float32 -> this model bfloat16 (rounded), 15 tensors
[Rank 0] Loaded model weights from /tmp/core_docs_example/run/step_000011
torch.bfloat16
```

## Your own policy

When to save and what to keep are the project's to change, without a core option: the Trainer reaches the policy
through module-level functions that it looks up each time it needs them, so an orchestrator replaces them, explicitly,
before building the Trainer. The ones to replace: `should_save_checkpoint(step, max_steps, save_every) -> bool`,
`cleanup_old_checkpoints(save_dir, keep_last_n, current_step)`, or the whole `save_checkpoint_if_needed`. Keeping
only the final checkpoint of a run, say:

```python run
original = checkpoint_manager.should_save_checkpoint
checkpoint_manager.should_save_checkpoint = lambda step, max_steps, save_every: step == max_steps - 1

shutil.rmtree(run_dir)
for step in range(12):
    checkpoint_manager.save_checkpoint_if_needed(step, system, optimizers, config, rank=0, world_size=1,
                                                 max_steps=12)
print(sorted(os.listdir(run_dir)))
checkpoint_manager.should_save_checkpoint = original        # (put back for the rest of this page)
```
```text output
✓ Checkpoint saved: /tmp/core_docs_example/run/step_000011
['step_000011']
```

A project that saves on its own instead (outside the Trainer) calls `save_checkpoint_dcp(checkpoint_dir, model,
optimizers, meta_data, rank, world_size, dataloader_state=...)`, with `meta_data` holding at least `step`. On several
GPUs it passes the run's snapshot, `merge_data_snapshots(loader, this_ranks_snapshot, world_size)`, which every rank
calls (it is a collective); with rank 0's own snapshot instead, the data resume is approximate or refused, depending
on the source ([../data/data_pipeline.md](../data/data_pipeline.md), "Resume").

## Old checkpoints

- **A data snapshot in the list format** (nanoinfra before 0.3.0) is refused by the loader with an error naming the
  converter: `core.compatibility.accept_list_snapshot(loader, source_order)` at load time, or
  `python -m core.compatibility.loader_snapshot <checkpoint_dir> <source names>` once
  ([../data/data_pipeline.md](../data/data_pipeline.md), "Resume").
- **A checkpoint without `model_config`** predates the self-description: the architecture checks are skipped, and
  `load_system` cannot rebuild it; build the config yourself and use `load_model_only`.
- **An optimizer group key the checkpoint lacks** (one this version of the optimizer builder or of torch adds) stops
  the load with DCP's "Missing key in checkpoint state_dict"; note that this `CheckpointException` derives from
  `BaseException`, not `Exception`. Keys only the checkpoint has are dropped.

## Reference

| | |
|---|---|
| `save_checkpoint_if_needed(step, model, optimizers, config, rank, world_size, dataloader_state=None, trainer_state=None, dataloader=None, *, max_steps)` | reads `config["checkpoint"]`: `save_dir`, `save_every`, `keep_last_n` |
| `save_checkpoint_dcp(checkpoint_dir, model, optimizers, meta_data, rank=0, world_size=1, dataloader_state=None, replace_existing=False)` | writes the shards, then `meta.json` (rank 0) |
| `should_save_checkpoint(step, max_steps, save_every)`, `cleanup_old_checkpoints(save_dir, keep_last_n, current_step)` | the default policy; `ValueError` for `keep_last_n` 0 or below -1 |
| `merge_data_snapshots(dataloader, snapshot, world_size)` | a collective on several ranks; returns the snapshot as it is when the loader cannot merge |
| `load_checkpoint_if_needed(checkpoint_config, model, optimizers, dataloader, rank, world_size)` | -> `(start_step, trainer_state)` |
| `resolve_checkpoint_path(resume_from, save_dir=None, rank=0)`, `find_latest_checkpoint(base_dir)` | `FileNotFoundError`, `NotADirectoryError`, `ValueError` (`auto` without `save_dir`) |
| `load_checkpoint_dcp(checkpoint_dir, model, optimizers, rank=0, world_size=1, allow_param_dtype_change=False)` | -> meta; `ValueError` for a mismatched architecture or storage, or no optimizers |
| `load_model_only(checkpoint_dir, model, rank=0, world_size=1)` | `ValueError` for a mismatched architecture |
| `load_metadata(checkpoint_dir)`, `get_model_config(checkpoint_dir)`, `config_from_meta(checkpoint_dir, config_cls, sequence_len=None)` | read `meta.json` only |
