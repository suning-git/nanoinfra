# Training

This directory turns a model and a data loader into a training run. `build_system` assembles the model (trunk plus
head) and places it on the GPUs; two functions compile its trunk; `build_optimizers` makes the optimizers; the
`Trainer` runs the loop: learning-rate schedule, gradient accumulation, clipping, logging, evaluation, checkpoints.
The training entry of a project, the *orchestrator*, calls them in order. What it trains on (the data, the objective,
the evaluators) is decided there, not here.

| Piece | Code |
|---|---|
| `build_system`, `load_system`, `compile_system_trunk`, `compile_blocks` | [model_setup.py](model_setup.py) |
| `Trainer`, `clip_gradients`, `eval_is_due` | [trainer.py](trainer.py) |
| `build_optimizers` | [optim.py](optim.py) |
| the optimizer hyperparameters on a resume | [optim_resume.py](optim_resume.py) |
| `build_lr_schedule` | [lr_schedulers.py](lr_schedulers.py) |
| the precision convention | [precision.py](precision.py) |
| `set_torch_global_defaults`, `process_identity` | [../utils.py](../utils.py) |
| replicated data parallel (`NanoDDP`) | [../parallel/nano_ddp.py](../parallel/nano_ddp.py) |
| the sample configs' shared base | [../configs/train_base.yaml](../configs/train_base.yaml) |

`build_system`, `load_system` and the `Trainer` need CUDA. On the CPU you can assemble a System by hand
(`LMSystem(GPT(config), LMHead(...))`) and use the optimizers, the schedules, the data pipeline and the evaluation;
the examples marked as running below do that.

## How an orchestrator puts a run together

The order of `modalities/text/train_text.py`, the reference entry:

```text
set_torch_global_defaults()                         process-wide settings, before anything touches CUDA
tokenizer, vocabulary layout                        the modality's; they give vocab_size and n_token_types
gpt_config = GPTConfig(...)                         the trunk's geometry
setup = build_system(GPT, gpt_config, ...)          the System (trunk + head) on this process's GPU, sharded or not
system = setup["system"]
compile_system_trunk(system) or compile_blocks(...) optional: one of the two, or neither
ddp = NanoDDP(...)                                  only for parallel="ddp" on several GPUs
loader = MixedDataLoader(...)                       ../data/data_pipeline.md
optimizers = build_optimizers(system, config["optimizer"])          config: the training config, a plain dict
evaluators = [...]                                  ../evaluation/evaluation.md
Trainer(system, optimizers, loader, config, rank, world_size, evaluators=evaluators, ddp=ddp).train()
```

The smallest complete run, on core's synthetic token stream (random tokens, so the loss stays near ln 1024):

```python gpu
from core.utils import set_torch_global_defaults
set_torch_global_defaults()                         # first: the allocator setting is read when CUDA starts

from core.data import synthetic_token_loader
from core.model.gpt import GPT, GPTConfig
from core.model.system import prepare_unpacked_AR_trunk_inputs
from core.training.model_setup import build_system, compile_system_trunk
from core.training.optim import build_optimizers
from core.training.trainer import Trainer

config = {
    "max_steps": 20, "sequence_len": 256, "device_batch_size": 8, "total_batch_size": 8 * 256 * 2,
    "optimizer": {"type": "adamw", "lr_max": 0.02, "max_grad_norm": 1.0,
                  "scheduler": {"type": "linear", "warmup_steps": 2, "warmdown_ratio": 0.2}},
    "logging": {"log_every": 5},
    "wandb": {"enabled": False},
}
gpt_config = GPTConfig(sequence_len=256, vocab_size=1024, n_layer=2, n_head=4, n_kv_head=4, n_embd=256,
                       n_token_types=1)
setup = build_system(GPT, gpt_config, prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)
system = setup["system"]
compile_system_trunk(system)
loader = synthetic_token_loader(B=8, T=256, device=setup["device"], vocab_size=1024)
trainer = Trainer(system, build_optimizers(system, config["optimizer"]), loader, config,
                  setup["rank"], setup["world_size"])
trainer.train()
```

On one RTX 5090 this takes about ten seconds, most of it the first step's compilation; its log is under "What it
logs" below.

## The process

**`set_torch_global_defaults()`** is the entry's first line. It sets two process-wide things and nothing else: the
CUDA caching allocator's expandable segments (left alone if `PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF` is set
in the environment, even to an empty value), and `torch.set_float32_matmul_precision("high")`, which lets float32
matmuls use TF32. The allocator reads its setting when CUDA initializes, so the call raises if CUDA is already up and it would have
to set the allocator itself.
core never calls it itself: importing or assembling must not change the process. An entry that wants another matmul
precision sets it after the call; seeds and deterministic mode are each run's own.

**With `torchrun` or without.** `torchrun` sets `RANK`, `LOCAL_RANK` and `WORLD_SIZE`; then the process is
distributed, even when it is the only one (`--nproc_per_node=1`), and `build_system` selects GPU `cuda:LOCAL_RANK`
and creates the NCCL process group. A process group is created once per process, so under `torchrun` one process
calls `build_system` or `load_system` once. A plain `python` run is not distributed: it uses the current CUDA device
and creates no process group. `process_identity()` answers from those variables
alone, before or after the process group exists; some but not all of them set is refused. `print0` prints on rank 0
only.

## `build_system`

```python
setup = build_system(trunk_cls, config, head_softcap=15.0, seed=42, parallel="fsdp", head_ce="naive",
                     prepare_trunk_inputs_fn=None, param_dtype=torch.float32)
# -> {"system": LMSystem, "device": torch.device, "rank": int, "world_size": int}
```

Which model is the orchestrator's choice (`trunk_cls` and its `config`); how to assemble it is this function's.
In order:

```text
torch.manual_seed(seed)                             every rank draws the same initial weights
init_distributed()                                  under torchrun: GPU cuda:LOCAL_RANK, NCCL process group
trunk = trunk_cls(config) on the meta device, moved to the GPU uninitialized, trunk.init_weights()
head = LMHead(config.n_embd, config.vocab_size), init_weights() zeroes it
head_ce="liger": the head's class becomes LigerLMHead                     before sharding
every parameter rounded to bf16, then stored in param_dtype               buffers keep their own dtype
placement: plain python                    -> nothing to do
           under torchrun, parallel="fsdp" -> each block, then the trunk, then the head, sharded (bf16 gather and
                                              reduce); with one process too
           under torchrun, parallel="ddp"  -> nothing: every rank holds the whole model; gradient sync is NanoDDP's
system = LMSystem(trunk, head, prepare_trunk_inputs_fn=...); system.arch = trunk_cls.__name__
head_ce="compiled": head.loss and head.loss_per_token compiled
```

The trunk is never compiled here (next section). The arguments:

- **`prepare_trunk_inputs_fn`**: how a batch becomes the trunk's arguments, `batch -> (idx, kwargs)`: core's
  `prepare_unpacked_AR_trunk_inputs` (one sample per row), `prepare_packed_AR_trunk_inputs` (several), or a
  project's own. Training and evaluation both go through it; a System without one raises at its first batch. Pure
  inference, which never translates a batch, may leave it out.
- **`head_ce`**: which cross-entropy implementation the head uses. `"naive"` (the default) is `F.cross_entropy`
  over full fp32 logits: it needs no optional package and is the numeric reference for the other two. `"liger"` is
  Liger's fused linear + CE in token chunks, the lowest peak memory; it raises if `liger_kernel` is not installed
  rather than fall back. `"compiled"` is the naive math compiled into one graph: the fastest at every depth measured
  (one RTX 5090, vocabulary 32768, sequence length 1024, batch 32: MFU 33.9 / 64.3 / 70.0 % at depth 2 / 6 / 8,
  against 6.6 / 26.5 / 37.5 % naive),
  because the head is a large part of a small model's step.
- **`param_dtype`**: what the parameters, and so their gradients and AdamW moments, are stored in. `torch.float32`
  (the default) starts from the same values as bf16 storage, stored wider. For the reference GPT under bf16 autocast
  the arithmetic is unchanged: from the same weights the loss and every matrix gradient are bit for bit those of
  bf16 storage (the one exception is the head's weight gradient under `head_ce="liger"`); what differs is what is
  stored, so an update smaller than half a bf16 step of its weight is no longer rounded away. `torch.bfloat16` is
  for comparing with runs that were trained in bf16 storage.
- **`seed`** resets torch's default generators, the CPU's and every GPU's, before the trunk is built; Python's and
  NumPy's are not touched. `load_system` goes through here too, so loading a checkpoint resets them to 42: sample
  with a `torch.Generator` of your own.
- **`parallel`** picks between the two strategies of a `torchrun` launch and is ignored by a plain `python` run;
  whether the process is distributed is the launcher's decision, not this argument's (see "Several GPUs").

**A trunk of your own** works if it keeps the trunk contract that `GPT` implements:

- `__init__(config)`, keeping the config as `self.config` (the System's `config`, the optimizer builder and the
  checkpoints read it there);
- `init_weights()`, which must initialize every parameter and rebuild every buffer: the trunk is built on the meta
  device and moved to the GPU uninitialized, so no default initialization ran (a LayerNorm must set its own weight to
  1 and bias to 0; GPT recomputes its rotary table here);
- `forward(idx, ...) -> hidden [B, T, H]`;
- a `blocks` property, the per-layer modules (one FSDP shard group and one NanoDDP bucket each);
- `estimate_flops()`, the FLOPs per position;
- a class attribute `Config`, the config dataclass, which must carry `n_embd` and `vocab_size` (they size the head)
  and `n_layer` (printed while assembling). The GPT and the System side are in
[../model/model.md](../model/model.md).

Every parameter of the trunk is weight-decayed unless one of its modules declares it exempt in
`no_weight_decay_names` ("Weight decay exemptions", below): a gain whose neutral value is 1, a log-parametrized rate.

## Compiling the trunk

`build_system` does not compile. After it, the orchestrator calls one of two functions, or neither. Both compile in
place, so `system.trunk` stays the registered module (training, evaluation and inference all run the compiled code,
and the checkpoint keys stay `trunk.*`), and a trunk takes only one of them: either order raises.

| | |
|---|---|
| `compile_system_trunk(system, **compile_kwargs)` | the whole trunk (`nn.Module.compile`); the keyword arguments go to `torch.compile` (the text entry passes `dynamic=True`) |
| `compile_blocks(trunk, dynamic=None)` | each block's forward separately; for `parallel="ddp"` on several GPUs |

Why per block under NanoDDP: compiled as one graph, a dense trunk gets every parameter's gradient at the very end of
the backward, so NanoDDP has nothing to send while the backward still runs; compiled per block, each block's
gradients are ready as its backward ends and their all-reduce overlaps the next block's. A trunk that already breaks
into several graphs has these seams anyway (`torch._dynamo.explain(trunk.forward)` shows it). The text entry:

```python
if world_size > 1 and parallel == "ddp":
    compile_blocks(system.trunk)
else:
    compile_system_trunk(system, dynamic=True)
```

Training on packed rows (a BlockMask as the trunk's `attention`) requires one of the two: the trunk refuses an
uncompiled flex attention call while gradients are recorded. To run a compiled trunk eagerly for a moment, for
debugging, use `torch.compiler.set_stance("force_eager")`.

## `load_system`

The inference entry: a runnable System straight from a checkpoint.

```python
setup = load_system(checkpoint_dir, trunk_cls=GPT, sequence_len=None, head_softcap=15.0, head_ce="naive",
                    prepare_trunk_inputs_fn=None, param_dtype=torch.bfloat16)
# -> build_system's dict plus "gpt_config" (the config built) and "meta" (the checkpoint's meta.json)
```

It rebuilds the trunk's config from the checkpoint's `meta.json`, assembles through `build_system`, loads the
weights only, and puts the System in eval mode. `trunk_cls` must match the architecture the checkpoint records
(`model_arch`), or it raises instead of building the wrong model. `sequence_len` may differ from training (the
rotary table is computed, not stored). The weights load into bf16 by default whatever the training storage; for the
reference GPT under bf16 autocast that changes no result. The trunk comes back uncompiled. A checkpoint without a
recorded config predates them: build the config yourself and load the weights with
`core.model.checkpoint_manager.load_model_only`.

## The `Trainer`

```python
trainer = Trainer(system, optimizers, dataloader, config, rank, world_size, evaluators=None, ddp=None)
trainer.train()
```

`system` is an `nn.Module` with `loss(batch) -> scalar` and `estimate_flops()` (the FLOPs per position, for MFU;
`LMSystem` has both); `optimizers` a list; `dataloader` any
iterable of batches; `config` a plain dict (an orchestrator resolves its Hydra or YAML config to one first). When it
is built, the Trainer works out the gradient accumulation and the run length, builds the learning-rate schedule and,
if the config asks for it, loads a checkpoint (resume or weights-only start). `train()` runs from the step it starts
at to the end of the launch.

### The keys it reads

| Key | Meaning | Default |
|---|---|---|
| `max_steps` | optimizer steps in the run; -1: `target_param_data_ratio` × the System's parameters / `total_batch_size` | required |
| `target_param_data_ratio` | tokens per parameter for `max_steps: -1` | 20 |
| `num_iterations` | stop this launch after that many steps from where it started; the learning rate still follows `max_steps`; -1: run to `max_steps` | -1 |
| `sequence_len` | T, positions per row | required |
| `device_batch_size` | B, rows per micro-batch on each GPU | required |
| `total_batch_size` | positions per optimizer step, over all GPUs | required |
| `optimizer.max_grad_norm` | the gradient-norm clipping threshold | required |
| `optimizer.scheduler` | the learning-rate schedule (below) | linear, no warmup, warmdown 0.2, to 0 |
| `abort_on_nonfinite` | stop when the gradient norm is nan or inf (below) | true |
| `logging.log_every`, `logging.wandb_log_every` | console line and metrics-sink cadence, in steps | 10, 100 |
| `checkpoint.*` | saving, resuming, weights-only start (`enabled` defaults to false here; train_base.yaml sets true) | [../model/checkpoints.md](../model/checkpoints.md) |
| `profiling.*` | a PyTorch profiler trace of steps `start_step` to `end_step` | off |
| `wandb.*` | `enabled`, `project`, `name`; `WANDB_PROJECT` and `WANDB_NAME` in the environment take precedence | enabled |

The other `optimizer` keys are `build_optimizers`'s. Evaluation has no key here (see
[../evaluation/evaluation.md](../evaluation/evaluation.md)). [train_base.yaml](../configs/train_base.yaml) is the
sample configs' shared base, inherited through Hydra; it carries the samples' choices, not the Trainer's defaults,
and an orchestrator may inherit it, copy it or write its own.

### One step

`total_batch_size` fixes how many micro-batches make one optimizer step: K = `total_batch_size` /
(`device_batch_size` × `sequence_len` × world size), which must come out whole. With `total_batch_size: 524288`,
`device_batch_size: 16`, `sequence_len: 2048` and two GPUs, K = 524288 / 65536 = 8. Step s:

```text
lr = initial_lr x schedule(s), in every parameter group
K times: batch = next(dataloader)
         loss = system.loss(batch), under bf16 autocast
         (loss / K).backward()                      gradients add up; with NanoDDP only the K-th backward syncs them
clip the gradient norm to optimizer.max_grad_norm  stop here if it is nan or inf
every optimizer steps; gradients set to None
console log, metrics sink, evaluation, checkpoint  each when due
```

The batches must be `device_batch_size` rows of `sequence_len` positions, already on the GPU: the Trainer moves
nothing, and the token counts and K assume that shape (the orchestrator builds its batcher from the same two
numbers). The Trainer reads one key of a batch itself,
`state_dict` (the data position to save in a checkpoint); the whole batch goes to `system.loss`, so its other fields
are between the loader and the System. A loader that runs out ends training with its `StopIteration`.

**A non-finite gradient** stops the run at the clipping, before the optimizer writes it into the weights, with the
step in the error; `.grad` still holds the raw gradients for a post-mortem. There is no loss scaling under bf16, so a
nan or inf gradient is a bug, never a routine skip. `abort_on_nonfinite: false` trains on through it, as old runs
did.

### What it logs

On rank 0, first a summary of the configuration (steps, batch shape, accumulation, learning rates, GPU), then a line
every `log_every` steps and at the last step. The smallest run above printed:

```text
Step 00000/00020 (  0.0%) | loss: 6.931472 | grad_norm: 0.2502 | lrm: 0.50 | dt: 4939ms | tok/s: 829 | mfu: 0.00% | total time: 0.08m
Step 00005/00020 ( 25.0%) | loss: 7.016403 | grad_norm: 0.4600 | lrm: 1.00 | dt: 17ms | tok/s: 239,641 | mfu: 1.44% | total time: 0.08m
Step 00010/00020 ( 50.0%) | loss: 7.039424 | grad_norm: 0.3339 | lrm: 1.00 | dt: 17ms | tok/s: 239,337 | mfu: 1.44% | total time: 0.09m
Step 00015/00020 ( 75.0%) | loss: 7.025845 | grad_norm: 0.3293 | lrm: 1.00 | dt: 5ms | tok/s: 815,525 | mfu: 4.90% | total time: 0.09m
Step 00019/00020 ( 95.0%) | loss: 7.017576 | grad_norm: 0.3218 | lrm: 0.25 | dt: 5ms | tok/s: 872,340 | mfu: 5.24% | total time: 0.09m
```

- `loss` is a moving average of the step losses (weight 0.9 on the past, corrected for its start), not this step's
  loss. A step's loss is rank 0's own: the mean of its K micro-batch losses, not reduced over the ranks; and since
  `LMSystem.loss` is each micro-batch's mean over its targets, micro-batches with different numbers of targets
  count equally, not by targets. Here it starts at exactly ln 1024 = 6.931472, since the head starts at zero, and stays near ln 1024 because
  random tokens cannot be learned.
- `lrm` is the schedule's multiplier for the step: 0.5 in the first of two warmup steps; 0.25 at step 19, the last
  update of a warmdown that would reach 0 at step 20.
- `dt` is the step's wall time; step 0's includes the compilation.
- `tok/s` and `mfu` count positions, padding included, and the FLOPs are the System's per-position estimate times
  that count. That is right for one sample per row under dense causal attention; with padding, or with several
  samples per row each attending only to itself (packing), MFU reads high. An unknown GPU prints `mfu: n/a`.

Every `wandb_log_every` steps (and at the last) the same numbers, the peak GPU memory and the token and FLOP counts
go to the metrics sink: wandb by default, or a no-op when `wandb.enabled` is false or wandb is not installed. The sink
is whatever `Trainer._init_metrics()` returns; a subclass can return any object with `.log(dict)` and `.finish()`, a
JSONL writer say.

## `build_optimizers`

```python
optimizers = build_optimizers(system, optimizer_config)     # -> [AdamW(head, embeddings), AdamW(matrices)]
```

Three groups by the role of the parameters, each with its own learning rate, all scaled by (n_embd / 768)^-0.5 (the
rates were tuned on a 768-wide model):

| Group | Parameters | Learning rate (key, default) |
|---|---|---|
| unembedding | the head | `unembedding_lr`, 0.004 |
| embedding | every `nn.Embedding` in the trunk (token and type embeddings) | `embedding_lr`, 0.2 |
| matrix | every other trunk parameter | `lr_max`, 3e-4 |

`weight_decay` (default 0.01; a module can exempt its parameters, below) and `betas` (default [0.9, 0.95]) are the
same for all three; eps (1e-10) and the fused kernel are fixed. Each group's scaled rate is stored as its `initial_lr`, which the schedule multiplies. Only
`type: adamw` exists. For a two-layer, 32-wide GPT the scale is (32 / 768)^-0.5 ≈ 4.9:

```python run
import torch
from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.system import LMSystem, prepare_unpacked_AR_trunk_inputs
from core.training.optim import build_optimizers

cfg = GPTConfig(sequence_len=8, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, n_token_types=2)
trunk, head = GPT(cfg), LMHead(cfg.n_embd, cfg.vocab_size)
trunk.init_weights()
head.init_weights()
system = LMSystem(trunk, head, prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)

optimizers = build_optimizers(system, {"type": "adamw", "lr_max": 0.02})
names = [["unembedding (head)", "embedding"], ["matrix"]]
for i, opt in enumerate(optimizers):
    for j, group in enumerate(opt.param_groups):
        print(f"optimizer {i}, {names[i][j]:<18}: {len(group['params']):2d} tensors, initial_lr {group['initial_lr']:.6f}")
```
```text output
Scaling the LR for the AdamW parameters ∝1/√(32/768) = 4.898979
Using AdamW optimizer for matrix parameters (lr=0.02, betas=(0.9, 0.95), wd=0.01)
optimizer 0, unembedding (head):  1 tensors, initial_lr 0.019596
optimizer 0, embedding         :  2 tensors, initial_lr 0.979796
optimizer 1, matrix            : 12 tensors, initial_lr 0.097980
```

The groups come from module types, not from GPT's attribute names, so any trunk gets them. A project that wants
other groups or another optimizer builds its own list: the Trainer steps every optimizer it is given, and each
parameter group needs an `initial_lr`, which the schedule multiplies.

### Weight decay exemptions

Weight decay pulls a parameter toward 0. That is right only where 0 is the parameter's neutral value, and which
value is neutral depends on how its designer wrote it: a gain applied as `g * x` has 1 (decay would shrink the
layer), one written `(1 + w) * x` has 0 (decay is fine); Gated DeltaNet's `A_log` and `dt_bias`, which set how fast a
head forgets, have none at 0. So core infers nothing from shapes, names or module types: every parameter is decayed
unless the module that registers it names it in `no_weight_decay_names`, a tuple of its own parameters' local names:

```python
class GatedDeltaNet(nn.Module):
    no_weight_decay_names = ("A_log", "dt_bias")                     # a class attribute ...

self.ln_1 = nn.LayerNorm(n_embd)
self.ln_1.no_weight_decay_names = ("weight",)                        # ... or on an instance (it replaces the class's);
                                                                     # the bias's neutral value is 0: decayed
```

In each role the exempt parameters form a group of their own right after it, at the same learning rate and
`weight_decay` 0; with nothing declared the groups are the three above (the GPT declares nothing). A name that is
not one of the module's own parameters raises (a typo, or a parameter renamed without its declaration, would
otherwise keep its decay silently). The declaration is on the module, not on the parameter object, because
`build_system`'s initialization on the meta device, FSDP and `deepcopy` replace the parameter objects and keep the
modules and names; it is model code: not saved in a checkpoint, and a resume does not compare it with one.
`no_weight_decay_params(system)` lists the declared parameters, for a project that builds its own optimizers. A
designer who keeps the token embedding undecayed, on the same two-layer GPT:

```python run
ex_trunk, ex_head = GPT(cfg), LMHead(cfg.n_embd, cfg.vocab_size)
ex_trunk.transformer.wte.no_weight_decay_names = ("weight",)
ex_optimizers = build_optimizers(LMSystem(ex_trunk, ex_head), {"type": "adamw", "lr_max": 0.02})
for i, opt in enumerate(ex_optimizers):
    for group in opt.param_groups:
        print(f"optimizer {i}: {len(group['params']):2d} tensors, initial_lr {group['initial_lr']:.6f}, "
              f"weight_decay {group['weight_decay']}")
```
```text output
Scaling the LR for the AdamW parameters ∝1/√(32/768) = 4.898979
Using AdamW optimizer for matrix parameters (lr=0.02, betas=(0.9, 0.95), wd=0.01)
Weight decay exempt for 1 of the parameters, as their modules declare (no_weight_decay_names)
optimizer 0:  1 tensors, initial_lr 0.019596, weight_decay 0.01
optimizer 0:  1 tensors, initial_lr 0.979796, weight_decay 0.01
optimizer 0:  1 tensors, initial_lr 0.979796, weight_decay 0.0
optimizer 1: 12 tensors, initial_lr 0.097980, weight_decay 0.01
```

The embedding role keeps the type embedding in its group and gets a second group, the token embedding at
`weight_decay` 0.

## Learning-rate schedules

`optimizer.scheduler` describes the schedule; each step's rate is the group's `initial_lr` times the multiplier.

| Key | Meaning | Default |
|---|---|---|
| `type` | `linear`: warmup, constant, linear warmdown; `cosine`: warmup, cosine decay | `linear` |
| `warmup_steps` | warmup length in steps, an absolute number | 0 |
| `warmdown_ratio` | `linear` only: the warmdown's share of `max_steps` | 0.2 |
| `final_lr_frac` | where the decay heads, as a fraction of `initial_lr` | 0.0 |

The decay reaches `final_lr_frac` at step `max_steps`, one past the last update, so the last update still uses a
little more. A linear schedule whose warmdown rounds to zero steps (`round(warmdown_ratio × max_steps)`, as with
`warmdown_ratio: 0`) stays at 1 after the warmup, and its `final_lr_frac` has no effect. Warmup is absolute because it serves the early optimization whatever the run's length; warmdown is a
fraction because it is anchored to the end of the run, whose length may be computed (`max_steps: -1`). A recipe that
thinks of warmup as a fraction converts it in the orchestrator (`warmup_ratio` is refused). Ten steps, two of
warmup:

```python run
from core.training.lr_schedulers import build_lr_schedule

linear = build_lr_schedule({"type": "linear", "warmup_steps": 2, "warmdown_ratio": 0.4}, max_steps=10)
cosine = build_lr_schedule({"type": "cosine", "warmup_steps": 2, "final_lr_frac": 0.1}, max_steps=10)
print([round(linear(s), 2) for s in range(10)])
print([round(cosine(s), 2) for s in range(10)])
```
```text output
[0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25]
[0.5, 1.0, 1.0, 0.97, 0.87, 0.72, 0.55, 0.38, 0.23, 0.13]
```

All the rules (warmup and warmdown fit in `max_steps`, the ranges of the numbers) are checked when the Trainer is
built, before any checkpoint loads. Another schedule: replace the builder in the orchestrator before building the
Trainer, which looks it up on the module at that moment:

```python
from core.training import lr_schedulers
lr_schedulers.build_lr_schedule = my_builder     # (scheduler_config, max_steps) -> callable(step) -> multiplier
```

## Precision

Three things decide how training computes, and each has one owner:

- **storage**, what the parameters (and so their gradients and AdamW moments) are kept in: `build_system`'s
  `param_dtype`, fp32 by default (the sample configs' key `param_dtype: fp32 | bf16`);
- **execution**, how the training step computes: [precision.py](precision.py). The forward and the loss run under
  CUDA bf16 autocast, with no gradient scaler (bf16 has fp32's exponent range). Under FSDP the parameters are
  gathered in bf16 and each gradient is reduce-scattered in bf16, then accumulated in the storage dtype; FSDP also
  casts the floating-point inputs of every sharded module to bf16. Under NanoDDP the dtype the gradients are
  all-reduced in is NanoDDP's `reduce_dtype`, the orchestrator's choice: by default each gradient in its own dtype,
  the storage dtype (see "Several GPUs" below);
- **the model's own numerics**, such as GPT's bf16 rotary table or the head's fp32 logits, which neither of the other
  two touches.

Every checkpoint records the execution convention (`trainer_state.training_precision`, `"bf16_autocast_v1"`) and the
float32 matmul precision setting at the save (`trainer_state.cuda_matmul_fp32_precision`); a run synchronized by
NanoDDP also records its reduce dtype (`trainer_state.nano_ddp_reduce_dtype`: `null` for each gradient in its own
dtype, else `"bfloat16"` or `"float32"`). They are records: a resume does not read them back.

## Resuming

With `checkpoint.resume_from` set, the Trainer restores the model, the optimizers, the step and its own running
averages, and `train()` goes on from the next step. The data position comes back when the checkpoint holds a data
snapshot and the loader has `set_state()`; otherwise the data starts from its beginning, with a warning (the
synthetic loader of the smallest example has no `set_state`); what is saved and how a path is
found is in [../model/checkpoints.md](../model/checkpoints.md). For the optimizer hyperparameters it takes care of one
thing itself. Before loading, it copies the parameter groups this launch built; after loading, it compares the
adjustable ones, `initial_lr` (after the width scaling), `weight_decay`, `betas` and `eps`:

- `checkpoint.override_optimizer_hparams: false` (the default): a difference is an error that names every
  differing value. A resume does not change a run's hyperparameters by accident.
- `true`: this launch's values are taken, all of them, while the weights, the moments and the step counts stay the
  checkpoint's.

Any other group option (`fused`, `amsgrad`, ...) must match either way. The schedule follows the `max_steps` of this
launch.

## Several GPUs

Launch with `torchrun --nproc_per_node=N ...`; every process runs the same entry. `build_system(parallel=...)` picks
the strategy:

- **`"fsdp"`** (the default) shards the parameters, block by block. Use it when the model does not fit on one GPU.
- **`"ddp"`** keeps a whole copy on every GPU and leaves the gradient synchronization to a `NanoDDP` the orchestrator
  builds and hands to the Trainer (`ddp=`). Where the model fits, it can be faster and smaller: the memory
  bottleneck is then the activations, and sharding costs an all-gather per block per step. Measured on two RTX 5090s
  with a 233M-parameter model: 106.2k tok/s and 19 GB replicated against 93.6k tok/s and 30 GB sharded; measure your
  own configuration before choosing.

NanoDDP all-reduces the gradients in buckets during the backward, each bucket as soon as its last gradient lands, so
the communication overlaps the computation. The buckets are issued in list order, which must be the order in which
their gradients complete:

```text
the backward produces gradients:   head  ->  block 11  ->  block 10  -> ... ->  block 0  ->  embeddings
NanoDDP issues the all-reduces:   [head]   [block 11]    [block 10]    ...    [block 0]    [embeddings and the rest]
```

```python
ddp = NanoDDP([[p for p in system.head.parameters() if p.requires_grad]] + block_buckets(system.trunk),
              module=system)
```

Each gradient goes on the wire in its own dtype unless the orchestrator passes `reduce_dtype`; a bucket mixing
dtypes (some parameters kept in fp32, the rest in bf16) is then reduced as one all-reduce per dtype.
`reduce_dtype=torch.bfloat16` converts every gradient to bf16 for the all-reduce and back into its `.grad`
afterwards, which halves the traffic of fp32 storage and pays across nodes; the text entry reads it from
`ddp_reduce_dtype` (`null`, `bf16` or `fp32`).

Only the K-th micro-batch's backward synchronizes; the Trainer brackets each backward with `sync_gradients(ddp,
enabled=...)`, which does nothing when `ddp` is None. Compile per block (above). What NanoDDP deliberately does not
do, and the measurements behind its design, are in the [module docstring](../parallel/nano_ddp.py). How each rank
reads its share of the data, and of the evaluation data, is in [../data/data_pipeline.md](../data/data_pipeline.md)
and [../evaluation/evaluation.md](../evaluation/evaluation.md).

## Extending the training

- **Another objective** needs no Trainer change: a System whose `loss(batch)` computes it ([../model/model.md](../model/model.md)).
- **Another step or another metrics sink**: subclass the Trainer and override `_training_step(data_iter, step)`
  (returns the step's loss, gradient norm and data snapshot) or `_init_metrics()`.
- **Another schedule or checkpoint policy**: replace the module-level function the Trainer looks up when it needs
  it, explicitly, in the orchestrator before building the Trainer: `lr_schedulers.build_lr_schedule` above; the
  checkpoint functions in [../model/checkpoints.md](../model/checkpoints.md).

## Reference

| | |
|---|---|
| `build_system(trunk_cls, config, head_softcap=15.0, seed=42, parallel="fsdp", head_ce="naive", prepare_trunk_inputs_fn=None, param_dtype=torch.float32)` | -> `{"system", "device", "rank", "world_size"}`; raises `ValueError` for an unknown `parallel`, `head_ce` or `param_dtype`, `RuntimeError` for `head_ce="liger"` without `liger_kernel` |
| `load_system(checkpoint_dir, trunk_cls=GPT, sequence_len=None, head_softcap=15.0, head_ce="naive", prepare_trunk_inputs_fn=None, param_dtype=torch.bfloat16)` | -> `build_system`'s dict + `"gpt_config"`, `"meta"`; raises `ValueError` when `trunk_cls` does not match the recorded `model_arch` or the checkpoint has no recorded config |
| `compile_system_trunk(system, **compile_kwargs)`, `compile_blocks(trunk, dynamic=None)` | compile in place; `RuntimeError` when the trunk is already compiled by either |
| `Trainer(system, optimizers, dataloader, config, rank, world_size, evaluators=None, ddp=None)` | `.train()`; attributes `max_steps` (resolved), `start_step`, `grad_accum_steps`, `autocast_ctx` |
| `build_optimizers(system, optimizer_config)` | -> two AdamW; asserts `optimizer_config["type"] == "adamw"` |
| `build_lr_schedule(scheduler_config, max_steps)` | -> a callable `step -> multiplier`; `ValueError` for an unknown type, `warmup_ratio`, or a value out of range |
| `set_torch_global_defaults()` | `RuntimeError` when CUDA is already initialized and the allocator is not configured in the environment |
| `process_identity()` | -> `ProcessIdentity(distributed, rank, local_rank, world_size)`; `RuntimeError` for a partial launcher environment |
