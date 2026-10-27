# core

nanoinfra's training infrastructure, the same for every modality: the data pipeline, the model (a GPT trunk and a
head, held together as a System), the training loop, evaluation, checkpoints and inference. What makes a run about
text, motion or video lives outside core: tokenizers, vocabulary bands, data sources and recipes in `modalities/`,
and the training entries (*orchestrators*) that wire everything in `exemplars/` and `projects/`. core provides parts
that an orchestrator calls explicitly; it reads no global config and registers nothing.

```text
data sources -> SampleMixer -> sample function -> batcher -> MixedDataLoader -> batches   data/data_pipeline.md
                                                                                  |
                       System: prepare_trunk_inputs -> GPT trunk -> head -> loss          model/model.md
                                                                                  |
          build_system, Trainer: steps, accumulation, clipping, logs, schedule           training/training.md
                         evaluators, at the steps they choose                             evaluation/evaluation.md
                                                                                  |
                     checkpoints: resume, start from weights, load                        model/checkpoints.md
                     generation through a KV cache                                        model/inference_engine.md
```

| Doc | What it covers |
|---|---|
| [data/data_pipeline.md](data/data_pipeline.md) | data sources, recipes and token types, supervision, mixing, padding and packing, the loader, resuming the data exactly |
| [model/model.md](model/model.md) | the System and its `loss`, from a batch to the trunk's inputs, visibility rules and BlockMasks, the GPT, positions, the head |
| [training/training.md](training/training.md) | assembling a run, `build_system` and compiling, the Trainer, optimizers, schedules, precision, several GPUs |
| [evaluation/evaluation.md](evaluation/evaluation.md) | when evaluation runs, `LossEvaluator`, metrics, bits per byte, writing an evaluator |
| [model/checkpoints.md](model/checkpoints.md) | what is saved and when, resuming, starting from weights, checks, your own save policy |
| [model/inference_engine.md](model/inference_engine.md) | `autoregressive_generate`, the two KV caches, visibility per forward, `rewind` |

## I want to ...

| ... | Where | core's part |
|---|---|---|
| put a training run together | [training.md](training/training.md), "How an orchestrator puts a run together" | `build_system`, `build_optimizers`, `Trainer` |
| train on my own data | [data_pipeline.md](data/data_pipeline.md), "Data sources" | `DataSource`; `ResumableDistributedSampler` for an indexable dataset |
| build sequences from fields, supervise only some segments | [data_pipeline.md](data/data_pipeline.md), "Sequences from fields" | `SequenceRecipe`, `VocabLayout` |
| mix several sources | [data_pipeline.md](data/data_pipeline.md), "Mixing" | `SampleMixer`, the loader's weights |
| put several samples in a row | [data_pipeline.md](data/data_pipeline.md), "Batching"; [model.md](model/model.md) | `PackingBatcher` + `prepare_packed_AR_trunk_inputs`, and a compiled trunk |
| train another objective | [model.md](model/model.md), "A System of your own" | subclass `LMSystem`, override `loss` |
| control which positions see which | [model.md](model/model.md), "Your own visibility rule" | `same_sample`, `causal`, `pad_to_flex_block`, a BlockMask |
| use my own positions (3-D RoPE, ...) | [model.md](model/model.md), "Positions" | `set_rotary_table`, `position_ids` |
| use another trunk architecture | [training.md](training/training.md), "A trunk of your own" | the trunk contract |
| choose the head's CE implementation | [training.md](training/training.md), `build_system` | `head_ce`: naive, liger, compiled |
| store parameters in fp32 or bf16 | [training.md](training/training.md), "Precision" | `param_dtype` |
| train on several GPUs | [training.md](training/training.md), "Several GPUs" | `parallel="fsdp"`, or `"ddp"` with `NanoDDP` |
| change the optimizer groups or the schedule | [training.md](training/training.md) | `build_optimizers`, `optimizer.scheduler`, replacing `build_lr_schedule` |
| evaluate during training, with my own metric | [evaluation.md](evaluation/evaluation.md) | `LossEvaluator`, `metric(batch, ce) -> (numerator, denominator)` |
| survive preemption | [checkpoints.md](model/checkpoints.md), "When the Trainer saves", "Starting from a checkpoint" | `checkpoint.enabled: true` with a `save_dir` and `save_every`, and `checkpoint.resume_from: auto` |
| start from pretrained weights | [checkpoints.md](model/checkpoints.md) | `checkpoint.init_model_from` |
| keep other checkpoints | [checkpoints.md](model/checkpoints.md), "Your own policy" | replace `should_save_checkpoint` or `cleanup_old_checkpoints` |
| log somewhere other than wandb | [training.md](training/training.md), "What it logs" | override `Trainer._init_metrics` |
| load a model and generate | [training.md](training/training.md), `load_system`; [inference_engine.md](model/inference_engine.md) | `load_system`, `autoregressive_generate` |
| write a block-diffusion or other custom inference loop | [inference_engine.md](model/inference_engine.md) | `KVCache` / `StaticKVCache`, `attention="bidirectional"`, `rewind` |

## Examples to copy

| Entry | What it shows |
|---|---|
| [modalities/text/train_text.py](../modalities/text/train_text.py) | the reference orchestrator: text pretraining, one window per row, FSDP or NanoDDP, compiling, evaluation streams |
| [exemplars/nano_motion/train_t2m.py](../exemplars/nano_motion/train_t2m.py) | text-to-motion: a recipe with supervised segments, padding or packing, an evaluator on frozen batches shared over the ranks |
| [exemplars/nano_world_model/train_wm.py](../exemplars/nano_world_model/train_wm.py) | block diffusion: a System with its own loss, a two-stream BlockMask, its own evaluator and sampler |
| [exemplars/text_pretrain/](../exemplars/text_pretrain/) | text recipes and a scaling study, with their results |
| [projects/](../projects/) | larger projects on core (`nano_multimodal`, `nano_dsv4`, ...) |

## Not in core: do not assume it

- **No global config schema.** The Trainer reads a plain dict, each part takes its own arguments, and
  `configs/train_base.yaml` is the sample configs' shared base, not a schema.
- **The Trainer moves no batch, evaluates nothing before the first update, and keeps no evaluator state** in
  checkpoints. Evaluation at step 0 is of the model after one update.
- **A resume is not a replay**: the random generators are not saved, and the schedule and the System's build
  arguments come from the relaunch's config ([checkpoints.md](model/checkpoints.md), "Starting from a checkpoint").
- **CUDA only** for `build_system`, `load_system` and the Trainer; bf16 autocast only, with no fp16 and no loss scaling.
- **Two parallel strategies**: FSDP and replicated data parallel (`NanoDDP`, verified on two GPUs). No tensor,
  pipeline or expert parallelism, and no activation checkpointing.
- **One optimizer builder**, AdamW in three groups; others are a project's own list of optimizers.
- **No tokenizer and no data sources** of a modality: they are in `modalities/`.
- **No selection of the best checkpoint and no early stopping** on a validation metric: a project writes them (an
  evaluator can track the best value; a save policy can act on it).
- **Generation takes prompts of one length**, without padding; block-diffusion and other inference loops are a
  project's, on core's caches.
