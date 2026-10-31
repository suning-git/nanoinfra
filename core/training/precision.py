"""The training execution convention: how core's Trainer and build_system compute, defined once.

Precision has three owners, and this file is one of them:
  * storage — what the parameters, and so their gradients and AdamW moments, are stored in: the entry chooses it
    (config key param_dtype) and build_system's `param_dtype` applies it, to the parameters only;
  * execution — this file: how the training step computes, whatever the storage;
  * a model's own numerics — GPT's bf16 rotary table, the head's fp32 logits, a router's fp32 load-balancing
    state — are the model's, and neither this file nor build_system touches them.

The convention, named NAME and recorded in every checkpoint the Trainer saves (meta.json
['trainer_state']['training_precision']; a checkpoint without it has no record, whatever ran it):
  * the forward and the loss run under CUDA bf16 autocast (training_autocast); the backward follows the forward's
    dtypes. No GradScaler: bf16 has fp32's exponent range.
  * under FSDP (build_system(parallel="fsdp")), parameters are gathered in bf16 for the forward and backward, and
    each gradient is reduce-scattered in bf16, then accumulated in the storage dtype (fsdp_mixed_precision). FSDP's
    default cast_forward_inputs=True also casts the floating-point inputs of every sharded module to bf16 (a block
    that takes its rotary table as an argument receives it in bf16).
  * under NanoDDP (parallel="ddp") the gradient all-reduce's dtype is not part of it: it is NanoDDP's reduce_dtype,
    which the orchestrator chooses (by default each gradient in the dtype it has, the storage dtype).

Not part of it, because it is not the training step's: the float32 matmul precision
(torch.set_float32_matmul_precision) acts on the whole process — training, evaluation, inference, any other model
in it — so the entry sets it (the training entries through core.utils.set_torch_global_defaults: "high"); the
Trainer records the setting at each save (['trainer_state']['cuda_matmul_fp32_precision']).

One convention, so nothing here is a parameter to pass around: when a second one is supported, it becomes a
choice, and NAME tells the checkpoints of the two apart. A checkpoint's full record of how the run computed is NAME
plus the fields the Trainer writes beside it: the float32 matmul setting, and under NanoDDP its reduce dtype
(['trainer_state']['nano_ddp_reduce_dtype']: null = each gradient in its own dtype, a name = that dtype; the key is
absent when no NanoDDP ran, and in checkpoints saved before it was recorded). These are records only: nothing reads
them back, and a resume runs with whatever the orchestrator passes then.
"""
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

NAME = "bf16_autocast_v1"


def training_autocast():
    """A new autocast context for one Trainer: the Trainer enters it around the forward and the loss, and hands the
    same object to its evaluators."""
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def fsdp_mixed_precision():
    """The FSDP policy build_system shards with."""
    return MixedPrecisionPolicy(param_dtype=torch.bfloat16,      # the dtype FSDP gathers parameters into
                                reduce_dtype=torch.bfloat16)     # the dtype of the gradient reduce-scatter
