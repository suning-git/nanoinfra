"""Parallelism strategies: HOW work is spread across devices.

Distinct from `core.training`, which is HOW a training run proceeds, and from
`core.training.model_setup`, which merely CALLS a strategy (torch's `fully_shard`,
or the replicas this package's NanoDDP synchronizes). What lives here is an
implementation of a strategy, in the sense that `torch.distributed.fsdp` is one.

    from core.parallel import NanoDDP, sync_gradients, block_buckets
"""

from core.parallel.nano_ddp import (
    NanoDDP,
    block_buckets,
    replica_divergence,
    sync_gradients,
)

__all__ = ["NanoDDP", "block_buckets", "replica_divergence", "sync_gradients"]
