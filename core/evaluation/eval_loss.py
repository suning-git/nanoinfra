"""
Evaluation service layer: stateless loss computation functions.

Two functions with identical interface, differing in how they compute loss:
- evaluate_loss_logits: head(hidden) forward + F.cross_entropy (materializes logits)
- evaluate_loss_fused: head.loss / head.type_losses (memory-safe)
"""

import math

import torch
import torch.nn.functional as F
import torch.distributed as dist

from core.tokenization.vocab_layout import VocabLayout


@torch.no_grad()
def evaluate_loss_logits(model, loader, steps, type_ids=None, token_bytes=None):
    """
    Evaluate loss via the head's forward (logits) + F.cross_entropy.

    Materializes logits [B, T, V]. Supports per-token analysis.

    Args:
        model: System (model.trunk(idx) returns hidden; model.head(hidden) returns logits)
        loader: Iterable yielding batch dicts with idx, targets, token_types, etc.
        steps: Number of batches to evaluate
        type_ids: Optional list of type IDs for per-type loss (requires target_types in batch)
        token_bytes: Optional [vocab_size] tensor for BPB computation

    Returns:
        dict: Always has "total_loss". Optionally "type_losses", "bpb".
    """
    device = next(model.parameters()).device

    total_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_count = torch.tensor(0, dtype=torch.int64, device=device)

    has_types = type_ids is not None
    if has_types:
        type_loss_sums = [torch.tensor(0.0, dtype=torch.float32, device=device) for _ in type_ids]
        type_counts = [torch.tensor(0, dtype=torch.int64, device=device) for _ in type_ids]

    has_bpb = token_bytes is not None
    if has_bpb:
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(loader)
    for _ in range(steps):
        batch = next(batch_iter)

        x = batch["idx"]
        y = batch["targets"]
        token_types = batch.get("token_types")

        # Forward (trunk) + head logits
        hidden = model.trunk(x, token_types=token_types)
        logits = model.head(hidden)

        # Per-token loss [B*T]
        loss_flat = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=VocabLayout.IGNORE_INDEX,
            reduction='none',
        )
        y_flat = y.reshape(-1)
        valid = (y_flat >= 0)

        # Total CE
        total_loss_sum += loss_flat[valid].sum()
        total_count += valid.sum()

        # Per-type CE
        if has_types:
            if 'target_types' not in batch:
                raise ValueError(
                    "type_ids provided but batch missing 'target_types'."
                )
            tt_flat = batch['target_types'].reshape(-1)
            for i, tid in enumerate(type_ids):
                type_valid = valid & (tt_flat == tid)
                type_loss_sums[i] += loss_flat[type_valid].sum()
                type_counts[i] += type_valid.sum()

        # BPB
        if has_bpb:
            y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
            num_bytes = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y_flat, dtype=token_bytes.dtype),
            )
            bpb_valid = valid & (num_bytes > 0)
            total_nats += loss_flat[bpb_valid].sum()
            total_bytes += num_bytes[bpb_valid].sum()

    # Distributed all_reduce
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        all_tensors = [total_loss_sum, total_count]
        if has_types:
            all_tensors.extend(type_loss_sums)
            all_tensors.extend(type_counts)
        if has_bpb:
            all_tensors.extend([total_nats, total_bytes])
        for t in all_tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # Build results
    results = {}
    n = total_count.item()
    results["total_loss"] = total_loss_sum.item() / n if n > 0 else float('inf')

    if has_types:
        type_losses = {}
        for i, tid in enumerate(type_ids):
            c = type_counts[i].item()
            type_losses[tid] = type_loss_sums[i].item() / c if c > 0 else float('inf')
        results["type_losses"] = type_losses

    if has_bpb:
        tb = total_bytes.item()
        results["bpb"] = total_nats.item() / (math.log(2) * tb) if tb > 0 else float('inf')

    return results


@torch.no_grad()
def evaluate_loss_fused(model, loader, steps, type_ids=None, token_bytes=None):
    """
    Evaluate loss via head_loss / head_type_losses (fused CE).

    Memory-safe: never materializes full [B, T, V] logits.

    Args:
        model: GPT model (forward returns hidden, has head_loss/head_type_losses)
        loader: Iterable yielding batch dicts with idx, targets, token_types, etc.
        steps: Number of batches to evaluate
        type_ids: Optional list of type IDs for per-type loss (requires target_types in batch)
        token_bytes: Optional [vocab_size] tensor for BPB computation

    Returns:
        dict: Always has "total_loss". Optionally "type_losses", "bpb".
    """
    device = next(model.parameters()).device

    total_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_count = torch.tensor(0, dtype=torch.int64, device=device)

    has_types = type_ids is not None
    if has_types:
        type_loss_sums = [torch.tensor(0.0, dtype=torch.float32, device=device) for _ in type_ids]
        type_counts = [torch.tensor(0, dtype=torch.int64, device=device) for _ in type_ids]

    has_bpb = token_bytes is not None
    if has_bpb:
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(loader)
    for _ in range(steps):
        batch = next(batch_iter)

        x = batch["idx"]
        y = batch["targets"]
        token_types = batch.get("token_types")

        # Forward (trunk) + fused head loss
        hidden = model.trunk(x, token_types=token_types)
        loss = model.head.loss(hidden, y)

        # Count valid tokens for this batch
        y_flat = y.reshape(-1)
        n_valid = (y_flat >= 0).sum()

        total_loss_sum += loss * n_valid
        total_count += n_valid

        # Per-type losses
        if has_types:
            if 'target_types' not in batch:
                raise ValueError(
                    "type_ids provided but batch missing 'target_types'."
                )
            type_losses_batch = model.head.type_losses(hidden, y, batch['target_types'], type_ids)
            tt_flat = batch['target_types'].reshape(-1)
            for i, tid in enumerate(type_ids):
                n_type = ((y_flat >= 0) & (tt_flat == tid)).sum()
                type_loss_sums[i] += type_losses_batch[tid] * n_type
                type_counts[i] += n_type

        # BPB
        if has_bpb:
            valid = (y_flat >= 0)
            y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
            num_bytes = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y_flat, dtype=token_bytes.dtype),
            )
            bpb_valid = valid & (num_bytes > 0)
            if n_valid > 0:
                total_nats += loss * bpb_valid.sum()
            total_bytes += num_bytes[bpb_valid].sum()

    # Distributed all_reduce
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        all_tensors = [total_loss_sum, total_count]
        if has_types:
            all_tensors.extend(type_loss_sums)
            all_tensors.extend(type_counts)
        if has_bpb:
            all_tensors.extend([total_nats, total_bytes])
        for t in all_tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # Build results
    results = {}
    n = total_count.item()
    results["total_loss"] = total_loss_sum.item() / n if n > 0 else float('inf')

    if has_types:
        type_losses = {}
        for i, tid in enumerate(type_ids):
            c = type_counts[i].item()
            type_losses[tid] = type_loss_sums[i].item() / c if c > 0 else float('inf')
        results["type_losses"] = type_losses

    if has_bpb:
        tb = total_bytes.item()
        results["bpb"] = total_nats.item() / (math.log(2) * tb) if tb > 0 else float('inf')

    return results
