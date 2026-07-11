"""Compatibility evaluation helper for per-token-loss mock models and GPT models."""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from core.tokenization.vocab_layout import VocabLayout


@torch.no_grad()
def evaluate_loss(model, loader, steps, type_ids=None, token_bytes=None):
    """Evaluate loss from either a GPT-style model or a per-token-loss mock model."""
    device = next(model.parameters()).device
    total_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_weight = torch.tensor(0.0, dtype=torch.float32, device=device)

    has_types = type_ids is not None
    if has_types:
        type_loss_sums = [torch.tensor(0.0, dtype=torch.float32, device=device) for _ in type_ids]
        type_weights = [torch.tensor(0.0, dtype=torch.float32, device=device) for _ in type_ids]

    has_bpb = token_bytes is not None
    if has_bpb:
        token_bytes = token_bytes.to(device)
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0.0, dtype=torch.float32, device=device)

    batch_iter = iter(loader)
    for _ in range(steps):
        batch = next(batch_iter)
        x = batch["idx"].to(device)
        y = batch["targets"].to(device)
        token_types = batch.get("token_types")
        if token_types is not None:
            token_types = token_types.to(device)

        if hasattr(model, "head_logits"):
            hidden = model(x, token_types=token_types)
            logits = model.head_logits(hidden)
            loss_flat = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=VocabLayout.IGNORE_INDEX,
                reduction="none",
            )
        else:
            per_token_loss = model(x, token_types=token_types, targets=y, loss_reduction="none")
            loss_flat = per_token_loss.reshape(-1).to(device)

        y_flat = y.reshape(-1)
        valid = y_flat >= 0
        sample_weights = batch.get("loss_weights")
        if sample_weights is None:
            weights_flat = torch.ones_like(loss_flat, dtype=torch.float32, device=device)
        else:
            weights_flat = sample_weights.reshape(-1).to(device=device, dtype=torch.float32)

        weighted_valid = valid & (weights_flat > 0)
        total_loss_sum += (loss_flat[weighted_valid] * weights_flat[weighted_valid]).sum()
        total_weight += weights_flat[weighted_valid].sum()

        if has_types:
            if "target_types" not in batch:
                raise ValueError("type_ids provided but batch missing 'target_types'.")
            tt_flat = batch["target_types"].reshape(-1).to(device)
            for i, tid in enumerate(type_ids):
                type_valid = weighted_valid & (tt_flat == tid)
                type_loss_sums[i] += (loss_flat[type_valid] * weights_flat[type_valid]).sum()
                type_weights[i] += weights_flat[type_valid].sum()

        if has_bpb:
            byte_valid = valid
            y_safe = torch.where(byte_valid, y_flat, torch.zeros_like(y_flat))
            num_bytes = torch.where(
                byte_valid,
                token_bytes[y_safe].to(torch.float32),
                torch.zeros_like(y_flat, dtype=torch.float32, device=device),
            )
            bpb_valid = byte_valid & (num_bytes > 0)
            total_nats += loss_flat[bpb_valid].sum()
            total_bytes += num_bytes[bpb_valid].sum()

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        tensors = [total_loss_sum, total_weight]
        if has_types:
            tensors.extend(type_loss_sums)
            tensors.extend(type_weights)
        if has_bpb:
            tensors.extend([total_nats, total_bytes])
        for tensor in tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    weight = total_weight.item()
    results = {"total_loss": total_loss_sum.item() / weight if weight > 0 else float("inf")}

    if has_types:
        results["type_losses"] = {}
        for tid, loss_sum, type_weight in zip(type_ids, type_loss_sums, type_weights):
            w = type_weight.item()
            results["type_losses"][tid] = loss_sum.item() / w if w > 0 else float("inf")

    if has_bpb:
        byte_count = total_bytes.item()
        results["bpb"] = total_nats.item() / (math.log(2) * byte_count) if byte_count > 0 else float("inf")

    return results


__all__ = ["evaluate_loss"]
