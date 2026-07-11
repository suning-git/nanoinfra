"""AdamW optimizer construction for an LMSystem (multi-LR groups by module role).

Replaces the old GPT.setup_optimizers: the parameter groups are now sourced by
MODULE OWNERSHIP on the assembled system (trunk vs head) rather than by attributes
on a monolithic GPT. The parameter objects and their order are unchanged, so the
optimizer state maps identically (behavior-preserving).
"""

import torch

from core.utils import get_dist_info


def build_optimizers(system, optimizer_config: dict, world_size: int = 1):
    """Two AdamW optimizers with role-based LR groups, built from an LMSystem.

    Groups by role, derived generically so ANY trunk works (not just the modern GPT):
      - embedding:   the trunk's nn.Embedding modules   (lr = embedding_lr)
      - matrix:      every other trunk parameter        (lr = lr_max)
      - unembedding: the head                           (lr = unembedding_lr)
    For the modern GPT this reproduces the old wte+type_emb / h split exactly (same
    param objects, same order). All LRs scale by 1/sqrt(n_embd / 768). Returns
    [adamw(unemb+emb), adamw(matrix)].
    """
    assert optimizer_config['type'] == 'adamw', \
        f"Only AdamW supported, got {optimizer_config['type']}"
    unembedding_lr = optimizer_config.get('unembedding_lr', 0.004)
    embedding_lr = optimizer_config.get('embedding_lr', 0.2)
    adam_matrix_lr = optimizer_config.get('lr_max', 3e-4)  # lr_max is the matrix-group LR
    weight_decay = optimizer_config.get('weight_decay', 0.01)
    betas = tuple(optimizer_config.get('betas', [0.9, 0.95]))

    trunk, head = system.trunk, system.head
    model_dim = trunk.config.n_embd
    rank, _, _, _ = get_dist_info()

    # Derived generically so any trunk satisfies it: embeddings (token / position /
    # type) take the embedding LR; every other trunk param (blocks, norms, ...) is a
    # matrix param; the head is the unembedding. For the modern GPT this is exactly
    # the old wte+type_emb / h split (same param objects, same order) -> state maps 1:1.
    embedding_params, seen = [], set()
    for module in trunk.modules():
        if isinstance(module, torch.nn.Embedding):
            for p in module.parameters():
                embedding_params.append(p)
                seen.add(id(p))
    matrix_params = [p for p in trunk.parameters() if id(p) not in seen]
    unembedding_params = list(head.parameters())

    # Scale AdamW LRs ∝ 1/√dmodel (LRs were tuned for a 768-dim model).
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    if rank == 0:
        print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

    adamw_kwargs = dict(betas=betas, eps=1e-10, weight_decay=weight_decay)
    adam_groups = [
        dict(params=unembedding_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
    ]
    adamw_optimizer = torch.optim.AdamW(adam_groups, fused=True, **adamw_kwargs)

    if rank == 0:
        print(f"Using AdamW optimizer for matrix parameters "
              f"(lr={adam_matrix_lr}, betas={betas}, wd={weight_decay})")
    matrix_groups = [dict(params=matrix_params, lr=adam_matrix_lr * dmodel_lr_scale)]
    matrix_optimizer = torch.optim.AdamW(matrix_groups, fused=True, **adamw_kwargs)

    optimizers = [adamw_optimizer, matrix_optimizer]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers
