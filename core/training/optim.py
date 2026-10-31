"""AdamW optimizer construction for an LMSystem (multi-LR groups by module role).

Replaces the old GPT.setup_optimizers: the parameter groups are now sourced by
MODULE OWNERSHIP on the assembled system (trunk vs head) rather than by attributes
on a monolithic GPT. The parameter objects and their order are unchanged, so the
optimizer state maps identically (behavior-preserving).
"""

import torch

from core.utils import print0


def no_weight_decay_params(system):
    """The parameters of the system's trunk and head that their modules declare exempt from weight decay.

    A module declares them in `no_weight_decay_names`: a tuple of names of its OWN parameters (local names, no dotted
    paths), as a class attribute or set on the instance, which then replaces the class's. Whether decay is right for
    a parameter depends on whether 0 is its neutral value, which only its designer knows: a gain written g * x has 1
    (exempt it), one written (1 + w) * x has 0 (keep the decay); so nothing is inferred from shapes, names or module
    types. The declaration is model code: it is not saved in a checkpoint, and a resume does not compare it with one.

    It sits on the module, not on the parameter object, because the objects get replaced: build_system's to_empty from
    the meta device, FSDP2's fully_shard and deepcopy all create new parameters, while the module and the parameter's
    name stay. It is read from the instance's own attributes and its class, not through getattr: a wrapper forwarding
    attribute access to the module it wraps (activation checkpointing's) would present that module's declaration as
    its own. A parameter registered under two names can be declared under either; one shared by two modules is exempt
    if either declares it. A name that is not one of the module's own parameters raises: a typo, or a parameter
    renamed (or moved by a parametrization) without its declaration, would otherwise keep the decay silently.
    -> the parameters, each once, in declaration order."""
    exempt = {}
    for part_name in ("trunk", "head"):
        for path, module in getattr(system, part_name).named_modules(prefix=part_name):
            names = vars(module).get("no_weight_decay_names", getattr(type(module), "no_weight_decay_names", ()))
            if not names:
                continue
            own = dict(module.named_parameters(recurse=False, remove_duplicate=False))
            for name in names:
                if name not in own:
                    raise ValueError(
                        f"{path} ({type(module).__name__}): no_weight_decay_names lists {name!r}, which is not one of "
                        f"its own parameters {sorted(own)}; declare a parameter by its local name on the module that "
                        f"registers it")
                exempt.setdefault(id(own[name]), own[name])
    return list(exempt.values())


def _exempt_split(groups, exempt):
    """Each group holding exempt parameters becomes two with the same settings: the others, then the exempt ones at
    weight_decay 0, each in the group's order. A group without any is kept as it is, so with nothing declared the
    groups are unchanged; a side left empty is not built (DCP refuses an empty group when there are several
    optimizers)."""
    out = []
    for group in groups:
        free = [p for p in group["params"] if id(p) in exempt]
        if not free:
            out.append(group)
            continue
        rest = [p for p in group["params"] if id(p) not in exempt]
        if rest:
            out.append(dict(group, params=rest))
        out.append(dict(group, params=free, weight_decay=0.0))
    return out


def build_optimizers(system, optimizer_config: dict):
    """Two AdamW optimizers with role-based LR groups, built from an LMSystem.

    Groups by role, derived generically so ANY trunk works (not just the modern GPT):
      - embedding:   the trunk's nn.Embedding modules   (lr = embedding_lr)
      - matrix:      every other trunk parameter        (lr = lr_max)
      - unembedding: the head                           (lr = unembedding_lr)
    For the modern GPT this reproduces the old wte+type_emb / h split exactly (same
    param objects, same order). All LRs scale by 1/sqrt(n_embd / 768). Returns
    [adamw(unemb+emb), adamw(matrix)].

    weight_decay applies to every parameter except those their modules declare exempt
    (no_weight_decay_params): in each role they form a group of their own right after
    it, at the same LR and weight_decay 0. With nothing declared (the modern GPT) the
    groups are exactly the three above.
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
    print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

    # eps and the fused kernel are fixed, not config: eps=1e-10 follows nanochat (it only acts on a parameter whose
    # gradient scale nears it), and fused=True runs the same algorithm as the unfused implementations, faster, though
    # not bit for bit (a different arithmetic order; for bf16 tensors it also keeps intermediates in fp32).
    adamw_kwargs = dict(betas=betas, eps=1e-10, weight_decay=weight_decay)
    exempt = {id(p) for p in no_weight_decay_params(system)}
    adam_groups = _exempt_split([
        dict(params=unembedding_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
    ], exempt)
    adamw_optimizer = torch.optim.AdamW(adam_groups, fused=True, **adamw_kwargs)

    print0(f"Using AdamW optimizer for matrix parameters "
           f"(lr={adam_matrix_lr}, betas={betas}, wd={weight_decay})")
    if exempt:
        print0(f"Weight decay exempt for {len(exempt)} of the parameters, as their modules declare "
               f"(no_weight_decay_names)")
    matrix_groups = _exempt_split([dict(params=matrix_params, lr=adam_matrix_lr * dmodel_lr_scale)], exempt)
    matrix_optimizer = torch.optim.AdamW(matrix_groups, fused=True, **adamw_kwargs)

    optimizers = [adamw_optimizer, matrix_optimizer]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers
