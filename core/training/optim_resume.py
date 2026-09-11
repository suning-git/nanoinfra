"""Reconcile optimizer hyperparameters after restoring a checkpoint.

Trainer snapshots the constructed optimizer groups before loading. After a resume,
adopt=False rejects differing adjustable values; adopt=True copies them from the
snapshot, preserving the loaded weights, moments and step counts. The builder is
the sole source of group values, including defaults and LR scaling.

Adjustable fields: initial_lr, weight_decay, betas, eps. Trainer recomputes the
transient lr before the next update, so it is ignored here. Other group options
(amsgrad, fused, maximize, etc.) must match under either policy.

DCP loads using the live optimizer's keys: missing checkpoint keys fail the load,
while checkpoint-only keys are omitted. This helper compares the resulting groups.
"""

import copy
import math

ADJUSTABLE = ("initial_lr", "weight_decay", "betas", "eps")
_SKIP = ("params", "lr")
# Allow rounding differences from equivalent LR scaling expressions.
# With adopt=True, values within this tolerance are still copied from the snapshot.
REL_TOL = 1e-12


def snapshot_hparams(optimizers):
    """Copies of every param_group entry except the parameters themselves."""
    return [[copy.deepcopy({k: v for k, v in g.items() if k != "params"}) for g in opt.param_groups]
            for opt in optimizers]


def _same(a, b):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0)
    return a == b


def reconcile_hparams(optimizers, snapshot, *, adopt):
    """Compare the loaded groups with the pre-load snapshot; refuse or adopt.

    Returns changes beyond the comparison tolerance as
    [(opt_idx, group_idx, key, checkpoint_value, new_value)].
    All checks precede hyperparameter writes. A failure aborts training;
    it does not roll back the preceding checkpoint load.
    """
    if not isinstance(adopt, bool):
        raise TypeError(f"checkpoint.override_optimizer_hparams must be true or false, got {adopt!r}")
    if len(snapshot) != len(optimizers) or any(
            len(groups) != len(opt.param_groups) for groups, opt in zip(snapshot, optimizers)):
        raise ValueError(
            f"resume: snapshot has {[len(g) for g in snapshot]} param groups per optimizer, the "
            f"optimizers have {[len(o.param_groups) for o in optimizers]}; they must be the same objects")

    def finite(v):
        return all(math.isfinite(x) for x in v) if isinstance(v, (tuple, list)) else \
            (math.isfinite(v) if isinstance(v, float) else True)

    diffs, incompatible, changes, writes = [], [], [], []
    for i, (opt, groups) in enumerate(zip(optimizers, snapshot)):
        for j, (group, before) in enumerate(zip(opt.param_groups, groups)):
            where = f"optimizer[{i}].param_groups[{j}]"
            if "initial_lr" not in before:
                raise ValueError(f"resume: {where} has no initial_lr; the LR schedule needs it "
                                 "(core's build_optimizers sets it)")
            # Require matching field sets before comparing or adopting values.
            for k in sorted((set(before) | set(group)) - set(_SKIP)):
                if k not in group:
                    incompatible.append(f"{where}.{k}: this launch {before[k]!r}, absent in the checkpoint")
                    continue
                if k not in before:
                    incompatible.append(f"{where}.{k}: checkpoint {group[k]!r}, absent in this launch")
                    continue
                mine, loaded = before[k], group[k]
                if k in ADJUSTABLE:
                    if not finite(mine):
                        raise ValueError(f"resume: this launch's {where}.{k}={mine!r} is not a usable value")
                    if adopt:
                        writes.append((group, k, mine))          # this launch's value, always
                        if not _same(mine, loaded):
                            changes.append((i, j, k, loaded, mine))
                    elif not _same(mine, loaded):
                        diffs.append(f"{where}.{k}: checkpoint {loaded!r}, this launch {mine!r}")
                elif mine != loaded:
                    incompatible.append(f"{where}.{k}: checkpoint {loaded!r}, this launch {mine!r}")

    problems = []
    if incompatible:
        problems.append(
            "optimizer options differ from the checkpoint's and cannot be overridden (they shape "
            "the optimizer state or the update): " + "; ".join(incompatible)
            + ". Build the optimizer the way the checkpoint's was built")
    if diffs:
        problems.append(
            "this launch's optimizer hyperparameters differ from the checkpoint's: " + "; ".join(diffs)
            + ". A resume keeps the checkpoint's values. Either set the config to those values, or "
              "set checkpoint.override_optimizer_hparams=true to adopt this launch's values (all of "
              "them). LR values are per-group initial_lr after the builder's scaling, not the "
              "config's lr keys")
    if problems:
        raise ValueError("resume: " + ". ".join(problems) + ".")
    for group, k, value in writes:
        group[k] = value
    return changes
