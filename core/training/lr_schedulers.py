"""
LR Scheduler Registry for nanoinfra trainer.

Provides configurable learning rate schedules via a registry pattern.
Each scheduler is a pure function: (step, config) -> multiplier.

Usage in config:
    optimizer:
      scheduler:
        type: cosine       # 'linear' | 'cosine'
        warmup_steps: 200  # ABSOLUTE steps (see note below)
        final_lr_frac: 0.0

Convention — core speaks ABSOLUTE steps for warmup: warmup exists to stabilize
early optimization, so its natural scale is steps, independent of how long the
run happens to be (a ratio silently re-tunes the recipe whenever the budget
changes). If a recipe thinks in fractions, the ORCHESTRATOR converts
(steps = ratio * max_steps). Warmdown stays a FRACTION on purpose: it is
anchored to the END of the horizon (and with Chinchilla auto-sizing the
horizon isn't statically known), so a fraction IS its natural parameter.

Usage in code:
    from core.training.lr_schedulers import get_lr_multiplier
    multiplier = get_lr_multiplier(step, scheduler_config)
"""

import math
from typing import Callable

# Type alias for scheduler functions
SchedulerFn = Callable[[int, dict], float]


def linear_schedule(step: int, config: dict) -> float:
    """
    Warmup -> Constant -> Linear warmdown schedule.

    Config params:
        max_steps: Total training steps (required)
        warmup_steps: ABSOLUTE warmup steps (default: 0; see module note)
        warmdown_ratio: Fraction of steps for warmdown (default: 0.2)
        final_lr_frac: Final LR as fraction of max (default: 0.0)

    Returns:
        LR multiplier in [final_lr_frac, 1.0]
    """
    assert 'warmup_ratio' not in config, (
        "warmup_ratio was removed: core speaks ABSOLUTE warmup steps (a ratio "
        "re-tunes the recipe whenever the budget changes). Convert at the "
        "orchestrator: warmup_steps = ratio * max_steps.")
    max_steps = config['max_steps']
    warmup_steps = int(config.get('warmup_steps', 0))
    warmdown_ratio = config.get('warmdown_ratio', 0.2)
    final_lr_frac = config.get('final_lr_frac', 0.0)

    # Validate parameters
    assert max_steps > 0, f"max_steps must be positive, got {max_steps}"
    assert 0 <= warmdown_ratio < 1, f"warmdown_ratio must be in [0, 1), got {warmdown_ratio}"
    assert 0 <= final_lr_frac <= 1, f"final_lr_frac must be in [0, 1], got {final_lr_frac}"

    warmdown_steps = round(warmdown_ratio * max_steps)
    assert 0 <= warmup_steps <= max_steps - warmdown_steps, \
        f"warmup_steps ({warmup_steps}) + warmdown ({warmdown_steps}) exceed max_steps ({max_steps})"

    if step < warmup_steps:
        # Linear warmup: 0 -> 1
        return (step + 1) / warmup_steps
    elif step <= max_steps - warmdown_steps:
        # Constant phase
        return 1.0
    else:
        # Linear warmdown: 1 -> final_lr_frac
        progress = (max_steps - step) / warmdown_steps
        return progress * 1.0 + (1 - progress) * final_lr_frac


def cosine_schedule(step: int, config: dict) -> float:
    """
    Warmup -> Cosine annealing schedule.

    Config params:
        max_steps: Total training steps (required)
        warmup_steps: ABSOLUTE warmup steps (default: 0; see module note)
        final_lr_frac: Final LR as fraction of max (default: 0.0)

    Returns:
        LR multiplier in [final_lr_frac, 1.0]
    """
    assert 'warmup_ratio' not in config, (
        "warmup_ratio was removed: core speaks ABSOLUTE warmup steps (a ratio "
        "re-tunes the recipe whenever the budget changes). Convert at the "
        "orchestrator: warmup_steps = ratio * max_steps.")
    max_steps = config['max_steps']
    warmup_steps = int(config.get('warmup_steps', 0))
    final_lr_frac = config.get('final_lr_frac', 0.0)

    # Validate parameters
    assert max_steps > 0, f"max_steps must be positive, got {max_steps}"
    assert 0 <= warmup_steps < max_steps, f"warmup_steps must be in [0, max_steps), got {warmup_steps}"
    assert 0 <= final_lr_frac <= 1, f"final_lr_frac must be in [0, 1], got {final_lr_frac}"

    if step < warmup_steps:
        # Linear warmup: 0 -> 1
        return (step + 1) / warmup_steps
    else:
        # Cosine annealing: 1 -> final_lr_frac
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return final_lr_frac + (1 - final_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))


# Registry: scheduler_type -> function
SCHEDULERS: dict[str, SchedulerFn] = {
    'linear': linear_schedule,
    'cosine': cosine_schedule,
}


def get_lr_multiplier(step: int, config: dict) -> float:
    """
    Get LR multiplier for current step using configured scheduler.

    Args:
        step: Current training step
        config: Scheduler config dict containing:
            - type: Scheduler type ('linear' or 'cosine')
            - max_steps: Total training steps
            - Scheduler-specific params (warmup_steps, etc.)

    Returns:
        LR multiplier to apply to initial learning rates

    Raises:
        ValueError: If scheduler type is unknown
    """
    scheduler_type = config.get('type', 'linear')
    if scheduler_type not in SCHEDULERS:
        available = ', '.join(sorted(SCHEDULERS.keys()))
        raise ValueError(f"Unknown scheduler type: '{scheduler_type}'. Available: {available}")
    return SCHEDULERS[scheduler_type](step, config)
