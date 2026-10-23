"""
Learning-rate schedules for the Trainer: a multiplier on each parameter group's initial_lr, step by step.

    schedule = build_lr_schedule(scheduler_config, max_steps)   # once, when the run's max_steps is known
    multiplier = schedule(step)                                  # every step: arithmetic only

build_lr_schedule reads the config, fills its defaults and checks every rule there, once; the schedule it returns
holds the resolved numbers and only computes. A rule that depends on max_steps (warmup and warmdown must fit in it)
waits for the final value, which the Trainer resolves when the config says -1.

Another schedule: the orchestrator replaces lr_schedulers.build_lr_schedule before it builds the Trainer,
with a function (config, max_steps) -> any callable step -> multiplier; the Trainer looks it
up on this module.

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
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearSchedule:
    """Warmup -> constant -> linear warmdown toward final_lr_frac. The target is reached at step max_steps, one past
    the last update (steps run 0 .. max_steps - 1), so the last update uses a little more; with no warmdown it stays
    1."""
    max_steps: int
    warmup_steps: int
    warmdown_ratio: float
    final_lr_frac: float

    @property
    def warmdown_steps(self) -> int:
        return round(self.warmdown_ratio * self.max_steps)

    def __call__(self, step: int) -> float:
        warmdown_steps = self.warmdown_steps
        if step < self.warmup_steps:
            # Linear warmup: 0 -> 1
            return (step + 1) / self.warmup_steps
        elif step <= self.max_steps - warmdown_steps:
            # Constant phase
            return 1.0
        else:
            # Linear warmdown: 1 -> final_lr_frac
            progress = (self.max_steps - step) / warmdown_steps
            return progress * 1.0 + (1 - progress) * self.final_lr_frac


@dataclass(frozen=True)
class CosineSchedule:
    """Warmup -> cosine annealing toward final_lr_frac, reached at step max_steps, one past the last update."""
    max_steps: int
    warmup_steps: int
    final_lr_frac: float

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup: 0 -> 1
            return (step + 1) / self.warmup_steps
        else:
            # Cosine annealing: 1 -> final_lr_frac
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.final_lr_frac + (1 - self.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def build_lr_schedule(config: dict, max_steps: int):
    """
    The schedule `config` describes, for a run of `max_steps` steps; every rule is checked here.

    Config keys (optimizer.scheduler):
        type: 'linear' (default) or 'cosine'
        warmup_steps: ABSOLUTE warmup steps (default: 0; see module note)
        warmdown_ratio: linear only, the fraction of steps for warmdown (default: 0.2)
        final_lr_frac: where the decay heads, as a fraction of each group's initial LR (default: 0.0)

    Raises:
        ValueError: an unknown type, the removed warmup_ratio, or a value out of range
    """
    if 'warmup_ratio' in config:
        raise ValueError(
            "warmup_ratio was removed: core speaks ABSOLUTE warmup steps (a ratio "
            "re-tunes the recipe whenever the budget changes). Convert at the "
            "orchestrator: warmup_steps = ratio * max_steps.")
    scheduler_type = config.get('type', 'linear')
    warmup_steps = int(config.get('warmup_steps', 0))
    final_lr_frac = config.get('final_lr_frac', 0.0)
    if scheduler_type not in ('linear', 'cosine'):
        raise ValueError(f"Unknown scheduler type: '{scheduler_type}'. Available: cosine, linear")
    if not max_steps > 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if not 0 <= final_lr_frac <= 1:
        raise ValueError(f"final_lr_frac must be in [0, 1], got {final_lr_frac}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

    if scheduler_type == 'linear':
        warmdown_ratio = config.get('warmdown_ratio', 0.2)
        if not 0 <= warmdown_ratio < 1:
            raise ValueError(f"warmdown_ratio must be in [0, 1), got {warmdown_ratio}")
        schedule = LinearSchedule(max_steps, warmup_steps, warmdown_ratio, final_lr_frac)
        if not warmup_steps <= max_steps - schedule.warmdown_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) + warmdown ({schedule.warmdown_steps}) exceed "
                             f"max_steps ({max_steps})")
        return schedule

    if not warmup_steps < max_steps:
        raise ValueError(f"warmup_steps must be in [0, max_steps), got {warmup_steps}")
    return CosineSchedule(max_steps, warmup_steps, final_lr_frac)
