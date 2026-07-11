"""
Trainer for Nanoinfra GPT models.

Single-stage training class that can be used standalone or within curriculum training.
Extensible via inheritance for domain-specific trainers.
"""

import os
import time

try:
    import wandb
except ImportError:
    wandb = None

import torch

from core.utils import print0, DummyWandb
from core.training.lr_schedulers import get_lr_multiplier


def create_optimizers(system, optimizer_config: dict, world_size: int):
    """Build the multi-LR AdamW optimizers for an LMSystem.

    Thin wrapper kept for orchestrator import-compatibility; delegates to
    core.training.optim.build_optimizers (groups sourced by trunk/head ownership).
    """
    from core.training.optim import build_optimizers
    return build_optimizers(system, optimizer_config, world_size)


def detect_gpu_type() -> tuple[str, float]:
    """
    Auto-detect GPU type and return corresponding TFLOPS (bf16).

    Returns:
        tuple: (gpu_name, promised_flops)
            - gpu_name: str - 'H100' or 'H20'
            - promised_flops: float - TFLOPS in bf16 precision

    Raises:
        ValueError: If GPU type is not supported (not H100 or H20)

    Supported GPUs:
        - H100 SXM5: 989 TFLOPS (bf16)
        - H20: 296 TFLOPS (bf16)
    """
    gpu_device_name = torch.cuda.get_device_name(0).upper()

    # GPU performance specs (bf16 TFLOPS)
    gpu_specs = {
        'H100': 989e12,  # H100 SXM5 80GB
        'H20': 296e12,   # H20
        # Consumer Blackwell (local dev / first-step verification box). bf16 dense
        # tensor throughput, no sparsity; consumer FP32-accumulate may run lower, so
        # MFU computed against this is approximate. Affects the MFU metric only, not
        # training math.
        'RTX 5090': 209.5e12,
    }

    # Detect GPU type
    for gpu_name, flops in gpu_specs.items():
        if gpu_name in gpu_device_name:
            return gpu_name, flops

    # Unsupported GPU - raise error
    supported_gpus = ', '.join(gpu_specs.keys())
    raise ValueError(
        f"Unsupported GPU type: {gpu_device_name}\n"
        f"Currently supported GPUs: {supported_gpus}\n"
        f"Please add your GPU type to detect_gpu_type() in trainer.py"
    )


class Trainer:
    """
    Single-stage trainer for GPT models.

    Can be used standalone or within curriculum training.
    Extensible via inheritance for custom evaluation or training steps.

    DataLoader Interface (duck typing):
        The dataloader must be an iterable yielding dicts with these required keys:
        - idx: [B, T] token IDs
        - token_types: [B, T] token type IDs (0=text, 1=motion, 2=control)
        - targets: [B, T] target token IDs

        Optional keys (ignored by Trainer):
        - state_dict: Dataloader checkpoint state
        - attention_mask: Internal use only (already converted to -1 in targets)

        IMPORTANT: The dataloader MUST be infinite (cycle through data indefinitely).
        If it raises StopIteration, training will fail immediately to expose the issue.

    Example:
        >>> trainer = Trainer(
        ...     system=system,
        ...     optimizers=optimizers,
        ...     dataloader=dataloader,
        ...     config=config,
        ...     rank=0,
        ...     world_size=1,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        system,          # LMSystem (trunk + head); holds its own compiled trunk
        optimizers,      # List of optimizers
        dataloader,      # Data loader (duck typing interface)
        config,          # Config dict with all needed fields
        rank,
        world_size,
        debug_tokenizer=None,  # Optional tokenizer for debug printing
        evaluators=None,       # List of Evaluator objects for validation
    ):
        """
        Initialize Trainer.

        Args:
            system: LMSystem exposing loss(batch)->scalar + nn.Module (params/state_dict/
                train-eval). The compiled trunk lives inside it; grad clip/optim use its
                raw registered params. Replaces the old model/orig_model split.
            optimizers: List of optimizers
            dataloader: Iterable yielding batches (dict with idx, token_types, targets)
            config: Full training configuration dict with required top-level keys:
                - max_steps: Total training steps (int, -1 for auto-calculation)
                - target_param_data_ratio: For auto-calculating max_steps (int, default: 20)
                - sequence_len: Sequence length (int)
                - device_batch_size: Batch size per device (int)
                - total_batch_size: Total batch size in tokens (int)
                - optimizer: Optimizer config dict (includes max_grad_norm)
                - logging: Logging config dict (optional)
                - checkpoint: Checkpoint config dict (optional)
                - profiling: Profiling config dict (optional)
                - wandb: Wandb config dict (optional)
            rank: Distributed rank
            world_size: Distributed world size
        """
        self.system = system
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.debug_tokenizer = debug_tokenizer
        self.evaluators = evaluators or []

        # Extract config fields (from flattened structure)
        self.max_steps = config['max_steps']
        self.num_iterations = config.get('num_iterations', -1)  # Early stopping iterations (-1 = disabled)
        self.optimizer_config = config['optimizer']
        self.sequence_len = config['sequence_len']
        self.device_batch_size = config['device_batch_size']
        self.total_batch_size = config['total_batch_size']
        self.max_grad_norm = self.optimizer_config['max_grad_norm']
        self.logging_config = config.get('logging', {})

        # Calculate gradient accumulation
        tokens_per_step = self.device_batch_size * self.sequence_len * world_size
        assert self.total_batch_size % tokens_per_step == 0, \
            f"total_batch_size ({self.total_batch_size}) must be divisible by tokens_per_step ({tokens_per_step})"
        self.grad_accum_steps = self.total_batch_size // tokens_per_step

        # Setup autocast for mixed precision
        self.autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

        # Master process flag
        self.master_process = (rank == 0)

        # Auto-calculate max_steps if not specified (max_steps == -1)
        # Uses Chinchilla optimal scaling: tokens = target_param_data_ratio * num_params
        if self.max_steps == -1:
            target_param_data_ratio = config.get('target_param_data_ratio', 20)
            num_params = sum(p.numel() for p in system.parameters())
            target_tokens = target_param_data_ratio * num_params
            self.max_steps = target_tokens // self.total_batch_size

            if self.master_process:
                print0(f"\n{'='*80}")
                print0(f"Auto-calculating max_steps (max_steps was -1):")
                print0(f"  Model parameters: {num_params:,}")
                print0(f"  Target param:data ratio: {target_param_data_ratio}")
                print0(f"  Target tokens: {target_tokens:,}")
                print0(f"  Total batch size: {self.total_batch_size:,}")
                print0(f"  Calculated max_steps: {self.max_steps:,}")
                print0(f"{'='*80}\n")

        # Performance tracking
        self.get_max_memory = torch.cuda.max_memory_allocated
        self.sync_device = torch.cuda.synchronize  # Pre-bind CUDA sync
        self.num_flops_per_token = system.estimate_flops()

        # Auto-detect GPU type and get promised FLOPS
        self.gpu_type, gpu_flops = detect_gpu_type()
        self.promised_flops_per_sec = gpu_flops * world_size

        # Evaluation schedule: each evaluator answers should_eval(step) itself
        # (default: every interval_steps; or an explicit eval_at set, e.g. a
        # log-spaced schedule computed by the orchestrator). The Trainer just
        # asks every step — a few python calls, no schedule math in core.

        # Logging configuration (extract once)
        self.log_every = self.logging_config.get('log_every', 10)
        self.wandb_log_every = self.logging_config.get('wandb_log_every', 100)

        # Checkpoint configuration
        self.checkpoint_config = config.get('checkpoint', {})
        self.checkpoint_enabled = self.checkpoint_config.get('enabled', False)

        # Scheduler config: extract from optimizer.scheduler and add max_steps
        scheduler_config = self.optimizer_config.get('scheduler', {})
        self.scheduler_config = {**scheduler_config, 'max_steps': self.max_steps}

        # Profiling configuration (optional, for performance debugging)
        profiling_config = config.get('profiling', {})
        self.profiling_enabled = profiling_config.get('enabled', False)
        self.profile_start_step = profiling_config.get('start_step', 5)
        self.profile_end_step = profiling_config.get('end_step', 10)
        self.profile_save_dir = os.path.expandvars(profiling_config.get('save_dir', './traces'))
        self.profile_rank_0_only = profiling_config.get('profile_rank_0_only', True)
        self.record_shapes = profiling_config.get('record_shapes', True)
        self.with_stack = profiling_config.get('with_stack', True)
        self.profile_memory = profiling_config.get('profile_memory', False)
        self.profiler = None

        # Load checkpoint if needed (resume or init)
        from core.model.checkpoint_manager import load_checkpoint_if_needed
        self.start_step, self._resumed_trainer_state = load_checkpoint_if_needed(
            checkpoint_config=self.checkpoint_config,
            model=self.system,
            optimizers=self.optimizers,
            dataloader=self.dataloader,
            rank=self.rank,
            world_size=self.world_size,
        )

    def train(self):
        """
        Main training loop for one stage.

        This method can be overridden in subclasses for custom training loops.
        """
        # Print training info
        self._print_training_info()

        # Initialize wandb
        wandb_run = self._init_wandb()

        # Training setup
        self.system.train()
        data_iter = iter(self.dataloader)
        step = self.start_step  # Resume from checkpoint step if loaded

        # Track dataloader state for checkpointing
        current_dataloader_state = None

        # Timing and statistics
        t0 = time.time()
        ema_beta = 0.9  # EMA smoothing factor for loss tracking

        # Initialize or restore trainer state
        if self._resumed_trainer_state:
            # Resume from checkpoint
            smooth_train_loss = self._resumed_trainer_state.get('smooth_train_loss', 0.0)
            ema_factor = self._resumed_trainer_state.get('ema_factor', 1.0)
            total_training_time = self._resumed_trainer_state.get('total_training_time', 0.0)
        else:
            # Fresh training
            smooth_train_loss = 0.0  # EMA smoothed loss
            total_training_time = 0.0  # Cumulative training time
            ema_factor = 1.0  # Tracks (1 - beta^(step+1)) for debiasing

        while step < self.max_steps:
            # Start profiling
            if self.profiling_enabled and step == self.profile_start_step:
                self._start_profiler()

            # Learning rate schedule
            lrm = self._apply_lr_schedule(step)

            # Training step with gradient accumulation
            loss_accum, grad_norm, current_dataloader_state = self._training_step(data_iter, step)

            # Stop profiling and export
            if self.profiling_enabled and step == self.profile_end_step:
                self._stop_and_export_profiler()

            # Timing (synchronize before measuring to get accurate GPU time)
            self.sync_device()
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Update EMA smoothed loss (with iterative debiasing)
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss_accum.item()
            ema_factor = ema_factor * ema_beta
            debiased_smooth_loss = smooth_train_loss / (1 - ema_factor)

            # Accumulate training time
            total_training_time += dt

            # Console logging
            if step % self.log_every == 0 or step == self.max_steps - 1:
                metrics = self._calculate_metrics(step, dt)
                progress_str = self._get_progress_string(step)
                print0(
                    f"{progress_str} | "
                    f"loss: {debiased_smooth_loss:.6f} | "
                    f"grad_norm: {grad_norm:.4f} | "
                    f"lrm: {lrm:.2f} | "
                    f"dt: {dt*1000:.0f}ms | "
                    f"tok/s: {metrics['tokens_per_sec']:,} | "
                    f"mfu: {metrics['mfu']:.2f}% | "
                    f"total time: {total_training_time/60:.2f}m"
                )
                # Debug: print optimizer state (disabled)
                # self._debug_print_optimizer_state(step)

            # GPU memory breakdown at step 10
            # if step == 10 and self.device_type == 'cuda':
            #     self._debug_print_gpu_memory(step)

            # Wandb logging
            if step % self.wandb_log_every == 0 or step == self.max_steps - 1:
                metrics = self._calculate_metrics(step, dt)
                log_data = {
                    "step": step,
                    "tokens": self.total_batch_size * step,
                    "total_training_flops": metrics['flops_so_far'],
                    "total_training_time": total_training_time,
                    "train/loss": debiased_smooth_loss,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/tok_per_sec": metrics['tokens_per_sec'],
                    "train/mfu": metrics['mfu'],
                    "train/grad_norm": grad_norm,
                    "train/peak_memory_gb": self.get_max_memory() / 1024**3,
                }
                wandb_run.log(log_data)

            # Evaluation
            last_step = (step == self.max_steps - 1)
            if last_step or any(ev.should_eval(step) for ev in self.evaluators):
                eval_results = self._evaluate(step, force=last_step)
                metrics = self._calculate_metrics(step, dt)
                eval_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in eval_results.items()
                    if isinstance(v, (int, float))
                )
                print0(f"Step {step:05d} | {eval_str}")

                # Log to wandb
                if self.master_process:
                    wandb_run.log({
                        "step": step,
                        "total_training_flops": metrics['flops_so_far'],
                        "total_training_time": total_training_time,
                        **eval_results,
                    })

            # Checkpoint saving
            if self.checkpoint_enabled:
                from core.model.checkpoint_manager import save_checkpoint_if_needed
                # Package trainer internal state for checkpoint
                trainer_state = {
                    'smooth_train_loss': smooth_train_loss,
                    'ema_factor': ema_factor,
                    'total_training_time': total_training_time,
                }
                save_checkpoint_if_needed(
                    step, self.system, self.optimizers,
                    self.config, self.rank, self.world_size,
                    dataloader_state=current_dataloader_state,
                    trainer_state=trainer_state,
                    dataloader=self.dataloader,
                )

            # Increment step counter
            step += 1

            # Early stopping check (num_iterations = steps to run from start)
            if self.num_iterations != -1 and (step - self.start_step) >= self.num_iterations:
                if self.master_process:
                    print0(f"\n{'='*80}")
                    print0(f"Early stopping: reached num_iterations={self.num_iterations}")
                    print0(f"{'='*80}\n")
                break

        # Training complete
        self._print_completion_info()
        wandb_run.finish()

    def _apply_lr_schedule(self, step):
        """
        Apply learning rate schedule for current step.

        Args:
            step: Current training step

        Returns:
            float: Learning rate multiplier
        """
        lrm = get_lr_multiplier(step, self.scheduler_config)
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['initial_lr'] * lrm
        return lrm

    def _calculate_metrics(self, step, dt):
        """
        Calculate training performance metrics.

        Args:
            step: Current training step
            dt: Time delta for this step (seconds)

        Returns:
            dict: Metrics including tokens_per_sec, flops_so_far, flops_per_sec, mfu
        """
        tokens_per_sec = int(self.total_batch_size / dt) if dt > 0 else 0
        flops_so_far = self.num_flops_per_token * self.total_batch_size * step
        flops_per_sec = self.num_flops_per_token * self.total_batch_size / dt if dt > 0 else 0
        mfu = 100 * flops_per_sec / self.promised_flops_per_sec

        return {
            'tokens_per_sec': tokens_per_sec,
            'flops_so_far': flops_so_far,
            'flops_per_sec': flops_per_sec,
            'mfu': mfu,
        }

    def _training_step(self, data_iter, step):
        """
        Single training step with gradient accumulation.

        This method can be overridden in subclasses for custom training logic.

        Args:
            data_iter: Iterator over dataloader (must be infinite)
            step: Current training step (for debug logging)

        Returns:
            tuple: (loss_accum, grad_norm, dataloader_state)
        """
        loss_accum = 0.0
        last_dataloader_state = None

        for micro_step in range(self.grad_accum_steps):
            # Get batch (will raise StopIteration if dataloader is exhausted)
            batch = next(data_iter)

            # Debug: print first batch text (only on first micro-step) (disabled)
            # if micro_step == 0:
            #     self._debug_print_batch_text(step, batch)

            # Track dataloader state from last batch
            last_dataloader_state = batch.get('state_dict')

            # Forward pass with autocast (FSDP handles gradient sync automatically)
            # Note: Dataloader is responsible for GPU placement (all tensors already on device)
            with self.autocast_ctx:
                loss = self.system.loss(batch)

            # Scale loss by grad_accum_steps
            loss = loss / self.grad_accum_steps
            loss_accum += loss.detach()

            # Backward
            loss.backward()

        # Clip gradients on the system's raw registered params (shared with the
        # compiled trunk, so backward populated their .grad)
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.system.parameters(),
            self.max_grad_norm
        )
        # Convert to scalar (handle DTensor from FSDP)
        grad_norm = grad_norm_tensor.item() if hasattr(grad_norm_tensor, 'item') else float(grad_norm_tensor)

        # Optimizer step - step all optimizers
        for opt in self.optimizers:
            opt.step()

        # Zero gradients for next iteration
        self.system.zero_grad(set_to_none=True)

        return loss_accum, grad_norm, last_dataloader_state

    def _debug_print_batch_text(self, step, batch):
        """
        Debug helper: Print decoded text from first batch at each step.

        Only prints if:
        - step >= 10
        - debug_tokenizer was provided
        - this is the master process

        Args:
            step: Current training step
            batch: Batch dict with 'idx' key containing token IDs
        """
        if step < 10 or not self.debug_tokenizer or not self.master_process:
            return

        # Get first sequence from batch [B, T] -> [T]
        tokens = batch['idx'][0].cpu().tolist()

        # Decode first 50 tokens to avoid too much output
        tokens_to_decode = tokens[:50]
        text = self.debug_tokenizer.decode(tokens_to_decode)

        print0(f"\n[DEBUG] Step {step:05d} first text:")
        print0(f"  Tokens: {tokens_to_decode}")
        print0(f"  Text: {repr(text)}\n")

    def _debug_print_gpu_memory(self, step):
        """
        Debug helper: Print GPU memory summary.

        Only prints if this is the master process.

        Args:
            step: Current training step
        """
        if not self.master_process:
            return

        print0("\n" + "="*80)
        print0(f"GPU Memory Summary at Step {step}")
        print0("="*80)
        print0(torch.cuda.memory_summary(abbreviated=False))
        print0("="*80 + "\n")

    def _debug_print_optimizer_state(self, step):
        """
        Debug helper: Print optimizer internal state.

        Prints optimizer step counter and momentum/variance statistics.
        Only prints if this is the master process.

        Args:
            step: Current training step
        """
        if not self.master_process:
            return

        print0(f"\n[DEBUG OPTIMIZER] Step {step:05d}")

        for opt_idx, opt in enumerate(self.optimizers):
            opt_state = opt.state_dict()
            print0(f"  Optimizer {opt_idx}:")

            # Print per-parameter-group info
            for pg_idx, param_group in enumerate(opt.param_groups):
                print0(f"    ParamGroup {pg_idx}: lr={param_group['lr']:.6e}")

            # Print state for first parameter (to check momentum/variance)
            if 'state' in opt_state and len(opt_state['state']) > 0:
                # Get first parameter's state
                first_param_id = list(opt_state['state'].keys())[0]
                first_state = opt_state['state'][first_param_id]

                print0(f"    First param state (id={first_param_id}):")
                if 'step' in first_state:
                    print0(f"      step: {first_state['step']}")
                if 'exp_avg' in first_state:
                    exp_avg = first_state['exp_avg']
                    if hasattr(exp_avg, 'abs'):
                        print0(f"      exp_avg (momentum) mean: {exp_avg.abs().mean().item():.6e}")
                if 'exp_avg_sq' in first_state:
                    exp_avg_sq = first_state['exp_avg_sq']
                    if hasattr(exp_avg_sq, 'abs'):
                        print0(f"      exp_avg_sq (variance) mean: {exp_avg_sq.abs().mean().item():.6e}")
            else:
                print0(f"    No state (optimizer not stepped yet)")

        print0("")  # Empty line for readability

    def _evaluate(self, step, force=False):
        """
        Run evaluation by iterating over configured evaluators.

        Each evaluator runs when its own should_eval(step) fires — except when
        force=True (the FINAL step), where every evaluator runs so a training
        run always ends with a complete eval regardless of schedule.

        Args:
            step: Current training step
            force: Run all evaluators regardless of their interval

        Returns:
            dict[str, float]: Merged evaluation metrics from all evaluators
        """
        self.system.eval()
        results = {}
        for ev in self.evaluators:
            if force or ev.should_eval(step):
                results.update(ev.evaluate(self.system, self.autocast_ctx))
        self.system.train()
        return results

    def _init_wandb(self):
        """
        Initialize wandb for this stage.

        Reads wandb config from:
          1. config['wandb'] (project, name, enabled)
          2. Environment variables (WANDB_PROJECT, WANDB_NAME) - higher priority

        Returns:
            wandb run or DummyWandb
        """
        if not self.master_process:
            return DummyWandb()

        # Get wandb config from config dict (with defaults)
        wandb_config = self.config.get('wandb', {})
        enabled = wandb_config.get('enabled', True)

        # Check if wandb is disabled
        if not enabled:
            print0("wandb disabled (config['wandb']['enabled'] = False)\n")
            return DummyWandb()
        if wandb is None:
            print0("wandb is not installed; using no-op logger\n")
            return DummyWandb()

        # Get project and name (environment variables override config)
        wandb_project = os.environ.get('WANDB_PROJECT', wandb_config.get('project', 'nanoinfra'))
        wandb_name = os.environ.get('WANDB_NAME', wandb_config.get('name', 'training'))

        # Build config for wandb
        run_config = {
            'max_steps': self.max_steps,
            'device_batch_size': self.device_batch_size,
            'total_batch_size': self.total_batch_size,
            'sequence_len': self.sequence_len,
            'grad_accum_steps': self.grad_accum_steps,
            'world_size': self.world_size,
            'gpu_type': self.gpu_type,
            **self.optimizer_config,
        }

        wandb_run = wandb.init(project=wandb_project, name=wandb_name, config=run_config)
        print0(f"wandb initialized: {wandb_project}/{wandb_name}\n")

        return wandb_run

    def _print_training_info(self):
        """Print training configuration information."""
        print0(f"\n{'='*80}")
        print0(f"Training Configuration")
        print0(f"{'='*80}")
        print0(f"Max steps: {self.max_steps}")
        print0(f"Batch configuration:")
        print0(f"  Device batch size: {self.device_batch_size}")
        print0(f"  Sequence length: {self.sequence_len}")
        print0(f"  World size: {self.world_size}")
        print0(f"  Grad accum steps: {self.grad_accum_steps}")
        print0(f"  Total batch size: {self.total_batch_size} tokens")
        print0(f"Learning rate schedule:")
        print0(f"  lr_max: {self.optimizer_config['lr_max']}")
        print0(f"  scheduler: {self.scheduler_config.get('type', 'linear')}")
        print0(f"  warmup_steps: {self.scheduler_config.get('warmup_steps', 0)}")
        if self.scheduler_config.get('type', 'linear') == 'linear':
            print0(f"  warmdown_ratio: {self.scheduler_config.get('warmdown_ratio', 0.2)}")
        print0(f"  final_lr_frac: {self.scheduler_config.get('final_lr_frac', 0.0)}")
        print0(f"Performance:")
        print0(f"  GPU type: {self.gpu_type}")
        print0(f"  Promised FLOPS: {self.promised_flops_per_sec:e}")
        print0(f"  Estimated FLOPs per token: {self.num_flops_per_token:e}")
        if self.evaluators:
            schedules = [f"eval_at<{len(ev.eval_at)} steps>" if ev.eval_at is not None
                         else f"every {ev.interval_steps}" for ev in self.evaluators]
            print0(f"  Evaluation: {len(self.evaluators)} evaluators, schedules={schedules}")
        else:
            print0(f"  Evaluation: disabled (no evaluators)")
        if self.profiling_enabled:
            print0(f"Profiling:")
            print0(f"  Enabled: True (steps {self.profile_start_step}-{self.profile_end_step})")
            print0(f"  Save directory: {self.profile_save_dir}")
            print0(f"  Profile rank 0 only: {self.profile_rank_0_only}")
        print0(f"{'='*80}\n")

    def _print_completion_info(self):
        """Print training completion information."""
        print0(f"\n{'='*80}")
        print0(f"Training complete!")
        print0(f"{'='*80}\n")

    def _get_progress_string(self, step):
        """
        Get progress string for logging.

        Args:
            step: Current step

        Returns:
            str: Progress string
        """
        pct_done = 100 * step / self.max_steps
        return f"Step {step:05d}/{self.max_steps:05d} ({pct_done:5.1f}%)"

    def _start_profiler(self):
        """Start PyTorch profiler."""
        import warnings
        from torch.profiler import profile, ProfilerActivity

        # Only profile on designated ranks
        if self.profile_rank_0_only and self.rank != 0:
            return

        print0(f"Starting profiler (steps {self.profile_start_step}-{self.profile_end_step})...")

        # Suppress profiler cycle warnings (we use manual start/stop, not cycles)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Profiler clears events.*')
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
            )
            self.profiler.start()

    def _stop_and_export_profiler(self):
        """Stop profiler and export trace file."""
        if self.profiler is None:
            return

        print0("Stopping profiler...")
        self.profiler.stop()

        # Generate filename with metadata
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get model depth from the trunk config
        model_depth = self.system.trunk.config.n_layer

        trace_filename = f"trace_d{model_depth}_rank{self.rank}_step{self.profile_start_step}-{self.profile_end_step}_{timestamp}.json"
        trace_path = os.path.join(self.profile_save_dir, trace_filename)

        os.makedirs(self.profile_save_dir, exist_ok=True)

        print0(f"Exporting trace to {trace_path}...")
        print0("(This may take 1-5 seconds...)")
        self.profiler.export_chrome_trace(trace_path)
        print0(f"✓ Trace saved: {trace_path}")

        self.profiler = None
