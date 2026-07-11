"""
Model setup for distributed training.

Provides hardware deployment layer for:
- Distributed initialization (multi-GPU)
- Model creation with FSDP
- Model compilation

Note: nanoinfra is designed for CUDA training only.
"""

import os

# IMPORTANT: Configure CUDA memory allocator (matching the training baseline)
# Note: This setting is necessary for consistency with baseline, but testing shows
# it does NOT resolve the ~2-3% MFU difference (34% vs 37% on d20/2nodes).
# The MFU gap likely stems from torch.compile() compilation paths or other
# structural differences between the standalone and integrated trainers.
# This is acceptable for test infrastructure; loss trajectories remain identical.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    fully_shard, MixedPrecisionPolicy, FSDPModule, register_fsdp_forward_method,
)
from torch.distributed.device_mesh import init_device_mesh

from core.model.gpt import GPT
from core.model.heads import LMHead, LigerLMHead, LIGER_AVAILABLE
from core.model.system import LMSystem


def print0(*args, **kwargs):
    """Print only from rank 0."""
    if int(os.environ.get('RANK', 0)) == 0:
        print(*args, **kwargs)


def get_distributed_info():
    """
    Get distributed training info from environment variables.

    Returns:
        tuple: (is_distributed, rank, local_rank, world_size)
    """
    if int(os.environ.get('RANK', -1)) != -1:
        # Distributed mode (launched via torchrun)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        return True, rank, local_rank, world_size
    else:
        # Single GPU mode
        return False, 0, 0, 1


def init_distributed(seed: int = 42):
    """
    Initialize distributed training environment (CUDA only).

    Returns:
        tuple: (is_distributed, rank, local_rank, world_size, device)
    """
    assert torch.cuda.is_available(), "CUDA is required for training"

    # Set seeds for reproducibility (config key `seed`; was hardcoded — the LR
    # audit could not measure seed-to-seed variance without a knob)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Precision settings: use TF32 for matmuls
    torch.set_float32_matmul_precision("high")

    # Get distributed info
    is_distributed, rank, local_rank, world_size = get_distributed_info()

    # Setup device
    if is_distributed:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
        print0(f"Distributed training: world_size={world_size}")
    else:
        device = torch.device("cuda")

    return is_distributed, rank, local_rank, world_size, device


def build_system(trunk_cls, config, use_compile=True, head_softcap=15.0, seed=42):
    """
    Assemble the LMSystem (trunk + head) with FSDP and compile.

    WHICH model is the orchestrator's decision — it imports the trunk class and
    passes it here (e.g. `build_system(GPT, gpt_config)`). HOW to assemble it
    is mechanism, and stays here: the order below is load-bearing under FSDP
    (inject before shard, register after — see the assert further down).

    construct -> init -> INJECT behavior family (__class__, before shard) -> shard
    (blocks, trunk, head) -> register head's extra forward methods (after shard) ->
    wrap in LMSystem -> compile the trunk.

    Trunk contract (see GPT for the reference implementation): __init__(config),
    init_weights(), forward(idx, ...) -> hidden [B,T,H], `blocks` (per-layer
    modules for FSDP grouping), `estimate_flops()`, class attr `Config`
    (for checkpoint blueprint recovery). Config must carry n_embd and
    vocab_size — they size the head. init_weights() must initialize EVERY parameter:
    the trunk is built on `meta` and `to_empty`'d before it runs, so a module's default
    constructor init does NOT apply (e.g. a LayerNorm must set its own weight=1 / bias=0,
    or its block silently outputs garbage). This is the LM-shaped assembler
    (trunk + LMHead); a different composition = write your own System
    and assembly (examples over interfaces).

    Args:
        trunk_cls: The trunk class to instantiate (the orchestrator's choice).
        config: trunk_cls's config instance (n_embd/vocab_size size the head)
        use_compile: Whether to compile the trunk (default: True)
        head_softcap: logit softcap for the LM head
        seed: RNG seed for init + training reproducibility (config key `seed`)

    Returns:
        dict: {'system': LMSystem, 'device', 'device_type', 'rank', 'world_size'}
    """
    # Initialize distributed environment
    is_distributed, rank, local_rank, world_size, device = init_distributed(seed=seed)

    # --- Trunk on meta for efficient init ---
    print0(f"Creating trunk: {trunk_cls.__name__}, "
           f"{config.n_layer} layers, {config.n_embd} hidden dims")
    with torch.device("meta"):
        trunk = trunk_cls(config)
    trunk.to_empty(device=device)
    trunk.init_weights()

    # --- Head (un-embedding); classifier init is deterministic zero (no RNG) ---
    with torch.device("meta"):
        head = LMHead(config.n_embd, config.vocab_size, softcap=head_softcap)
    head.to_empty(device=device)
    head.init_weights()

    # --- Inject the behavior family ONCE, BEFORE shard (never at runtime). Liger
    #     fused CE if available, else the naive LMHead path stays. ---
    if LIGER_AVAILABLE and device.type == "cuda":
        LigerLMHead.setup(head)
        print0("Injected LigerLMHead (fused CE) into the head")

    if is_distributed:
        print0("Wrapping trunk + head with FSDP")
        trunk = trunk.to(dtype=torch.bfloat16)
        head = head.to(dtype=torch.bfloat16)
        mesh = init_device_mesh("cuda", (world_size,))
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

        # Trunk: per-block shard groups, then the trunk root (embeddings etc.).
        # `blocks` is the trunk contract — where they live internally is the
        # trunk class's private layout; assembly must not know its paths.
        for block in trunk.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp)
        fully_shard(trunk, mesh=mesh, mp_policy=mp)
        # Head: its own shard group (a separate FSDP root).
        fully_shard(head, mesh=mesh, mp_policy=mp)

        # loss()/type_losses() touch head params OUTSIDE the trunk forward, so each
        # needs its own FSDP window. register_fsdp_forward_method REQUIRES the head to
        # already be an FSDPModule — assert loudly, because otherwise it SILENTLY
        # no-ops and multi-GPU reads sharded params (wrong results, no error).
        assert isinstance(head, FSDPModule), (
            "head must be fully_shard'd BEFORE register_fsdp_forward_method — "
            "otherwise the registration silently no-ops"
        )
        register_fsdp_forward_method(head, "loss")
        register_fsdp_forward_method(head, "type_losses")
    else:
        trunk = trunk.to(dtype=torch.bfloat16)
        head = head.to(dtype=torch.bfloat16)

    system = LMSystem(trunk, head)
    # Record the architecture as an assembly fact for checkpoint
    # self-description. Recorded HERE (not at save time) because fully_shard
    # rewrites the instance's class to FSDP{ClassName} — type(trunk).__name__
    # at save time would lie.
    system.arch = trunk_cls.__name__

    param_count = sum(p.numel() for p in system.parameters()) / 1e9
    print0(f"Model parameters: {param_count:.2f}B")

    # Compile only the trunk (head stays eager — liger is a fused kernel, and an
    # independent frame anyway). The compiled view is held off the module registry.
    if use_compile:
        print0("Compiling trunk (JIT on first forward)...")
        system.set_compiled_trunk(torch.compile(trunk, dynamic=True))

    return {
        'system': system,
        'device': device,
        'device_type': 'cuda',
        'rank': rank,
        'world_size': world_size,
    }


def load_system(checkpoint_dir, trunk_cls=GPT, sequence_len=None,
                use_compile=False, head_softcap=15.0):
    """Assemble a runnable System straight from a self-describing checkpoint.

    The standard inference entry: blueprint from the artifact
    (config_from_meta reads meta.json['model_config'] into trunk_cls.Config),
    assembly through the SAME path training uses, weights via DCP (validated
    against the recorded config), eval() mode. Instantiation choices stay with
    the caller: sequence_len (defaults to the trained value), use_compile,
    head_softcap.

    trunk_cls is a CHECKED default: which code to load is the caller's
    decision, what the checkpoint was trained with is the artifact's recorded
    fact ('model_arch'), and the fact audits the decision — a mismatch raises
    loudly instead of silently building the wrong architecture. Checkpoints
    that predate the arch tag skip the audit (they are all GPT-era).

    Returns the setup dict of build_system plus:
        'gpt_config': the config actually built (trunk_cls.Config instance)
        'meta':       the checkpoint's meta.json contents

    Raises ValueError for checkpoints that predate self-description — build
    the config yourself and use load_model_only then.
    """
    from core.model.checkpoint_manager import (
        config_from_meta, load_metadata, load_model_only)

    meta = load_metadata(checkpoint_dir)
    recorded_arch = meta.get('model_arch')
    if recorded_arch is not None and recorded_arch != trunk_cls.__name__:
        raise ValueError(
            f"{checkpoint_dir} records model_arch='{recorded_arch}' but load_system "
            f"was asked to build '{trunk_cls.__name__}' — pass the matching class: "
            f"load_system(..., trunk_cls=<the {recorded_arch} class>)")

    config_cls = getattr(trunk_cls, 'Config', None)
    if config_cls is None:
        raise TypeError(
            f"{trunk_cls.__name__} has no `Config` class attribute — the trunk "
            f"contract requires it for blueprint recovery (see GPT.Config)")
    config = config_from_meta(checkpoint_dir, config_cls, sequence_len=sequence_len)
    if config is None:
        raise ValueError(
            f"{checkpoint_dir} has no model_config in meta.json (checkpoint predates "
            f"self-description) — construct the config yourself and use load_model_only")
    setup = build_system(trunk_cls, config, use_compile=use_compile,
                         head_softcap=head_softcap)
    load_model_only(checkpoint_dir, setup['system'],
                    rank=setup['rank'], world_size=setup['world_size'])
    setup['system'].eval()
    setup['gpt_config'] = config
    setup['meta'] = meta
    return setup


def setup_model_for_training(gpt_config, **kwargs):
    """Back-compat wrapper: older callers use the GPT-implied single-config
    signature; live callers use build_system(trunk_cls, config)."""
    return build_system(GPT, gpt_config, **kwargs)
