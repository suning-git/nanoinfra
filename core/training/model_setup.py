"""
Model setup for distributed training.

Provides hardware deployment layer for:
- Distributed initialization (multi-GPU)
- Model creation with FSDP or replicated (DDP) placement
- Trunk compilation MECHANISMS (whole trunk, or per block); whether to use one,
  and which, is the caller's decision after assembly

Note: nanoinfra is designed for CUDA training only.

Two things here serve multi-GPU placement specifically.

1. `build_system(..., parallel=...)` chooses how the model is placed. "fsdp"
shards parameters per block; "ddp" replicates them and leaves gradient
synchronization to the caller (core/parallel/nano_ddp.py). Sharding is the right
default when the model does not fit; when it does, sharding costs an all-gather per
block per step and buys nothing, because at that size the memory bottleneck is
activations rather than parameters (measured on 2x5090, 233M params: FSDP 93.6k
tok/s / 30GB against replicated 106.2k tok/s / 19GB).

2. `build_system` does NOT compile the trunk. Two mechanisms sit beside it and the
orchestrator calls ONE of them after assembly, once it knows world_size and how it
synchronizes gradients: `compile_system_trunk(system, ...)`
compiles the trunk as a whole and registers the wrapper with the System;
`compile_blocks(trunk)` compiles each block separately, in place. They are free
functions rather than build_system flags because WHETHER to compile, and which way,
follows from facts this function does not know — but they are mechanism, so they
belong beside the other assembly mechanics rather than copy-pasted into every
orchestrator.
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


def compile_blocks(trunk, dynamic=None):
    """Compile each transformer block separately, in place, instead of the trunk as
    a whole. Use with replicated data parallel (why: below). build_system does not
    compile the trunk; the orchestrator calls this OR compile_system_trunk, not both
    (only the per-block-then-whole order is detected — this function sees the trunk,
    not the System, so it cannot tell that a whole-trunk wrapper is already registered).

    `dynamic` is forwarded to torch.compile. None (default) lets dynamo decide;
    False pins static shapes, which is worth a few percent on architectures whose
    guards would otherwise re-specialize (measured on nano-dsv4).

    A whole-graph compile of a DENSE trunk makes AOTAutograd finalize every
    parameter's gradient at the very end of backward (PyTorch #109774), so every
    all_reduce piles up after the compute instead of overlapping it. Per-block compile
    leaves an eager seam at each block boundary, which is where the gradient hooks
    fire. Measured on dense GPT at 1.4B/seq8192: backward 384ms -> 312ms, compute
    stream never stalling.

    This does NOT generalize to every architecture, and it is worth checking rather
    than assuming. A trunk that already graph-breaks has the seams anyway: nano-dsv4
    compiles to 20 graphs / 19 breaks (mostly its router's data-dependent bincount),
    and there per-block vs whole-trunk is 78.0 vs 78.2 ms/step on two ranks — inside
    the noise. `torch._dynamo.explain(trunk.forward)` answers this in one call.
    """
    # Compile each block's FORWARD rather than replacing the block with an
    # OptimizedModule. Replacing it registers the compiled wrapper as a child and
    # inserts `_orig_mod.` into every one of that block's state_dict keys — which is
    # exactly what LMSystem.set_compiled_trunk goes out of its way to avoid, and what
    # breaks `isinstance(block, Block)` and any consumer of a raw state_dict. Binding
    # the compiled callable as an instance attribute leaves the module registry, the
    # keys, and the types untouched. (Verified: identical keys, bitwise-identical
    # outputs, and dynamo does compile it.)
    for block in trunk.blocks:
        if getattr(block.forward, "_torchdynamo_orig_callable", None) is not None:
            raise RuntimeError(
                "compile_blocks: these blocks are already compiled. Calling it twice "
                "would wrap the wrapper; call it once.")
        block.forward = torch.compile(block.forward, dynamic=dynamic)
    print0(f"trunk compile: per-block, dynamic={dynamic}")   # the action, not a status
    return trunk


def compile_system_trunk(system, **compile_kwargs):
    """Compile the trunk as a whole and register the wrapper for the System's hot
    path (LMSystem._run_trunk). The thin counterpart of compile_blocks: after
    build_system returns, the orchestrator calls one of the two — or neither.

    `compile_kwargs` go to torch.compile unchanged (dynamic=..., mode=..., ...): the
    text exemplars use dynamic=True, nano-dsv4 dynamic=False, the bench mode=...

    Only consumed by loss paths that go through `_run_trunk`. A System whose loss
    calls `self.trunk(...)` or loops over the blocks itself (the block-diffusion
    Systems) never sees the wrapper — those use compile_blocks, which rebinds each
    block's own forward and so travels with the modules wherever they are reused.

    Refuses a trunk whose blocks are already compiled per block (stacking the two
    mechanisms would trace a compiled callable inside a compiled callable); a second
    whole-trunk registration is refused by set_compiled_trunk itself. The reverse
    order — this first, compile_blocks after — is NOT detected (compile_blocks only
    sees the trunk); the rule is still one mechanism per trunk.
    """
    for block in system.trunk.blocks:
        if getattr(block.forward, "_torchdynamo_orig_callable", None) is not None:
            raise RuntimeError(
                "compile_system_trunk: the blocks are already compiled per block "
                "(compile_blocks). Pick ONE mechanism for a trunk.")
    print0(f"trunk compile: whole, {compile_kwargs or 'torch.compile defaults'}")
    system.set_compiled_trunk(torch.compile(system.trunk, **compile_kwargs))
    return system


def build_system(trunk_cls, config, head_softcap=15.0, seed=42,
                 parallel="fsdp", head_ce="naive"):
    """
    Assemble the LMSystem (trunk + head) with FSDP. The trunk is NOT compiled here.

    WHICH model is the orchestrator's decision — it imports the trunk class and
    passes it here (e.g. `build_system(GPT, gpt_config)`). HOW to assemble it
    is mechanism, and stays here: the order below is load-bearing under FSDP
    (inject before shard, register after — see the assert further down).

    construct -> init -> INJECT behavior family (__class__, before shard) -> shard
    (blocks, trunk, head) -> register head's extra forward methods (after shard) ->
    wrap in LMSystem. Compiling the trunk is the caller's NEXT line, once it knows
    world_size and its gradient-sync mechanism: compile_system_trunk (whole) or
    compile_blocks (per block).

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
        head_ce: which CE implementation the head uses: eager PyTorch, a Liger fused
            kernel, or compiled PyTorch. This is one choice, not separate flags for
            kernel selection and compilation. Compiling around liger raised an
            InductorError on the measured stack. The head is a real fraction
            of a SMALL model's step (the un-embedding outweighs a shallow trunk), so
            this is a throughput knob, not a detail. Measured on one RTX 5090,
            vocab 32768, seq 1024, batch 32 — MFU for naive/liger/compiled:
                depth 2   6.6 / 16.3 / 33.9 %
                depth 6  26.5 / 50.6 / 64.3 %
                depth 8  37.5 / 58.3 / 70.0 %
            One shape for all three depths, because the batch is half the answer and
            a series measured at two shapes is not a series. The spread collapses
            with depth because the head's share of the parameters does: 48% at d2,
            29% at d8. Peak memory ranks the other way round — liger lowest, naive
            highest, compiled between.
            "naive" (default) F.cross_entropy over materialized [B,T,V] fp32 logits.
                       The default is a POLICY, not a theorem:
                       it depends on no optional package (which environment a run
                       lands on must never decide its loss trajectory), it is the
                       plain numeric reference for the other two, and it starts
                       no compilation — in line with compiling being the caller's
                       decision. A project may well make liger or compiled ITS default.
            "liger"    fused linear+CE in token chunks, avoiding the full [B,T,V]
                       allocation. Lowest peak memory in the measured configurations.
                       Errors if liger_kernel is absent rather than falling back —
                       an arm that silently becomes another arm is not an arm.
            "compiled" the same math as naive in one dynamo frame. Measured peak
                       2.16 / 1.20 / 0.76 GiB at N = 32k / 16k / 8k tokens, V = 32768,
                       i.e. about N·V·2 bytes plus a constant — consistent with the bf16
                       projection materializing and the fp32 logits / softmax
                       intermediates not (inferred from the peaks; the generated code
                       was not inspected). Holds at V = 32768 (1.05x); at V = 96787 the
                       peak is about 2x N·V·2 (measured 2026-09-11) — a vocab-dependent
                       reading, not a formula. Fastest at every depth measured, and
                       numerically the closest to naive.
                       Works under all three placements (single / ddp / fsdp).
            `head.loss_per_token` (unreduced CE, for an objective that weights tokens
            itself) follows the same choice, naive or compiled; the liger arm refuses
            it — see LigerLMHead.loss_per_token.
        head_softcap: logit softcap for the LM head
        seed: RNG seed for init + training reproducibility (config key `seed`)
        parallel: which MULTI-DEVICE strategy to use. Ignored on one device — whether
            this process is one of many is decided by the launcher (the RANK
            environment variable: `python` versus `torchrun`), not by this argument.
            So the three placements are single-device / "fsdp" / "ddp", but only the
            last two are a choice made here.
            "fsdp" (default) shards parameters per block — use when the model does
                not fit, and it stays the default so existing projects are unchanged.
            "ddp" replicates the model on every rank and installs NO gradient
                synchronization: that is the caller's, via core/parallel/nano_ddp.py.
                Faster and smaller whenever the model fits, because at that point the
                memory bottleneck is activations rather than parameters.

    Returns:
        dict: {'system': LMSystem, 'device', 'device_type', 'rank', 'world_size'}
    """
    # Validated before anything expensive happens: a typo should not cost a process
    # group and a full model build first.
    if parallel not in ("fsdp", "ddp"):
        raise ValueError(f"unknown parallel={parallel!r}; expected 'fsdp' or 'ddp'")
    if head_ce not in ("naive", "liger", "compiled"):
        raise ValueError(f"unknown head_ce={head_ce!r}; expected "
                         "'naive', 'liger' or 'compiled'")

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

    # --- Inject the behavior family ONCE, BEFORE shard (never at runtime). ---
    if head_ce == "liger":
        if not LIGER_AVAILABLE:
            raise RuntimeError("head_ce='liger' but liger_kernel is not installed "
                               "(pip install -e '.[liger]')")
        LigerLMHead.setup(head)
        print0("Head: liger fused CE")
    else:
        print0(f"Head: naive CE{' (loss compiled)' if head_ce == 'compiled' else ''}")

    # --- Placement. Exactly three modes, and the branch says so ---------------
    # Note the two are on different axes: whether this is one device or many comes
    # from the LAUNCHER (the RANK environment variable, i.e. python vs torchrun) and
    # is not something `parallel` can choose; `parallel` picks between the multi-device
    # strategies and is ignored on one device.
    trunk = trunk.to(dtype=torch.bfloat16)
    head = head.to(dtype=torch.bfloat16)

    if not is_distributed:
        pass                       # one device: there is nothing to place
    elif parallel == "fsdp":
        print0("Wrapping trunk + head with FSDP")
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

        # loss()/loss_per_token()/type_losses() touch head params OUTSIDE the trunk
        # forward, so each needs its own FSDP window. register_fsdp_forward_method
        # REQUIRES the head to already be an FSDPModule — assert loudly, because
        # otherwise it SILENTLY no-ops and multi-GPU reads sharded params (wrong
        # results, no error).
        assert isinstance(head, FSDPModule), (
            "head must be fully_shard'd BEFORE register_fsdp_forward_method — "
            "otherwise the registration silently no-ops"
        )
        register_fsdp_forward_method(head, "loss")
        register_fsdp_forward_method(head, "loss_per_token")
        register_fsdp_forward_method(head, "type_losses")
    elif parallel == "ddp":
        # Replicate. Deliberately nothing else: the replicas already agree because
        # every rank ran the same init_weights() under the same seed, so no broadcast
        # is needed, and gradient synchronization is the caller's — see
        # core/parallel/nano_ddp.py, which the orchestrator constructs and hands to
        # the Trainer. "ddp" here means only "do not shard".
        print0(f"Replicated data parallel across {world_size} ranks "
               f"(gradient sync is the caller's job — see core/parallel/nano_ddp.py)")
    else:
        # Unreachable: `parallel` was validated before the process group was created.
        # Kept so this branch is exhaustive BY CONSTRUCTION rather than by trusting a
        # check forty lines earlier — a new mode added above must be handled here too.
        raise ValueError(f"unhandled parallel={parallel!r}")

    system = LMSystem(trunk, head)
    # Record the architecture as an assembly fact for checkpoint
    # self-description. Recorded HERE (not at save time) because fully_shard
    # rewrites the instance's class to FSDP{ClassName} — type(trunk).__name__
    # at save time would lie.
    system.arch = trunk_cls.__name__

    param_count = sum(p.numel() for p in system.parameters()) / 1e9
    print0(f"Model parameters: {param_count:.2f}B")

    # The trunk is NOT compiled here. Whether, and whole vs per block, is the
    # orchestrator's decision once it knows world_size and how it synchronizes
    # gradients: compile_system_trunk / compile_blocks, after this returns (H9).

    if head_ce == "compiled":
        # Bound as an INSTANCE attribute, which is also what keeps it out of
        # state_dict (a compiled bound method is not a Module). Note this DOES reach
        # eval: the fused evaluator calls head.loss directly. head.type_losses is the
        # one path pinned back to the eager class method (see heads.py), so a per-type
        # eval does not drag eval-shaped batches into dynamo.
        head.loss = torch.compile(head.loss)
        # The unreduced entry sees a different N on every call (block diffusion masks
        # a random number of positions per batch, each position its own Bernoulli
        # draw), so it is dynamic from the start rather than one static graph, a
        # recompile, then dynamic.
        head.loss_per_token = torch.compile(head.loss_per_token, dynamic=True)

    return {
        'system': system,
        'device': device,
        'device_type': 'cuda',
        'rank': rank,
        'world_size': world_size,
    }


def load_system(checkpoint_dir, trunk_cls=GPT, sequence_len=None,
                head_softcap=15.0, head_ce="naive"):
    """Assemble a runnable System straight from a self-describing checkpoint.

    The standard inference entry: blueprint from the artifact
    (config_from_meta reads meta.json['model_config'] into trunk_cls.Config),
    assembly through the SAME path training uses, weights via DCP (validated
    against the recorded config), eval() mode. Instantiation choices stay with
    the caller: sequence_len (defaults to the trained value), head_ce, head_softcap.
    The trunk comes back eager; compiling it, if wanted, is the caller's next line.

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
    setup = build_system(trunk_cls, config, head_ce=head_ce, head_softcap=head_softcap)
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
