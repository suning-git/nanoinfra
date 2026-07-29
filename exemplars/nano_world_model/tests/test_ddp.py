"""
test_ddp.py — the gradients two GPUs produce are the right ones.

Run:
    torchrun --nproc_per_node=2 --standalone \
        -m exemplars.nano_world_model.tests.test_ddp

CHECK 1 — NanoDDP correctness (bitwise).
    NanoDDP fires one all_reduce per transformer block from a parameter hook, DURING
    backward, on a side stream, so communication overlaps the remaining compute. The
    reference is the obvious thing it replaces: finish the whole backward, then
    all_reduce every gradient. Those must agree BITWISE — overlapping is a schedule
    change, not a numerical one. This is the check that has to pass before any of the
    performance argument matters.

CHECK 2 — data-parallel math.
    Two ranks each holding half the batch must produce the same gradient as one rank
    doing both halves with gradient accumulation. The two configurations are made
    comparable in the one way that IS under our control: both consume the SAME two
    micro-batches, and the diffusion noise is seeded from the rows in a micro-batch
    rather than from the step or the rank (see dataset.py), so identical
    micro-batches get identical masks.

    This check is written with a tolerance rather than as an equality, because in
    general it must be: floating-point addition is not associative, and the paths
    sum in different orders — NCCL computes (g_A + g_B)/2 inside the collective,
    accumulation computes g_A/2 + g_B/2 locally.

    At WORLD SIZE 2 it nevertheless comes out bitwise (measured: max rel-err 0.0),
    and that is a fact about 2, not luck: halving is exact in binary floating point,
    so g_A/2 + g_B/2 and (g_A + g_B)/2 involve one addition with the same operand
    magnitudes and therefore round identically. Do not carry the expectation to
    world size 3 — the tolerance is the real contract.

Both checks use the real row layout, the real objective, and real cache rows — a
synthetic model would not exercise the head's compiled cross-entropy, which is where
a per-token weight can silently go missing.
"""

import os

import torch
import torch.distributed as dist

from exemplars.nano_world_model import spec

spec.pin_tokenizer()

from core.model.gpt import GPT, GPTConfig                          # noqa: E402

from core.parallel import (                                        # noqa: E402
    NanoDDP, block_buckets, sync_gradients)
from core.training.model_setup import (                            # noqa: E402
    build_system, compile_blocks)
from exemplars.nano_world_model import row_layout, train_wm   # noqa: E402
from exemplars.nano_world_model.block_diffusion import BlockDiffusion  # noqa: E402
from exemplars.nano_world_model.dataset import VideoRowDataset  # noqa: E402

FRAMES, RES = 17, 128
MICRO_ROWS = 4          # rows per rank per micro-batch
DEPTH, DIM, HEADS = 4, 256, 2

failures = []


def check(name, ok, detail=""):
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}",
              flush=True)
    if not ok:
        failures.append(name)


def grads(system):
    return {n: p.grad.detach().clone() for n, p in system.named_parameters()
            if p.grad is not None}


def compare(a, b):
    """(n_mismatched, max relative error) over matching keys."""
    assert set(a) == set(b)
    bad, worst = 0, 0.0
    for k in a:
        if not torch.equal(a[k], b[k]):
            bad += 1
            denom = b[k].abs().max().clamp_min(1e-12)
            worst = max(worst, ((a[k] - b[k]).abs().max() / denom).item())
    return bad, worst


def main():
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    assert world == 2, "this test compares 2 ranks against 1"

    layout, resolver = train_wm.assemble_vocab()
    geom = spec.clip_geometry(FRAMES, RES)
    rows = train_wm.build_row_layout(layout, resolver, geom)
    row_layout.use_compiled_flex_attention()

    cfg = GPTConfig(sequence_len=2 * rows.row_len, vocab_size=layout.vocab_size,
                    n_layer=DEPTH, n_head=HEADS, n_kv_head=HEADS, n_embd=DIM,
                    n_token_types=layout.n_token_types)
    setup = build_system(GPT, cfg, use_compile=False, seed=0, parallel="ddp")
    system, device = setup["system"], setup["device"]
    rows.install_mirror_rope(system.trunk)
    compile_blocks(system.trunk)     # the configuration we actually train in

    objective = BlockDiffusion(rows, layout, resolver.resolve(spec.MASK_SLOT),
                               spec.T_MIN, spec.T_MAX, device)

    # Two fixed micro-batches from the frozen val split, one per rank. Both ranks
    # build BOTH so rank-local code can compute the single-process reference.
    ds = VideoRowDataset(spec.cache_dir(FRAMES, RES), "val")
    codes, actions = ds.take(2 * MICRO_ROWS)
    micro = [(rows.assemble(codes[i * MICRO_ROWS:(i + 1) * MICRO_ROWS],
                            actions[i * MICRO_ROWS:(i + 1) * MICRO_ROWS]).to(device),
              1000 + i)                                    # (idx, noise_seed)
             for i in range(2)]

    ddp = NanoDDP([list(system.head.parameters())] + block_buckets(system.trunk),
                  module=system)
    autocast = torch.autocast("cuda", torch.bfloat16)
    my_idx, my_seed = micro[rank]

    if rank == 0:
        print(f"\nmodel d{DEPTH}/{DIM}, row {rows.row_len}, "
              f"{len(ddp.buckets)} buckets, {MICRO_ROWS} rows/rank\n")

    # --- the path under test: overlapped per-bucket all_reduce during backward ---
    # Written the way the Trainer writes it — through the bracket, not the raw halves.
    system.zero_grad(set_to_none=True)
    with sync_gradients(ddp, enabled=True):
        with autocast:
            objective.loss(system, my_idx, my_seed).backward()
    g_ddp = grads(system)

    # The bracket must be exactly the raw halves, or the tests below prove nothing
    # about what the Trainer actually runs.
    system.zero_grad(set_to_none=True)
    ddp.reset(sync=True)
    with autocast:
        objective.loss(system, my_idx, my_seed).backward()
    ddp.finalize()
    bad, _ = compare(g_ddp, grads(system))
    check("sync_gradients() == reset()/finalize(), bitwise", bad == 0)

    # --- reference 1: plain backward, then all_reduce everything -----------------
    system.zero_grad(set_to_none=True)
    ddp.reset(sync=False)                        # hooks inert; no overlap, no buckets
    with autocast:
        objective.loss(system, my_idx, my_seed).backward()
    for p in system.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    g_naive = grads(system)

    bad, worst = compare(g_ddp, g_naive)
    check("NanoDDP == backward-then-all_reduce, bitwise", bad == 0,
          f"{len(g_ddp)} tensors" if bad == 0 else f"{bad} differ, max rel-err {worst:.3e}")

    # --- reference 2: one process, gradient accumulation over BOTH micro-batches --
    system.zero_grad(set_to_none=True)
    for idx_m, seed_m in micro:
        ddp.reset(sync=False)                    # local only — this rank is "the" GPU
        with autocast:
            (objective.loss(system, idx_m, seed_m) / 2).backward()
    g_accum = grads(system)

    bad, worst = compare(g_ddp, g_accum)
    check("2 ranks == 1 rank with grad_accum=2, to fp tolerance",
          worst < 1e-5,
          f"{bad}/{len(g_ddp)} tensors differ (expected), max rel-err {worst:.3e}")

    # Every NanoDDP holds hooks on the parameters it was given, so a second one over
    # the same model would ALSO count this backward and raise. Retire this one before
    # building the throwaway objects below — which is what close() is for.
    ddp.close()

    # --- the unused-parameter guard, the reason NanoDDP is a class ---------------
    # Buckets are formed while everything still needs gradients, THEN block 0 is
    # frozen — so bucket 0 holds real parameters that never receive a gradient. That
    # is the actual failure mode (a bucket that never completes, so its all_reduce
    # never fires and the replicas drift apart in silence), not an empty bucket.
    guard_ddp = NanoDDP(block_buckets(system.trunk), module=system.trunk)
    frozen = system.trunk.blocks[0]
    saved = [p.requires_grad for p in frozen.parameters()]
    for p in frozen.parameters():
        p.requires_grad_(False)
    system.zero_grad(set_to_none=True)
    guard_ddp.reset(sync=True)
    with autocast:
        objective.loss(system, my_idx, my_seed).backward()
    raised = False
    try:
        guard_ddp.finalize()
    except RuntimeError as e:
        raised = "received no gradient" in str(e)
    check("guard: a bucket with no gradient raises instead of silently desyncing", raised)
    guard_ddp.close()
    for p, s in zip(frozen.parameters(), saved):
        p.requires_grad_(s)

    # --- forgetting to close a step must be loud, not silent ---------------------
    # This is the failure that costs a training run: the reduced gradients live in a
    # transient buffer until finalize() copies them into .grad, so skipping it means
    # stepping on LOCAL gradients with nothing to indicate it.
    loose = NanoDDP(block_buckets(system.trunk), module=system.trunk)
    system.zero_grad(set_to_none=True)
    loose.reset(sync=True)
    with autocast:
        objective.loss(system, my_idx, my_seed).backward()
    caught = False
    try:
        loose.reset(sync=True)            # next step, without finalizing the last one
    except RuntimeError as e:
        caught = "never written back" in str(e)
    check("skipping finalize() raises on the next step", caught)
    loose.close()

    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 60)
        print(f"DDP GATES {'FAILED: ' + str(failures) if failures else 'PASSED'}")
    dist.destroy_process_group()
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
