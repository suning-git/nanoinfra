"""
train_text_ddp — PHASE 4: does NanoDDP hold up on a model it was not written for?

NanoDDP was designed against a video world model. Something that only ever ran on
one model family has not earned the name "core capability", so before proposing it
for core it has to work on the other family in this repo: text pretraining, whose
loss path is completely different (Liger's fused cross-entropy over a 96k vocab, no
diffusion, no flex-attention mask, a MixedDataLoader instead of a memmap).

This file is an ORCHESTRATOR, not a copy of one. It imports the real text
assembly — `modalities.text.train_text`'s vocab assembly, source resolution and
evaluator construction, plus its config — and changes exactly three wires:

    build_system(..., parallel="ddp")   replicate instead of shard
    compile_blocks(trunk)               per-block instead of whole-graph
    Trainer(..., ddp=NanoDDP(...))      the seam under review

modalities/text/train_text.py itself is untouched. That is the point: if the patched
core is right, driving the text stack through it costs an orchestrator, not a fork.

Three things it answers, the three the plan asks:

  --check       (1) Do the gradients match the obvious reference, on THIS model?
                    Bitwise, same test as the video gate, different loss path.
  (default)     (2) Does the API fit core's standard Trainer? This runs real text
                    pretraining through it end to end — data, eval, checkpoint.
  --scale       (3) Does it hold at size? Throughput and peak memory at ~1.4B.

Run:
    torchrun --nproc_per_node=2 --standalone \
        -m modalities.tests.train_text_ddp --check
    torchrun --nproc_per_node=2 --standalone \
        -m modalities.tests.train_text_ddp -- max_steps=200
"""

import os
import sys
import time

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from core.data.mixed_dataloader import MixedDataLoader
from core.model.gpt import GPT, GPTConfig
from core.training.trainer import create_optimizers
from core.utils import print0

# Importing the text orchestrator also registers its ${eval:} config resolver and
# gives us its assembly helpers — we drive the real thing, not a lookalike.
from modalities.text import get_tokenizer
from modalities.text.streams import build_evaluators, resolve_sources
import modalities.text.train_text as text_orchestrator

from core.parallel import NanoDDP, block_buckets, sync_gradients
from core.training.model_setup import build_system, compile_blocks
from core.training.trainer import Trainer

CHECK = "--check" in sys.argv
SCALE = "--scale" in sys.argv
sys.argv = [a for a in sys.argv if a not in ("--check", "--scale", "--")]


def gradient_equivalence(system, dataloader, ddp, world):
    """The video gate's Gate C, on the text model: per-bucket overlapped all_reduce
    during backward must equal backward-then-all_reduce, BITWISE.

    Each rank uses its own batch (that is the point of data parallel), so the two
    paths being compared are on the same rank with the same data — the only
    difference is when and how the collective is issued."""
    batch = next(iter(dataloader))
    autocast = torch.autocast("cuda", torch.bfloat16)

    def run(sync_mode):
        system.zero_grad(set_to_none=True)
        ddp.reset(sync=sync_mode)
        with autocast:
            system.loss(batch).backward()
        if sync_mode:
            ddp.finalize()
        else:
            for p in system.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        return {n: p.grad.detach().clone() for n, p in system.named_parameters()
                if p.grad is not None}

    g_ddp, g_naive = run(True), run(False)
    bad, worst = 0, 0.0
    for k in g_ddp:
        if not torch.equal(g_ddp[k], g_naive[k]):
            bad += 1
            worst = max(worst, ((g_ddp[k] - g_naive[k]).abs().max()
                                / g_naive[k].abs().max().clamp_min(1e-12)).item())
    ok = bad == 0
    print0(f"\n  {'PASS' if ok else 'FAIL'}  NanoDDP == backward-then-all_reduce on the "
           f"TEXT model, bitwise  ({len(g_ddp)} tensors"
           f"{'' if ok else f', {bad} differ, max rel-err {worst:.3e}'})")
    return ok


def throughput(system, dataloader, ddp, tokens_per_step, steps=12, warmup=4):
    """Steps per second and peak memory in the real training configuration."""
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-5)
    autocast = torch.autocast("cuda", torch.bfloat16)
    it = iter(dataloader)
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        with sync_gradients(ddp, enabled=True):
            with autocast:
                system.loss(next(it)).backward()
        optimizer.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / (steps - warmup)
    print0(f"\n  {tokens_per_step / dt / 1e3:.1f}k tok/s/rank | {dt * 1000:.0f} ms/step | "
           f"peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")


@hydra.main(version_base=None, config_path="../text/configs",
            config_name="train_text")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    if SCALE:
        # ~520M. Sized to what fits BESIDE whatever else is on the box, not to the
        # largest interesting model: the 1.4B point (backward 384->312ms with
        # overlap, 19.1GB peak, seq 8192) was measured on a free machine in the
        # session that wrote NanoDDP and is not re-measured here. What this adds is
        # the same question asked of the TEXT family at a size where the buckets are
        # large enough for communication to matter. Override on the CLI to go bigger.
        config["model"].update(depth=20, dim=1280, n_head=10, n_kv_head=10)
        config["sequence_len"] = 1024
        config["device_batch_size"] = 2
        config["use_compile"] = True

    print0("=" * 80)
    print0("train_text_ddp — text pretraining through the PATCHED core (replicated DP)")
    print0("=" * 80)

    tokenizer = get_tokenizer()
    layout, control_resolver = text_orchestrator.assemble_vocab(tokenizer)

    model_config = config["model"]
    gpt_config = GPTConfig(
        sequence_len=config["sequence_len"], vocab_size=layout.vocab_size,
        n_layer=model_config["depth"], n_head=model_config["n_head"],
        n_kv_head=model_config["n_kv_head"], n_embd=model_config["dim"],
        n_token_types=layout.n_token_types)
    setup = build_system(GPT, gpt_config, use_compile=False,
                         seed=config.get("seed", 42), parallel="ddp")
    system, rank, world = setup["system"], setup["rank"], setup["world_size"]
    if config.get("use_compile", True):
        compile_blocks(system.trunk)

    tokenizers = {"text": tokenizer, "layout": layout,
                  "control_resolver": control_resolver}
    sources = resolve_sources(config["data"], config["sequence_len"], device="cuda")
    dataloader = MixedDataLoader(
        loader_config={"batch_size": config["device_batch_size"],
                       "data": {"sequence_len": config["sequence_len"],
                                "sources": sources}},
        tokenizers=tokenizers, source_types=text_orchestrator.SOURCE_TYPES,
        resume_state_dict=None)

    ddp = None
    if world > 1:
        ddp = NanoDDP([list(system.head.parameters())] + block_buckets(system.trunk),
                      module=system)
        print0(f"NanoDDP over {len(ddp.buckets)} buckets, world {world}")

    tokens_per_step = config["device_batch_size"] * config["sequence_len"]
    if CHECK:
        ok = gradient_equivalence(system, dataloader, ddp, world)
        throughput(system, dataloader, ddp, tokens_per_step)
        dist.barrier()
        dist.destroy_process_group()
        raise SystemExit(0 if ok else 1)
    if SCALE:
        throughput(system, dataloader, ddp, tokens_per_step)
        dist.barrier()
        dist.destroy_process_group()
        raise SystemExit(0)

    # The real thing: core's Trainer, the ddp seam, real data, real evaluation.
    config["total_batch_size"] = tokens_per_step * world
    optimizers = create_optimizers(system, config["optimizer"], world_size=world)
    evaluators = build_evaluators(config, tokenizers, text_orchestrator.SOURCE_TYPES,
                                  config["device_batch_size"], config["sequence_len"],
                                  device="cuda")
    Trainer(system=system, optimizers=optimizers, dataloader=dataloader, config=config,
            rank=rank, world_size=world, evaluators=evaluators, ddp=ddp).train()
    print0("\n✓ text pretraining ran to completion on the patched core")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
