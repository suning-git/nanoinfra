"""train_nano.py -- public training driver for nano-dsv4.

A thin copy-and-adapt of the core text driver (train_text) that touches no core
code. It differs from the plain GPT driver in only two places:
  1. the trunk config is NanoDSV4Config (with its expert / compression / mHC
     hyper-parameters) instead of GPTConfig;
  2. system.loss is wrapped so the indexer's KL auxiliary loss (accumulated
     inside the trunk during the forward pass) is added to the LM loss. The
     sparse-attention top-k selection is non-differentiable, so the lightning
     indexer can only be trained through this auxiliary distillation loss.

nano-dsv4 plugs into the nanoinfra trunk seam (build_system) with zero changes
to core -- that is the whole point of this example.

Usage (single GPU, from repo root):
  python projects/nano_dsv4/train_nano.py --arch dsv4 --max-steps 4000
  python projects/nano_dsv4/train_nano.py --arch dsv4 --dry     # build + param count + one timed step
  python projects/nano_dsv4/train_nano.py --arch gpt --gpt-dim 800 --gpt-heads 8   # a params-matched GPT baseline

Note: this is a reference implementation running in eager mode (~8% MFU on a
5090). It is meant to be read and to verify the architecture trains, not for
throughput-optimized training.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

import modalities.text
from modalities.text.train_text import SOURCE_TYPES, assemble_vocab
from modalities.text import TextEvaluator, get_tokenizer
from core.training.model_setup import build_system, print0
from core.training.trainer import Trainer, create_optimizers
from core.model.gpt import GPT, GPTConfig
from core.data.mixed_dataloader import MixedDataLoader

from arch.nano_dsv4 import NanoDSV4, NanoDSV4Config

CONFIG_DIR = Path(modalities.text.__file__).resolve().parent / "configs"


def make_trunk(arch, seq, layout, gpt_depth=12, gpt_dim=768, gpt_heads=6):
    common = dict(sequence_len=seq, vocab_size=layout.vocab_size,
                  n_token_types=layout.n_token_types)
    if arch == "dsv4":
        return NanoDSV4, NanoDSV4Config(**common)
    if arch == "gpt":
        return GPT, GPTConfig(n_layer=gpt_depth, n_head=gpt_heads,
                              n_kv_head=gpt_heads, n_embd=gpt_dim, **common)
    raise ValueError(arch)


def param_account(trunk, arch):
    """Report total / embedding / non-embedding active params (MoE counts only top-k experts)."""
    total = sum(p.numel() for p in trunk.parameters())
    emb = trunk.wte.weight.numel() + trunk.type_emb.weight.numel() if hasattr(trunk, "wte") \
        else trunk.transformer.wte.weight.numel() + trunk.type_emb.weight.numel()
    inactive = 0
    c = trunk.config
    if arch == "dsv4":
        n_moe = sum(1 for b in trunk.h if getattr(b, "is_moe", True))
        inactive = n_moe * (c.n_routed_experts - c.num_experts_per_tok) * 3 * c.n_embd * c.moe_expert_dim
    print0(f"[param_account] total={total/1e6:.1f}M  embeddings={emb/1e6:.1f}M  "
           f"non-emb total={(total-emb)/1e6:.1f}M  non-emb ACTIVE={(total-emb-inactive)/1e6:.1f}M")
    return total, total - emb - inactive


def eval_schedule(max_steps, n_evals=40, first=20):
    """Log-spaced eval steps, so curves can be compared at matched token counts."""
    if max_steps <= first:
        return list(range(1, max_steps + 1))
    pts = sorted({int(round(math.exp(x))) for x in
                  torch.linspace(math.log(first), math.log(max_steps - 1), n_evals).tolist()})
    return [p for p in pts if p < max_steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["dsv4", "gpt"])
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device-batch-size", type=int, default=16)
    ap.add_argument("--lr", default="3e-4")
    ap.add_argument("--name", default=None)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--gpt-depth", type=int, default=12)
    ap.add_argument("--gpt-dim", type=int, default=768)
    ap.add_argument("--gpt-heads", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="build + param count + one timed fwd/bwd, no training")
    args = ap.parse_args()
    name = args.name or f"nano_{args.arch}"

    overrides = [
        f"sequence_len={args.seq_len}",
        f"device_batch_size={args.device_batch_size}",
        f"optimizer.lr_max={args.lr}",
        f"max_steps={args.max_steps}",
        "use_compile=false",            # nano-dsv4 is eager (data-dependent expert loop / topk)
        "checkpoint.enabled=true",
        f"checkpoint.save_every={args.save_every}",
        f"checkpoint.save_dir=${{oc.env:NANOINFRA_BASE_DIR,./outputs}}/checkpoints/{name}",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train_text", overrides=overrides)
    config = OmegaConf.to_container(cfg, resolve=True)

    tokenizer = get_tokenizer()
    layout, resolver = assemble_vocab(tokenizer)
    trunk_cls, trunk_config = make_trunk(args.arch, args.seq_len, layout,
                                         gpt_depth=args.gpt_depth, gpt_dim=args.gpt_dim,
                                         gpt_heads=args.gpt_heads)
    use_compile = args.arch == "gpt"
    setup = build_system(trunk_cls, trunk_config, use_compile=use_compile,
                         seed=config.get("seed", 42))
    system = setup["system"]
    param_account(system.trunk, args.arch)

    # --- aux-loss wrap (indexer KL): produced in train mode only; pops to None at eval ---
    if hasattr(system.trunk, "pop_aux_loss"):
        orig_loss = system.loss

        def loss_with_aux(batch):
            loss = orig_loss(batch)
            aux = system.trunk.pop_aux_loss()
            return loss if aux is None else loss + aux

        system.loss = loss_with_aux

    sources = []
    for sc in config["data"]["sources"]:
        sc = dict(sc)
        sc.setdefault("sequence_len", args.seq_len)
        sc.setdefault("device", "cuda")
        sources.append(sc)
    dataloader = MixedDataLoader(
        loader_config={"batch_size": args.device_batch_size,
                       "data": {"sequence_len": args.seq_len, "sources": sources}},
        tokenizers={"text": tokenizer, "layout": layout, "control_resolver": resolver},
        source_types=SOURCE_TYPES, resume_state_dict=None,
    )

    if args.dry:
        system.train()
        batch = next(iter(dataloader))
        for i in range(3):  # one warmup, two timed
            t0 = time.perf_counter()
            loss = system.loss(batch)
            loss.backward()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            system.zero_grad(set_to_none=True)
            print0(f"[dry] fwd+bwd #{i}: loss={loss.item():.4f}  {dt*1e3:.0f} ms  "
                   f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        flops = system.estimate_flops()
        tok = args.device_batch_size * args.seq_len
        print0(f"[dry] est FLOPs/token={flops/1e6:.1f}M  tokens/microbatch={tok}")
        return

    optimizers = create_optimizers(system, config["optimizer"], world_size=setup["world_size"])

    eval_cfg = dict(config.get("evaluation", {}).get("text", {}))
    if args.max_steps > 0:
        eval_cfg["eval_at"] = eval_schedule(args.max_steps)
    evaluators = [TextEvaluator(eval_cfg, args.device_batch_size, args.seq_len)]

    trainer = Trainer(system=system, optimizers=optimizers, dataloader=dataloader,
                      config=config, rank=setup["rank"], world_size=setup["world_size"],
                      debug_tokenizer=tokenizer, evaluators=evaluators)
    print0(f"Starting {name}: arch={args.arch} steps={trainer.max_steps} "
           f"seq={args.seq_len} dbs={args.device_batch_size}")
    trainer.train()


if __name__ == "__main__":
    main()
