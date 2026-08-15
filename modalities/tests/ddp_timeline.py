"""
ddp_timeline.py — where the time in ONE backward actually goes, layer by layer.

Two arms, deliberately the SAME setup (same model, same batch, same per-block
compile, one micro-batch = one backward, no gradient accumulation):

  1 GPU   `python -m modalities.tests.ddp_timeline`
          per block: when its backward starts and ends, relative to T+0 = the
          instant `loss.backward()` is entered.

  2 GPU   `torchrun --nproc_per_node=2 --standalone -m modalities.tests.ddp_timeline`
          the same, PLUS for each bucket when its all_reduce starts and ends on
          the communication stream — the overlap NanoDDP exists to create, seen
          directly rather than inferred from a step time.

HOW IT IS MEASURED. CUDA events (`torch.cuda.Event(enable_timing=True)`), never
wall clock: the CPU has already returned from `backward()` long before the GPU is
done, so any host-side timer measures launch, not execution. Every number below is
`t0.elapsed_time(event)` in milliseconds, where t0 is an event recorded on the
compute stream immediately before `backward()`. Event timestamps are global to the
device, so compute-stream and comm-stream events are directly comparable — which is
the whole point, since the question is whether the two overlap.

WHERE THE COMPUTE EVENTS COME FROM. Not module backward hooks (which rewrite the
autograd graph) but a hook on the boundary TENSOR itself: block i's input is
literally block i-1's output object, so a `register_hook` there fires exactly when
the gradient crosses that seam. Block i's "end" and block i-1's "start" are
therefore the same instant, recorded twice — a built-in consistency check, and the
reason the compute column is contiguous.

WHERE THE COMM EVENTS COME FROM. `TimedNanoDDP` below overrides `_fire` with a
byte-faithful copy of core's, plus two events around the `all_reduce` INSIDE the
comm-stream context. `start` therefore records when the collective actually begins
on the GPU — after `wait_stream` has let the compute it depends on finish, and
after any earlier bucket still occupying that stream. That is the honest answer to
"when did this layer's communication start", as opposed to when it was issued.

Bucket layout (block_buckets, plus the head at the front — see nano_ddp.py):
    bucket 0  = LM head          (completes before any block)
    bucket 1  = block 11         (backward runs output-inwards, so the LAST block
    ...                           finishes first, and buckets are issued in that
    bucket 12 = block 0           order)
    bucket 13 = root             (embeddings — finalized last, nothing left to
                                  hide it behind)
"""
import argparse
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors as _flatten

from core.data.mixed_dataloader import MixedDataLoader
from core.model.gpt import GPT, GPTConfig
from core.parallel import NanoDDP, block_buckets, sync_gradients
from core.training.model_setup import build_system, compile_blocks
from core.utils import print0

from modalities.text import get_tokenizer
from modalities.text.streams import resolve_sources
import modalities.text.train_text as text_orchestrator

# The d12 recipe constants, inlined rather than imported from
# exemplars/text_pretrain/spec.py: this probe lives in the modalities layer, and
# modalities must not depend on exemplars (the dependency points the other way).
DEPTH = 12
SEED = 42

REPO = Path(__file__).resolve().parents[2]


def event():
    return torch.cuda.Event(enable_timing=True)


class BlockTimer:
    """Compute start/end for every transformer block, from tensor hooks on the
    seams between blocks. `arm()` before a step you want measured; warmup steps
    with slots=None register nothing and cost nothing."""

    def __init__(self, blocks):
        self.n = len(blocks)
        self.slots = None
        for i, block in enumerate(blocks):
            block.register_forward_pre_hook(self._on_input(i))
            block.register_forward_hook(self._on_output(i))

    def arm(self):
        self.slots = [{"start": event(), "end": event()} for _ in range(self.n)]

    def _on_input(self, i):
        # grad w.r.t. this block's INPUT is the last thing its backward produces
        def hook(_mod, args):
            x = args[0]
            if self.slots is not None and x.requires_grad:
                slot = self.slots[i]
                x.register_hook(lambda _g: slot["end"].record())
        return hook

    def _on_output(self, i):
        # grad w.r.t. this block's OUTPUT is what its backward starts from
        def hook(_mod, _args, out):
            if self.slots is not None and out.requires_grad:
                slot = self.slots[i]
                out.register_hook(lambda _g: slot["start"].record())
        return hook


class TimedNanoDDP(NanoDDP):
    """core's NanoDDP with two CUDA events around each bucket's collective.

    `_fire` is copied rather than wrapped because the events have to sit INSIDE
    the `torch.cuda.stream(comm_stream)` context, between the wait and the
    all_reduce — there is no seam to wrap from outside. Keep it byte-faithful to
    core/parallel/nano_ddp.py::_fire; if that changes, this must follow.
    """

    def arm(self):
        self.comm_ev = {}

    def _fire(self, bucket_index):
        bucket = self.buckets[bucket_index]
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in bucket]
        flat = _flatten(grads)
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            start = event()
            start.record(self.comm_stream)
            dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=self.group)
            end = event()
            end.record(self.comm_stream)
        self._inflight.append((flat, bucket_index))
        if getattr(self, "comm_ev", None) is not None:
            self.comm_ev[bucket_index] = (start, end)


def build(args):
    """The setup both arms share: d12 GPT, per-block compile, real FineWeb batches."""
    tokenizer = get_tokenizer()
    layout, resolver = text_orchestrator.assemble_vocab(tokenizer)
    dim = args.depth * 64
    cfg = GPTConfig(sequence_len=args.seq, vocab_size=layout.vocab_size,
                    n_layer=args.depth, n_head=max(1, (dim + 127) // 128),
                    n_kv_head=max(1, (dim + 127) // 128), n_embd=dim,
                    n_token_types=layout.n_token_types)
    # use_compile=False + compile_blocks = the per-block compile DDP requires, and
    # the single-GPU arm gets exactly the same treatment so the compute being timed
    # is the same compute. (A whole-graph compile would have no per-block seams at
    # all — every gradient finalizes at the end of backward.)
    setup = build_system(GPT, cfg, use_compile=False, seed=SEED, parallel="ddp")
    system = setup["system"]
    compile_blocks(system.trunk)

    sources = resolve_sources({"sequence_len": args.seq,
                               **{k: v for k, v in args.data.items()}},
                              args.seq, device="cuda")
    dataloader = MixedDataLoader(
        loader_config={"batch_size": args.bs,
                       "data": {"sequence_len": args.seq, "sources": sources}},
        tokenizers={"text": tokenizer, "layout": layout, "control_resolver": resolver},
        source_types=text_orchestrator.SOURCE_TYPES, resume_state_dict=None)
    return system, dataloader, setup["world_size"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    # the data declaration the orchestrator's config carries, inlined (this probe
    # does not go through hydra)
    args.data = {"datasets": {"fineweb": {"kind": "parquet_text",
                                          "splits": {"val": {"files": ["shard_005_00000.parquet"]},
                                                     "train": {"rest": True}}}},
                 "recipes": {"text_pretrain": {"template": ["bos", "text_start", "text_tokens",
                                                            "text_end", "eos"],
                                               "supervise": "all"}},
                 "sources": [{"type": "text", "dataset": "fineweb", "split": "train",
                              "recipe": "text_pretrain", "weight": 1.0,
                              "buffer_batch_size": 32, "tokenizer_threads": 4,
                              "tokenizer_batch_size": 128}]}

    system, dataloader, world = build(args)
    timer = BlockTimer(system.trunk.blocks)

    ddp = None
    if world > 1:
        ddp = TimedNanoDDP([[p for p in system.head.parameters() if p.requires_grad]]
                           + block_buckets(system.trunk), module=system)
    print0(f"\nd{args.depth} / seq {args.seq} / batch {args.bs} / world {world}"
           f"{f' / {len(ddp.buckets)} buckets' if ddp else ''} — "
           f"{args.warmup} warmup + {args.steps} measured steps\n")

    autocast = torch.autocast("cuda", torch.bfloat16)
    it = iter(dataloader)
    samples = []
    for step in range(args.warmup + args.steps):
        measured = step >= args.warmup
        system.zero_grad(set_to_none=True)
        if measured:
            timer.arm()
            if ddp is not None:
                ddp.arm()
        # T+0 must be the entry to BACKWARD, so the forward has to be launched and
        # timestamped first — an event recorded before `system.loss(...)` fires on
        # the GPU before the forward, and every number below would silently carry
        # ~24 ms of forward in it.
        t_fwd = event()
        t_fwd.record()
        with autocast:
            loss = system.loss(next(it))
        t0 = event()
        t0.record()
        with sync_gradients(ddp, enabled=True):
            loss.backward()
        t_bwd = event()
        t_bwd.record()
        torch.cuda.synchronize()
        if not measured:
            continue
        row = {"forward": t_fwd.elapsed_time(t0),
               "backward_end": t0.elapsed_time(t_bwd),
               "blocks": [{"start": t0.elapsed_time(s["start"]),
                           "end": t0.elapsed_time(s["end"])} for s in timer.slots]}
        if ddp is not None:
            row["comm"] = {i: {"start": t0.elapsed_time(s), "end": t0.elapsed_time(e)}
                           for i, (s, e) in sorted(ddp.comm_ev.items())}
        samples.append(row)

    if int(os.environ.get("RANK", 0)) != 0:
        dist.destroy_process_group()
        return

    def med(f):
        return median(f(s) for s in samples)

    n = args.depth
    print(f"T+0 = entry to backward().  median of {len(samples)} steps, milliseconds.\n")
    if ddp is None:
        print("  layer | compute start |  compute end |  duration")
        print("  ------+---------------+--------------+----------")
        for i in reversed(range(n)):            # backward order: last block first
            s = med(lambda r, i=i: r["blocks"][i]["start"])
            e = med(lambda r, i=i: r["blocks"][i]["end"])
            print(f"  {i:5d} | {s:13.3f} | {e:12.3f} | {e - s:8.3f}")
        print(f"\n  backward() returns (GPU) at T+{med(lambda r: r['backward_end']):.3f} ms")
        print(f"  (forward, for reference, took {med(lambda r: r['forward']):.3f} ms before T+0)")
    else:
        names = {0: "head", n + 1: "root"}
        print("  bucket        | compute start |  compute end |  comm start |    comm end | comm dur")
        print("  --------------+---------------+--------------+-------------+-------------+---------")
        for b in range(n + 2):
            layer = None if b in names else n - b        # bucket 1 -> block n-1
            label = names.get(b, f"block {layer}")
            cs = ce = None
            if layer is not None:
                cs = med(lambda r, i=layer: r["blocks"][i]["start"])
                ce = med(lambda r, i=layer: r["blocks"][i]["end"])
            ms = med(lambda r, b=b: r["comm"][b]["start"])
            me = med(lambda r, b=b: r["comm"][b]["end"])
            c1 = f"{cs:13.3f}" if cs is not None else " " * 13
            c2 = f"{ce:12.3f}" if ce is not None else " " * 12
            print(f"  {b:2d} {label:10s} | {c1} | {c2} | {ms:11.3f} | {me:11.3f} | {me - ms:8.3f}")
        print(f"\n  backward() returns (GPU) at T+{med(lambda r: r['backward_end']):.3f} ms")
        print(f"  (forward, for reference, took {med(lambda r: r['forward']):.3f} ms before T+0)")

    out = REPO / "outputs" / "ddp_timeline"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"timeline_world{world}.json"
    path.write_text(json.dumps({"config": {"depth": args.depth, "seq": args.seq,
                                           "batch": args.bs, "world": world},
                                "samples": samples}, indent=2))
    print(f"\nwrote {path}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
