"""
benchmark.py — does the fast decode path compute the same thing, and how much faster?

Two questions, in that order. A decode path that is fast and subtly wrong is worse
than a slow one, so equivalence is checked first and speed is only reported after.

    python -m exemplars.nano_world_model.inference.benchmark
    python -m exemplars.nano_world_model.inference.benchmark --ckpt <dir> --tokens 512

Without --ckpt it runs on a randomly initialised model. That is not a shortcut: both
questions are about kernels and scheduling, and neither the numerics of an attention
mask nor the cost of a kernel launch depends on what the weights contain. A trained
checkpoint changes which codes come out, not whether the two paths agree.

WHAT IS CHECKED

  1. Teacher-forced argmax agreement, with NO cascade. Both paths consume the SAME
     real token sequence, so any disagreement is pure numerics rather than one path
     having drifted into a different context. At each disagreement the top1-top2
     margin is printed: near-zero margins are legitimate near-ties that bf16 can flip
     either way, large margins would be a real bug.
  2. Greedy decode agreement over several segments. Finite-temperature draw-for-draw
     identity is NOT a meaningful target across a kernel change — bf16 noise flips
     near-tied candidates, and that is equally true of core's own compiled path.
     Greedy argmax is the check that is robust to kernel noise.
  3. Interactive continuation: a second segment decoded on the cache left by the
     first must agree too. That is the path an interactive session actually takes,
     and it is where an off-by-one in the write position would appear.
"""

import argparse
import time

import torch

from exemplars.nano_world_model import spec

spec.pin_tokenizer()

import modalities.control                              # noqa: E402
import modalities.text                                 # noqa: E402

from core.model.gpt import GPT, GPTConfig              # noqa: E402
from core.model.kv_cache import STATIC, KVCache, StaticKVCache  # noqa: E402
from core.training.model_setup import build_system, load_system  # noqa: E402

from exemplars.nano_world_model import train_wm        # noqa: E402
from exemplars.nano_world_model.inference import sample         # noqa: E402


def build(args, layout):
    if args.ckpt:
        return load_system(args.ckpt, sequence_len=args.seq_len)["system"]
    cfg = GPTConfig(sequence_len=args.seq_len, vocab_size=layout.vocab_size,
                    n_layer=args.depth, n_head=args.n_head, n_kv_head=args.n_head,
                    n_embd=args.dim, n_token_types=layout.n_token_types)
    return build_system(GPT, cfg, use_compile=False, seed=0)["system"]


@torch.no_grad()
def teacher_forced(system, layout, type_id, seq, static):
    """Per-position argmax over the band, plus the top1-top2 margin at each position."""
    toks = torch.tensor([seq], dtype=torch.long, device="cuda")
    types = layout.classify_token_types(toks)
    cache = STATIC if static else KVCache.for_model(system.trunk.config, 1, toks.size(1) + 8)
    hidden = system.trunk(toks, token_types=types, kv_cache=cache)
    lo, hi = layout.ranges[type_id]
    band = system.head(hidden)[0].float()[:, lo:hi]
    top2 = band.topk(2, dim=-1)
    return top2.indices[:, 0].cpu(), (top2.values[:, 0] - top2.values[:, 1]).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", default=None, help="a trained AR checkpoint (else random init)")
    ap.add_argument("--tokens", type=int, default=256, help="tokens per timed segment")
    ap.add_argument("--seq-len", type=int, default=1536)
    ap.add_argument("--depth", type=int, default=spec.DEPTH)
    ap.add_argument("--dim", type=int, default=spec.DIM)
    ap.add_argument("--n-head", type=int, default=spec.N_HEAD)
    args = ap.parse_args()

    layout, resolver = train_wm.assemble_vocab()
    system = build(args, layout)
    system.eval()
    v_off = layout.offset(spec.VIDEO_TYPE_ID)
    a_off = layout.offset(spec.ACTION_TYPE_ID)
    tid = spec.VIDEO_TYPE_ID
    g = torch.Generator(device="cuda").manual_seed(0)
    cpf = spec.clip_geometry()["codes_per_frame"]

    # A plausible prefix: the given latent frame, then the actions driving the next one.
    given = (torch.randint(0, spec.CODEC_VOCAB, (cpf,), generator=g, device="cuda") + v_off)
    prefix = ([resolver.resolve("bos"), resolver.resolve(spec.VIDEO_START)]
              + given.tolist() + [a_off + 11] * spec.clip_geometry()["td"])

    print(f"model: d{args.depth} dim{args.dim}, vocab {layout.vocab_size}, "
          f"{'checkpoint' if args.ckpt else 'random init'}")
    print(f"prefix {len(prefix)} tokens, {args.tokens} tokens per segment\n")

    # --- 1. teacher-forced agreement -----------------------------------------
    real = (torch.randint(0, spec.CODEC_VOCAB, (cpf,), generator=g, device="cuda") + v_off)
    seq = prefix + real.tolist()
    a_dyn, _ = teacher_forced(system, layout, tid, seq, static=False)
    system.trunk.attach_kv_cache(
        StaticKVCache.for_model(system.trunk.config, 1, args.seq_len))
    a_st, margin = teacher_forced(system, layout, tid, seq, static=True)
    dis = (a_dyn != a_st).nonzero().flatten()
    print(f"teacher-forced argmax: {len(a_dyn) - len(dis)}/{len(a_dyn)} agree"
          + (f" | top1-top2 margins where they differ: "
             f"{[round(float(margin[i]), 4) for i in dis[:8]]}" if len(dis) else ""))

    # --- 2 + 3. greedy decode, including an interactive continuation ----------
    dyn_cache = st_cache = None
    for si, pre in enumerate([prefix, [a_off + 9] * 4, [a_off + 13] * 4]):
        d, dyn_cache = sample.band_cached(
            system, layout, tid, pre, None, seq_len=args.seq_len,
            temperature=1.0, top_k=1, seed=0, fixed_len=args.tokens, cache=dyn_cache)
        s, st_cache = sample.band_static(
            system, layout, tid, pre, seq_len=args.seq_len,
            temperature=1.0, top_k=1, seed=0, fixed_len=args.tokens, cache=st_cache)
        agree = sum(a == b for a, b in zip(d, s))
        print(f"greedy segment {si}: {agree}/{len(d)} identical"
              f"{'' if agree == len(d) else '  <-- inspect'}")

    # --- speed ----------------------------------------------------------------
    def bench(label, reps=3):
        ts = []
        for r in range(reps):
            cache = StaticKVCache.for_model(system.trunk.config, 1, args.seq_len)
            torch.cuda.synchronize()
            t0 = time.time()
            sample.band_static(system, layout, tid, prefix, seq_len=args.seq_len,
                                      temperature=0.85, top_k=100, seed=r,
                                      fixed_len=args.tokens, cache=cache)
            torch.cuda.synchronize()
            ts.append((time.time() - t0) / args.tokens * 1000)
        best = min(ts)
        # A latent frame is cpf codes, and the codec's temporal /4 means each latent
        # frame is 4 game frames — so report both, and let the reader judge "real time"
        # against the rate the game actually runs at.
        lat_per_s = 1000.0 / best / cpf
        print(f"  {label:<24} {best:.3f} ms/token   "
              f"{lat_per_s:.2f} latent frames/s ({lat_per_s * spec.CODEC_TEMPORAL_DS:.1f} "
              f"game frames/s)")
        return best

    print("\nspeed:")
    eager = bench("static, eager")
    system.trunk.compile(mode="reduce-overhead")
    graphed = bench("static, cuda graphs", reps=5)
    print(f"\n  cuda graphs are {eager / graphed:.2f}x the eager static path")


if __name__ == "__main__":
    main()
