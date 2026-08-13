# nano-dsv4 — a ~100M reproduction of DeepSeek V4's architecture

A minimal, readable reimplementation of the three ideas that make **DeepSeek V4**
different from a vanilla GPT, trained at ~100M active parameters on FineWeb and
compared against a params-matched GPT baseline. It plugs into the nanoinfra
**trunk seam** (`build_system`) with **zero changes to core** — that is the point
of this example: a genuinely different frontier architecture drops in as a trunk.

The goal is *understanding*, not a leaderboard. At 100M you cannot (and should
not) measure which frontier model is "better"; the honest outputs are (1) the
architecture arithmetic — KV-cache size, parameter / FLOP accounting, true at any
scale — and (2) a same-data, same-budget cross-entropy curve, for feel.

## The three changes vs a vanilla GPT

- **Residual stream → 4 parallel streams (mHC).** Manifold-Constrained
  Hyper-Connections: the single residual `x + f(x)` becomes four streams with a
  per-token, doubly-stochastic mixing matrix. See `arch/nano_dsv4.py`
  (`HyperConnection`).
- **MLP → Mixture-of-Experts.** 32 fine-grained experts, top-4 + 1 shared,
  independent (non-softmax) scoring with auxiliary-loss-free load balancing.
  See `arch/parts.py` (`MoEBlock`, `NoAuxRouter`).
- **Attention rebuilt.** Shared-KV MQA (K = V, one KV head) + a 96-token sliding
  window + two levels of learned compression (CSA / HCA) whose entries a
  "lightning indexer" selects per query. See `arch/nano_dsv4.py`
  (`V4Attention`, `V4Compressor`, `V4Indexer`).

## Run

From the repo root (single GPU):

```bash
# reproduce nano-dsv4 (~92M active params)
python projects/nano_dsv4/scripts/train_nano.py --arch dsv4 --max-steps 4000

# a params-matched GPT baseline for comparison
python projects/nano_dsv4/scripts/train_nano.py --arch gpt --gpt-dim 800 --gpt-heads 8 --max-steps 4000

# build + parameter accounting + one timed forward/backward, no training
python projects/nano_dsv4/scripts/train_nano.py --arch dsv4 --dry
```

Result at 4000 steps (≈0.52B tokens, same data / eval): nano-dsv4 reaches a lower
validation CE than a **params-matched** GPT baseline by ~0.19 nat — the edge is
architectural, not a parameter-count artifact.

## Scope and honesty

- **Training-mode only.** No inference KV cache, sliding-window cache, or
  compressor cross-step state — over half of the original code's complexity is in
  serving incremental decode, which a single training forward never touches.
- **Reference implementation, eager.** `arch/nano_dsv4.py` is meant to be read and
  to verify the architecture trains, not for throughput-optimized training — it
  runs at single-digit MFU (data-dependent expert loop / top-k). A same-architecture
  high-throughput implementation exists (3.7–4.2x throughput, byte-identical parameter
  tree, so checkpoints interchange); it is not included here because it is not yet
  stable enough for long runs.
- **Every shrink is deliberate.** Each dimension differs from the original for a
  reason recorded in the config (`NanoDSV4Config`); unrelated quantities are kept
  numerically distinct so shapes never collide when you read them.

## Where to learn more

A full, module-by-module walkthrough (the lecture note this example was built for)
is published separately. `arch/nano_dsv4.py` and `arch/parts.py` are written to be
read top-to-bottom alongside it.
