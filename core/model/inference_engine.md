# Inference: KV caches and generation

Generating from a trained System, one token or one block at a time, without recomputing the prefix: a key/value cache
keeps every layer's keys and values for the positions already processed, and each forward adds only the new ones.
core provides the caches, the trunk's handling of them (what each forward may see, set by its `attention` argument),
`rewind` for taking positions back out, and a complete autoregressive sampler, `autoregressive_generate`. A project's
own inference loop (block diffusion's denoising passes, a video model's sliding window) is written on these pieces and
stays in the project.

| Piece | Code |
|---|---|
| `autoregressive_generate`, `sample_tokens` | [inference.py](inference.py) |
| `KVCache`, `StaticKVCache`, `STATIC` | [kv_cache.py](kv_cache.py) |
| the trunk's side: `attention`, `attach_kv_cache` | [gpt.py](gpt.py), described in [model.md](model.md) |

The examples on this page run on the CPU, in order, and share their names.

## `autoregressive_generate`

```text
prompt [B, T]  -> one causal forward through an empty cache       the cache now holds T positions
               -> the head on the last position only               logits [B, vocab]
repeat:  sample one token per row [B, 1] -> one forward of that token through the cache -> the head -> logits
until max_new_tokens tokens, or every row has produced stop_token (all ranks together)
```

```python
tokens = autoregressive_generate(system, prompt_ids, prompt_types, max_new_tokens, gen_token_type,
                                 stop_token=None, temperature=1.0, top_k=None, early_stop=True, generator=None)
# -> [B, S] new tokens, S <= max_new_tokens
```

- **The prompts have the same length**, `[B, T]` with T ≥ 1; there is no padding or attention mask for prompts of
  different lengths. `max_new_tokens` is at least 1, and `T + max_new_tokens` must fit in the trunk's rotary table.
- **`prompt_types`** are the prompt's token types (the vocabulary layout's `classify_token_types`), and every generated
  token gets `gen_token_type`, the type of the band the model generates in.
- **`stop_token`** is a global token ID; resolving a name (`"eos"`) to it is the caller's. A row that produces it is
  finished: it keeps stepping with the others, and its remaining positions are filled with `stop_token`. With
  `early_stop`, generation ends when every row on every rank is finished, so all ranks run the same number of
  forwards (under FSDP each forward is a collective). That holds only if every rank calls it with the same
  `max_new_tokens`, `stop_token` and `early_stop`; otherwise the ranks disagree on the number of forwards and wait on
  each other.
- **Sampling**: `temperature=0` is greedy; otherwise softmax at that temperature, after keeping only the `top_k` largest
  logits if given; pass a `torch.Generator` to make it reproducible. The same function is available alone as
  `sample_tokens(logits, temperature, top_k, generator)`.
- **The caller** puts the System in eval mode and enters any autocast; gradients are off inside.

The cache gives the same tokens as recomputing the whole sequence at every step. What it saves is the prefix's
recomputation: each step runs one new position through the trunk instead of the whole sequence, while that position
still attends to every cached one. The trunk here keeps PyTorch's default initialization rather than
`init_weights()`, which zeroes every block's output projections: freshly initialized, a trunk passes each token's
embedding straight through, its output does not depend on the prefix, and the comparison would prove nothing.

```python run
import torch
from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.system import LMSystem
from core.model.inference import autoregressive_generate

torch.manual_seed(0)
cfg = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, n_token_types=1)
trunk, head = GPT(cfg), LMHead(cfg.n_embd, cfg.vocab_size)   # PyTorch's default init: the blocks are not zero
torch.nn.init.normal_(head.lm_head.weight, std=0.5)           # a head with preferences, unlike the zero init
system = LMSystem(trunk, head).eval()                          # no input preparation: inference never needs one
with torch.no_grad():
    last = trunk(torch.tensor([[5, 9, 13]]))[:, -1]
    alone = trunk(torch.tensor([[13]]))[:, -1]
print("the last position depends on the prefix:", not torch.equal(last, alone))

prompt = torch.tensor([[5, 9, 13], [7, 3, 2]])
out = autoregressive_generate(system, prompt, torch.zeros_like(prompt), max_new_tokens=6, gen_token_type=0,
                              temperature=0)

seq = prompt                                                   # the same, recomputing everything at every step
with torch.no_grad():
    for _ in range(6):
        logits = head(trunk(seq, token_types=torch.zeros_like(seq)))[:, -1]
        seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)
print(tuple(out.shape), torch.equal(out, seq[:, 3:]))
```
```text output
the last position depends on the prefix: True
(2, 6) True
```

## Two caches

| | `KVCache` | `StaticKVCache` |
|---|---|---|
| what a layer gets back | a slice of its buffers that grows: its length says what is valid | the whole buffer, always; a mask says what is valid |
| write position | a Python int | a 0-dim tensor on the GPU, advanced in place |
| buffers | allocated at the first forward, in its dtype and device (so it follows autocast) | allocated when the cache is built |
| passed to the trunk | `trunk(ids, kv_cache=cache)` | attached once, `trunk.attach_kv_cache(cache)`, then `trunk(ids, kv_cache=STATIC)` |
| for | everything not chasing per-token latency | CUDA graphs |

CUDA graphs replay a recorded sequence of kernel launches, which requires every shape and every address to stay the
same from step to step; `StaticKVCache` is the cache built for that. A 12-layer, 768-wide model decoding one token at
a time (batch 1, RTX 5090) took 2.68 ms per token eager, 0.95 compiled, and 0.51 compiled and replayed as a graph. It
is attached rather than passed because a cache passed through `forward` is a mutated graph input to dynamo, which then
silently declines to capture the graph (measured 0.83 against 1.12 ms per token). Detach it with
`trunk.attach_kv_cache(None)`. It takes no BlockMask.

Both are sized from the trunk's config: `KVCache.for_model(config, batch_size, max_len)`,
`StaticKVCache.for_model(config, batch_size, max_len, device="cuda", dtype=torch.bfloat16)`. A static cache keeps
the keys and the values in one dtype, on one device, and converts nothing: both must arrive in that dtype, or the
first write fails. Under bf16 autocast they do not (the QK norm hands back fp32 keys while the values stay bf16), so
the static path runs with autocast off and the cache in the trunk's own dtype: a bf16 model, as `load_system` loads
by default, with the cache's default `dtype=torch.bfloat16`. The dynamic cache allocates its key and value buffers
separately, from what arrives, and so follows autocast. A forward writes its new
positions in every layer, and the position advances once, on the last layer's write. Writing past `max_len` fails: an
assertion for `KVCache`, an asynchronous device-side assertion for `StaticKVCache` (checking on the host would cost a
GPU sync per write, which is what the static cache exists to avoid).

## What each forward sees, and `rewind`

What a forward may read is that forward's `attention` argument, not a setting of the cache, so one cache serves causal
writes and bidirectional passes in alternation. With P positions cached and n new ones (the full table is in
[model.md](model.md)):

- `"causal"`: new position i reads the P cached positions and new positions 0 to i;
- `"bidirectional"`: every new position reads the P cached positions and all n new ones (a block-diffusion denoising
  pass over a block);
- a BlockMask (dynamic cache only): exactly what it says, built with `Q_LEN=n`, `KV_LEN=P+n`.

`rewind(n)` takes the last n positions back out of the cache, between two complete forwards; the next forward
overwrites them. A block-diffusion step uses it: the noisy block goes through the cache for each denoising pass and must
be taken out again before the next pass, and before the final, clean write of the block:

```python run
from core.model.kv_cache import KVCache

cache = KVCache.for_model(cfg, batch_size=1, max_len=16)
with torch.no_grad():
    trunk(torch.tensor([[5, 9, 13, 2]]), kv_cache=cache)                    # the clean prefix, causal
    print("after the prefix", cache.get_pos())
    for denoising_pass in range(2):                                         # a noisy block of 3, read both ways
        trunk(torch.tensor([[60, 60, 60]]), kv_cache=cache, attention="bidirectional")
        print("after a denoising pass", cache.get_pos())
        cache.rewind(3)
        print("after rewind(3)", cache.get_pos())
    trunk(torch.tensor([[21, 22, 23]]), kv_cache=cache)                     # the block, final and clean
    print("after the clean block", cache.get_pos())
```
```text output
after the prefix 4
after a denoising pass 7
after rewind(3) 4
after a denoising pass 7
after rewind(3) 4
after the clean block 7
```

- **`rewind` undoes history, not what was computed from it.** Positions that read the taken-back ones in a
  bidirectional pass still carry them in their keys and values. So rewinding into a bidirectional block takes the
  whole block out, never part of it. core does not record block boundaries: that is the project's inference loop's
  knowledge.
- **It takes a host integer**: not a tensor (reading one would sync the GPU, the cost the static cache avoids), not a
  bool, not a float. `StaticKVCache.rewind` moves its position on the GPU, so a replayed graph stays valid, and checks
  for going below zero with an asynchronous device-side assertion. `reset()` empties either cache and keeps its
  buffers.
- **A forward the trunk refuses leaves the cache as it was**: the trunk checks its arguments before any layer writes.
  A forward that fails midway leaves the cache partly written; the caller resets it or throws it away.

How many denoising passes to make, when to rewind and when to write the final block, and how a window slides are the
project's inference engine's decisions; core keeps to the cache operations, the visibility and the choice of kernel.

## Several GPUs

Under FSDP, the trunk and the head are separate shard groups, and the head's forward runs through FSDP's ordinary hooks;
every rank runs every forward. `autoregressive_generate` stops early only when every row on every rank is finished
(one all-reduce of a flag per step), so the ranks never disagree on the number of forwards.

## Not provided

Prompts of different lengths (a cache that understands padding), streaming or tool-use loops, and sampling restricted
to one band of the vocabulary are not in core: the last is a few lines in the caller (mask the logits outside the band
before sampling), and the first two will come when a project needs them.

## Reference

| | |
|---|---|
| `autoregressive_generate(system, prompt_ids, prompt_types, max_new_tokens, gen_token_type, stop_token=None, temperature=1.0, top_k=None, early_stop=True, generator=None)` | -> `[B, S]`; asserts that the prompt and the new tokens fit in the rotary table |
| `sample_tokens(logits, temperature=1.0, top_k=None, generator=None)` | `[B, vocab]` -> `[B, 1]` |
| `KVCache(n_layer, batch_size, n_kv_head, head_dim, max_len)`, `KVCache.for_model(config, batch_size, max_len)` | `get_pos()`, `insert_kv(layer, k, v)`, `rewind(n)`, `reset()` |
| `StaticKVCache(n_layer, batch_size, n_kv_head, head_dim, max_len, device="cuda", dtype=torch.bfloat16)`, `.for_model(...)` | `rewind(n)`, `reset()`, `insert_kv`, `rotary`, `attn_mask` |
| `GPT.attach_kv_cache(cache)` | attach a `StaticKVCache` (None detaches); forward with `kv_cache=STATIC` |
