# Model

What the Trainer trains is a *System*: a trunk that turns token IDs into hidden states, a head that turns hidden
states into logits and a loss, and a rule for turning a batch into the trunk's arguments. core's `LMSystem` holds the
three together and offers the one method the Trainer calls, `loss(batch)`; its trunk is `GPT` and its head `LMHead`.
A project with another objective writes its own System with the same method.

```text
batch {"idx", "targets", "token_types", "doc_id", "doc_pos", ...}
  |
  |- system.prepare_trunk_inputs(batch) -> idx, {"token_types": ..., "attention": ..., "position_ids": ...}
  |- system.trunk(idx, **kwargs)        -> hidden [B, T, H]          GPT: embeddings, blocks, final norm
  '- system.head.loss(hidden, batch["targets"]) -> a scalar         mean CE over the targets that are not IGNORE
```

| Piece | Code |
|---|---|
| `LMSystem`, `prepare_unpacked_AR_trunk_inputs`, `prepare_packed_AR_trunk_inputs` | [system.py](system.py) |
| `GPT`, `GPTConfig` | [gpt.py](gpt.py) |
| `LMHead`, `LigerLMHead` | [heads.py](heads.py) |
| `same_sample`, `causal`, `pad_to_flex_block`, `default_mask_builder` | [masks.py](masks.py) |

Assembling a System on the GPUs (`build_system`) is in [../training/training.md](../training/training.md); the KV
caches and generation in [inference_engine.md](inference_engine.md); saving and loading in
[checkpoints.md](checkpoints.md). The examples on this page run on the CPU, in order, and share their names.

## `LMSystem`

```python
system = LMSystem(trunk, head, prepare_trunk_inputs_fn=None)     # prepare_trunk_inputs_fn is keyword-only
```

`loss(batch)` is three lines: `idx, kwargs = self.prepare_trunk_inputs(batch)`, `hidden = self.trunk(idx,
**kwargs)`, `return self.head.loss(hidden, batch["targets"])`. The System registers the trunk and the head as
submodules, so the checkpoint keys are `trunk.*` and `head.*`, and `system.config` is the trunk's config, which a
checkpoint records. It holds no parameters of its own.

- **The input preparation is chosen when the System is built**: a function `batch -> (idx, kwargs)` passed as
  `prepare_trunk_inputs_fn` (or the same argument of `build_system` and `load_system`), or a subclass that defines
  the method `prepare_trunk_inputs`, not both (`TypeError`). It cannot be assigned afterwards (`AttributeError`).
  Without one, the first `loss` or evaluation batch raises `RuntimeError`; pure inference never calls it. Training
  and core's evaluation call the same one, so they see the same inputs.
- **Every micro-batch must hold at least one target** that is not IGNORE: the loss is a mean, and with none it is
  0/0. core's sample builder and batchers never make such a micro-batch; a project's own must not either.
- **`estimate_flops()`** returns one number, the FLOPs per position, which the Trainer multiplies by the positions
  per step for MFU: the trunk's estimate plus 6 × the head's parameters. That is right only if every position goes
  through the trunk and the head and attention is dense causal. A System that runs the head on a subset of positions,
  or attention under a sparse mask, overrides it (nano_world_model's block-diffusion System printed an MFU of 101.6%
  before it did).

**A System of your own** subclasses `LMSystem` and overrides `loss`, and `prepare_trunk_inputs` and
`estimate_flops` as needed. Subclassing rather than wrapping keeps the checkpoint keys `trunk.*` and `head.*` (a wrapper would add a prefix to
every key), so its checkpoints load into another System whose trunk and head have the same keys and shapes (a load
takes the whole model's state: a System with extra parameters, or another head, does not load them), and the Trainer
runs it unchanged. The worked example is
`BlockDiffusionSystem` in [exemplars/nano_world_model/block_diffusion.py](../../exemplars/nano_world_model/block_diffusion.py):
its loss is a weighted CE over masked positions only (`head.loss_per_token` on the selected positions), its input
preparation builds a two-stream BlockMask, and its `estimate_flops` corrects for both.

## From a batch to the trunk's inputs

core has the two ordinary autoregressive preparations; a project picks one explicitly, or writes its own.

| | `prepare_unpacked_AR_trunk_inputs` | `prepare_packed_AR_trunk_inputs(batch, *, mask_builder=None)` |
|---|---|---|
| for batches with | at most one sample per row, starting at column 0 (padding on the right) | several samples per row |
| `kwargs` | `{"token_types": ...}` | `{"token_types": ..., "attention": BlockMask, "position_ids": batch["doc_pos"]}` |
| who sees whom | ordinary causal attention | "same sample and causal" |
| positions | the column, 0 .. T-1 | each sample from 0 (`doc_pos`) |

`token_types` is `batch.get("token_types")`, None when the batch has none. The unpacked preparation refuses a packed
batch, reading the host-side `layout_facts` that core's collate puts on every batch (no GPU sync); a batch without
them was not made by core's collate, and whoever made it answers for its rows. The packed one reads `doc_id` and
`doc_pos` as collate makes them, without checks of its own (a batch without either is a `KeyError`); the trunk checks
the positions where it reads them. Three samples packed into two rows of six, and what the packed preparation makes
of them:

```python run
import torch
from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.system import LMSystem, prepare_packed_AR_trunk_inputs
from core.data import build_NTP_supervised_sample, PackingBatcher

torch.manual_seed(0)
cfg = GPTConfig(sequence_len=8, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, n_token_types=1)
trunk, head = GPT(cfg), LMHead(cfg.n_embd, cfg.vocab_size)
trunk.init_weights()
head.init_weights()
system = LMSystem(trunk, head, prepare_trunk_inputs_fn=prepare_packed_AR_trunk_inputs)

raw = [torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21, 22]), torch.tensor([30, 31, 32, 33, 34, 35])]
samples = [build_NTP_supervised_sample(t) for t in raw]          # sample A, B, C
batch = next(iter(PackingBatcher(samples, shape=(2, 6), pad_values={"idx": 0, "targets": -1})))
idx, kwargs = system.prepare_trunk_inputs(batch)
print(batch["idx"].tolist())
print(batch["doc_id"].tolist())
print(kwargs["position_ids"].tolist())
print(sorted(kwargs), type(kwargs["attention"]).__name__)
```
```text output
[[10, 11, 12, 20, 21, 0], [30, 31, 32, 33, 34, 0]]
[[0, 0, 0, 1, 1, -1], [2, 2, 2, 2, 2, -1]]
[[0, 1, 2, 0, 1, 0], [0, 1, 2, 3, 4, 0]]
['attention', 'position_ids', 'token_types'] BlockMask
```

The BlockMask's rule is "same sample and causal". Evaluated position by position for the first row (`x`: the query
on the left may read the key above):

```python run
from torch.nn.attention.flex_attention import and_masks
from core.model.masks import causal, same_sample

def show(rule, labels):
    pos = torch.arange(len(labels))
    visible = rule(torch.tensor(0), torch.tensor(0), pos[:, None], pos[None, :])     # row 0, every head
    print("query \\ key " + "".join(f"{label:>4}" for label in labels))
    for label, keys in zip(labels, visible.tolist()):
        print(f"{label:>11} " + "".join("   x" if v else "   ." for v in keys))

show(and_masks(same_sample(batch["doc_id"]), causal), ["a0", "a1", "a2", "b0", "b1", "pad"])
```
```text output
query \ key   a0  a1  a2  b0  b1 pad
         a0    x   .   .   .   .   .
         a1    x   x   .   .   .   .
         a2    x   x   x   .   .   .
         b0    .   .   .   x   .   .
         b1    .   .   .   x   x   .
        pad    .   .   .   .   .   .
```

B never reads A, and padding reads and is read by nothing. With `doc_pos` as the positions, B's first token sits at
position 0, exactly as if it had a row of its own.

**Training with a BlockMask requires a compiled trunk** (`compile_blocks` or `compile_system_trunk`): with gradient
recording on, the trunk refuses to run flex attention uncompiled, because torch's uncompiled flex is a debugging path
whose backward torch itself warns may be wrong. Without gradients (evaluation) it runs, with torch's warning that it
materializes the whole score matrix: fine for this example, but a long row needs the compiled kernel for memory (a
129-frame two-stream video batch, 2 rows of 17408 positions and 12 heads, asked for 27 GiB uncompiled). The zero head makes every target's CE ln 64:

```python run
with torch.no_grad():
    print(round(system.loss(batch).item(), 4))
try:
    system.loss(batch)                                              # gradients on, trunk not compiled
except RuntimeError as e:
    print("RuntimeError:", str(e).split(":")[0])
```
```text output
4.1589
RuntimeError: flex attention (a BlockMask) with gradient recording on must run compiled
```

## Your own visibility rule

A rule is a function `rule(b, h, q, kv) -> bool`: may query position `q` of row `b`, in head `h`, read key position
`kv`? Rules combine with torch's `and_masks` and `or_masks`, and a mask builder (`default_mask_builder(device)`:
torch's `create_block_mask`, compiled once per process on CUDA) turns one into a BlockMask for a batch:

```python
mask = default_mask_builder(device)(rule, B=B, H=None, Q_LEN=T, KV_LEN=T, device=device)   # H=None: all heads alike
```

flex attention works in tiles of 128 × 128 positions (`FLEX_BLOCK`): it skips a tile the rule hides entirely and
evaluates the rule at every pair inside a partly visible one. A row whose length is not a multiple of 128 has a last
tile reaching past it, so every per-position field a rule reads must be padded to the multiple first:
`pad_to_flex_block(field, value)`. The padding only lets the rule read those indices; the model's input stays T
long. `same_sample(doc_id)` pads `doc_id` itself, with -1.

A prefix-LM rule as an example: the first two positions of a sample are its prompt, which reads itself both ways;
the rest is causal. It reads a per-position field, padded first. A visibility rule needs its supervision to match:
here position p0 reads p1, and under next-token prediction p0's target is p1's token, so the prompt's tokens must not
be targets (`supervision_mask` False on them); the last prompt position still predicts the first token after the
prompt, which it cannot see.

```python run
from torch.nn.attention.flex_attention import or_masks
from core.model.masks import pad_to_flex_block

doc_id = torch.tensor([[0, 0, 0, 0, 0, -1]])                        # one sample of five positions, then padding
prompt = pad_to_flex_block(torch.tensor([[1, 1, 0, 0, 0, 0]]), 0)   # [1, 128]: the rule may read any index of the tile

def both_in_prompt(b, h, q, kv):
    return (prompt[b, q] == 1) & (prompt[b, kv] == 1)

show(and_masks(same_sample(doc_id), or_masks(causal, both_in_prompt)), ["p0", "p1", "s2", "s3", "s4", "pad"])
```
```text output
query \ key   p0  p1  s2  s3  s4 pad
         p0    x   x   .   .   .   .
         p1    x   x   .   .   .   .
         s2    x   x   x   .   .   .
         s3    x   x   x   x   .   .
         s4    x   x   x   x   x   .
        pad    .   .   .   .   .   .
```

A project wires its rule in through its own input preparation (`prepare_trunk_inputs_fn`, or a System subclass) and
passes the mask as the trunk's `attention`.

## The trunk: `GPT`

```python
GPTConfig(sequence_len=1024, vocab_size=50304, n_layer=12, n_head=6, n_kv_head=6, n_embd=768, n_token_types=3)
hidden = trunk(idx, token_types=None, kv_cache=None, attention="causal", position_ids=None)   # -> [B, T, n_embd]
# attention and position_ids are keyword-only
```

A decoder-only transformer: rotary positions and no position embedding, RMS norms without learned parameters (after
the embedding, before each attention and MLP, and at the end), QK norm, ReLU² MLPs four times as wide, no biases,
grouped-query attention (`n_kv_head` divides `n_head`), and an untied head (the head is a separate module). Its
config gives the geometry; `vocab_size` and `n_token_types` come from the vocabulary layout the orchestrator assembles.
Another trunk class can stand in for `GPT` if it keeps the contract `build_system` relies on ("A trunk of your own"
in [../training/training.md](../training/training.md)).

| Argument | |
|---|---|
| `idx` | `[B, T]` int64 token IDs |
| `token_types` | `[B, T]` int64, or None: adds a learned embedding per token type (`n_token_types` of them) |
| `kv_cache` | None for training and teacher forcing; a cache for step-by-step inference ([inference_engine.md](inference_engine.md)) |
| `attention` | `"causal"`, `"bidirectional"` or a BlockMask, below |
| `position_ids` | `[B, T]` int64, or None: the row of the rotary table each token reads, below |

`attention` says which positions each position reads, and nothing is added to a BlockMask:

| `attention` | without a cache | with P positions cached and n new ones |
|---|---|---|
| `"causal"` (default) | each position reads itself and the ones before it | new position i reads the P cached ones and new ones 0 .. i |
| `"bidirectional"` | every position reads every position | each new one reads the P cached ones and all n new ones |
| a BlockMask | exactly what the mask says (flex attention) | the same, with `Q_LEN=n`, `KV_LEN=P+n`; not with the static cache |

The trunk checks its arguments before any layer writes a cache, so a refused call leaves the cache as it was:
`attention` of another kind (`TypeError`); `position_ids` together with a cache, or not an int64 tensor shaped and
placed like `idx`; a BlockMask with the static cache (`ValueError`); `kv_cache=STATIC` with no static cache
attached (`RuntimeError`). Values of `position_ids` outside the table are
caught where the table is read: a device-side assert on CUDA, which stops the process (rerun with
`CUDA_LAUNCH_BLOCKING=1` to see where), a `ValueError` on the CPU.

**Token types** are added as `one_hot(token_types) @ W` rather than an embedding lookup, so their backward sums in a
fixed order and two runs give the same gradient bit for bit. They must be int64. Under autocast every embedding is cast
to the autocast dtype before the sum, so fp32 storage computes like bf16 storage there.

**Initialization** (`init_weights()`): linear layers normal with std 1/√fan_in × min(1, √(fan_out/fan_in)),
embeddings N(0, 1), the two output projections of every block and the type embedding zero, the rotary table
recomputed. The head starts at zero (`LMHead.init_weights()`). Parameters stay in the dtype they were built in;
`build_system` decides the storage.

**Weight decay**: `GPT` and `LMHead` declare no exemptions (`no_weight_decay_names`, "Weight decay exemptions" in
[../training/training.md](../training/training.md)), so every parameter is decayed: the norms have no learned
parameters and there are no biases.

**`estimate_flops()`** is 6 × (the parameters minus the token embedding) + 12 × layers × heads × head width × T, the
attention term charged for the full sequence length (the PaLM / nanoGPT convention, which does not credit the causal
half: 2.9 to 5.4% of the total at the sizes measured, and the same in every comparison of dense causal models).

Which kernel runs, for the record (the cases are tried in this order):

| Case | Kernel |
|---|---|
| static cache (`kv_cache=STATIC`) | a written-out masked attention (a bool mask would push SDPA off its fast kernels) |
| a BlockMask | flex attention |
| `"bidirectional"` | SDPA, no mask |
| no cache, or a cache that was empty before this call (the prompt's prefill) | SDPA with `is_causal=True` (FlashAttention on CUDA) |
| dynamic cache, one new position | SDPA, no mask: every cached position is in the past |
| dynamic cache, a chunk of new positions | SDPA with a lower-right causal bias (a bool mask when compiled or the dtypes differ) |

## Positions: the rotary table

The trunk rotates queries and keys by a table of S rows; `position_ids` says which row each token reads. The default
table is one-dimensional with S = 10 × `sequence_len` rows; without `position_ids` the token in column t reads row t
(row P + t after P cached positions). The packed preparation passes `doc_pos`, so every sample starts at row 0:

```text
row contents     a0  a1  a2  b0  b1  pad
doc_pos           0   1   2   0   1    0      position_ids: the rows of the table these tokens read
```

A project with another notion of position, three-dimensional RoPE for video say, computes its own table and installs
it:

```python
trunk.set_rotary_table(cos, sin)      # each [1, S, 1, head_dim / 2], bfloat16, on the current table's device
trunk.rotary_table_len                # S: position_ids must lie in 0 .. S - 1
```

- The table is not saved in checkpoints: install it after building or loading the model, and after
  `init_weights()`, which puts the default table back.
- Install it before creating any KV cache (cached keys carry the old rotation; with a static cache attached it is
  refused) and before compiling or the first compiled call. A compiled trunk reads whatever table is installed (a
  table of the same shape costs no recompile, another length one recompile), but a CUDA graph recorded with the old
  table stays behind.
- Do not edit an installed table in place; install a new one. Every check runs before anything is replaced, so a
  refused table leaves the old one.

S bounds the positions, not the row length: a packed row may be longer than S, as long as each token's position is
in the table.

## The head: `LMHead`

`LMHead(n_embd, vocab_size, softcap=15.0)`: a linear un-embedding without bias, then the logits squashed to
(-15, 15) by 15 · tanh(logits / 15), in fp32. It takes tensors, not batches:

| Method | Returns |
|---|---|
| `head(hidden)` | the logits `[B, T, vocab_size]`, fp32; the inference entry |
| `head.loss(hidden, targets)` | the mean CE over the targets that are not IGNORE (-1); the training entry |
| `head.loss_per_token(hidden, targets)` | the CE of every position, flattened to `[B · T]`, 0 at IGNORE; evaluation and weighted objectives |

```python run
hidden = trunk(idx)                                   # the packed batch above, causal attention for brevity
print(tuple(hidden.shape), tuple(head(hidden).shape), head(hidden).dtype,
      tuple(head.loss_per_token(hidden, batch["targets"]).shape))
```
```text output
(2, 6, 32) (2, 6, 64) torch.float32 (12,)
```

`build_system(head_ce=...)` picks the implementation of the two loss methods ([../training/training.md](../training/training.md)
compares them). `"liger"` swaps the head's class for `LigerLMHead` before sharding: its `loss` is Liger's fused
kernel on CUDA (the naive one on the CPU), and its `loss_per_token` serves only with gradients off, raising
`NotImplementedError` otherwise, because Liger's backward for unreduced CE gets per-position weights wrong; a training
objective that weights positions uses `"compiled"`. `"compiled"` keeps `LMHead` and binds compiled versions of `loss`
and `loss_per_token` on the instance, so call them through the instance (`head.loss_per_token(...)`), not the class.
Under FSDP the head is its own shard group, and `loss` and `loss_per_token` are registered as FSDP forward methods.

## Reference

| | |
|---|---|
| `LMSystem(trunk, head, *, prepare_trunk_inputs_fn=None)` | `loss(batch)`, `prepare_trunk_inputs(batch)`, `estimate_flops()`, `config`; `TypeError` when a subclass method and the argument are both given, `AttributeError` on assigning `prepare_trunk_inputs`, `RuntimeError` at first use without one |
| `prepare_unpacked_AR_trunk_inputs(batch)` | -> `(idx, {"token_types"})`; `ValueError` for a packed batch |
| `prepare_packed_AR_trunk_inputs(batch, *, mask_builder=None)` | -> `(idx, {"token_types", "attention", "position_ids"})`; bind another builder with `functools.partial` |
| `GPT(config)`, `GPT.forward(idx, token_types=None, kv_cache=None, *, attention="causal", position_ids=None)` | -> hidden `[B, T, n_embd]`; also `init_weights()`, `blocks`, `estimate_flops()`, `set_rotary_table(cos, sin)`, `rotary_table_len`, `attach_kv_cache(cache)`, `Config` |
| `LMHead(n_embd, vocab_size, softcap=15.0)` | `forward`, `loss`, `loss_per_token`, `init_weights()` |
| `same_sample(doc_id)`, `causal`, `pad_to_flex_block(x, value)`, `default_mask_builder(device)`, `FLEX_BLOCK` | in `core.model.masks` |
