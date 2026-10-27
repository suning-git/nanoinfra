# Evaluation

While it trains, the Trainer asks a list of evaluators for numbers at the steps they choose. core provides the
evaluator interface, one evaluator that measures cross-entropy metrics of a System on a fixed number of batches
(`LossEvaluator`), the function underneath it (`evaluate_loss`) and three metrics: mean CE, CE per token type and bits
per byte. The rest is the project's, written against the same interface: which data to evaluate on, how often, and
any other kind of measurement.

| Piece | Code |
|---|---|
| `Evaluator`, `LossEvaluator` | [evaluator.py](evaluator.py) |
| `evaluate_loss` | [eval_loss.py](eval_loss.py) |
| `mean_ce`, `type_ce`, `bits_per_byte` | [metrics.py](metrics.py) |
| when the Trainer evaluates (`eval_is_due`, `Trainer._evaluate`) | [../training/trainer.py](../training/trainer.py) |

The examples on this page run on the CPU, in order, and share their names.

## A first example

A two-layer GPT, three validation samples made into two batches once, and a `LossEvaluator` with three metrics (how
samples become batches is in [../data/data_pipeline.md](../data/data_pipeline.md)). The head comes straight from
`init_weights`, which zeroes it, so every prediction is uniform over the 64 tokens and every target's CE is
ln 64 ≈ 4.1589: easy numbers to check the metrics against.

```python run
import contextlib
import torch
from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.system import LMSystem, prepare_unpacked_AR_trunk_inputs
from core.data import build_NTP_supervised_sample, PaddingBatcher
from core.evaluation import LossEvaluator, mean_ce, type_ce, bits_per_byte

torch.manual_seed(0)
cfg = GPTConfig(sequence_len=8, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, n_token_types=2)
trunk, head = GPT(cfg), LMHead(cfg.n_embd, cfg.vocab_size)
trunk.init_weights()
head.init_weights()
system = LMSystem(trunk, head, prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)

# IDs 60-63 play the control tokens (type 1, no bytes); every other ID is a text token of two bytes
raw = [torch.tensor([60, 9, 13, 2, 61]), torch.tensor([60, 3, 62]), torch.tensor([60, 8, 15, 16, 23, 61])]
samples = [build_NTP_supervised_sample(t, token_types=(t >= 60).long()) for t in raw]   # idx = t[:-1], targets = t[1:]
pads = {"idx": 0, "targets": -1, "token_types": 0, "target_types": 0}
val_batches = list(PaddingBatcher(samples, shape=(2, 8), pad_values=pads))     # built once: the same every time
token_bytes = torch.full((64,), 2)
token_bytes[60:] = 0

ev = LossEvaluator(val_batches, eval_steps=len(val_batches),
                   metrics={"val/ce": mean_ce(), "val/ce_text": type_ce(0), "val/bpb": bits_per_byte(token_bytes)})
system.eval()
result = ev.evaluate(system, contextlib.nullcontext())     # the Trainer passes its bf16 autocast here
print({name: round(value, 4) for name, value in result.items()})
```
```text output
{'val/ce': 4.1589, 'val/ce_text': 4.1589, 'val/bpb': 3.0}
```

```text
                position     0     1     2     3     4     5     6     7
batch 0  row 0  idx         60     9    13     2     .     .     .     .
                targets      9    13     2  [61]     -     -     -     -
         row 1  idx         60     3     .     .     .     .     .     .
                targets      3  [62]     -     -     -     -     -     -
batch 1  row 0  idx         60     8    15    16    23     .     .     .
                targets      8    15    16    23  [61]     -     -     -
         row 1  (empty: the last batch holds one sample)

  .  padding input (pad value 0)    -  IGNORE target (-1)    [61]  a control target: type 1, no bytes
```

The first token of a sample is an input only, never a target, so the three samples have 11 targets: 8 text tokens
of two bytes each and 3 control tokens. `val/ce` averages over all 11, `val/ce_text` over the 8 text targets, and
`val/bpb` charges those 8 targets' CE, 6 bits each, to their 16 bytes: 3 bits per byte. Evaluating by hand like this,
the caller switches the System to eval mode and passes the autocast context; the Trainer does both itself.

## When the Trainer evaluates

`Trainer(..., evaluators=[...])` takes the list. With none, the default, it never evaluates: there is no
`evaluation.enabled` key, and core reads no evaluation config at all. The `evaluation` section of a recipe is read
by the project's code that builds its evaluators (the text modality reads `interval_steps`, `eval_at` and
`eval_tokens` there).

Where evaluation sits in one training step s:

```text
Trainer.train(), step s
 |- the learning rate for step s
 |- forward + backward on each micro-batch, optimizer update          the model now has s + 1 updates
 |- console log
 |- evaluation, if due: some evaluator's should_eval(s) is True, or s is the launch's last step
 |     system.eval()
 |     for each evaluator whose should_eval(s) is True (every one of them at the last step):
 |         results.update(evaluator.evaluate(system, autocast_ctx))
 |     system.train()
 |     rank 0: prints "Step 00100 | val/ce: 3.4123 | val/bpb: 1.0236", logs results to the metrics sink
 '- checkpoint, if due
```

- The reading logged at step s is of the model after s + 1 updates: step 0's is not a reading of the untrained model.
- The launch's last step is `max_steps - 1`, or the step where `num_iterations` stops it.
- `autocast_ctx` is the bf16 autocast the Trainer trains under.
- The evaluators' dicts go into one: a name that two evaluators both return keeps the later one's value, without a
  warning, so give every reading its own name. Rank 0 logs them with the step, the FLOPs and the training time so
  far.

`Evaluator.should_eval` follows `eval_at`, a set of steps, when it is set, and `interval_steps` otherwise (default
50): `step % interval_steps == 0`, which includes step 0. `LossEvaluator` takes neither as an argument; set the
attributes after building it, or subclass. `eval_is_due` is the Trainer's own test:

```python run
from core.training.trainer import eval_is_due

ev.interval_steps = 4
print([s for s in range(10) if eval_is_due([ev], s, last_step=(s == 9))])
ev.eval_at = {1, 3, 7}                      # replaces the interval
print([s for s in range(10) if eval_is_due([ev], s, last_step=(s == 9))])
```
```text output
[0, 4, 8, 9]
[1, 3, 7, 9]
```

For a reading of the model before its first update (untrained, or just loaded from a checkpoint), call the
evaluators yourself before `trainer.train()`, on every rank: `trainer.system.eval()`, then
`ev.evaluate(trainer.system, trainer.autocast_ctx)`. `train()` switches the System back to train mode when it starts.

## `LossEvaluator` and `evaluate_loss`

```python
LossEvaluator(dataloader, eval_steps, metrics)              # .evaluate(model, autocast_ctx) -> {name: float}
evaluate_loss(model, loader, steps, metrics) -> {name: float}
```

`LossEvaluator.evaluate` enters `autocast_ctx` and calls `evaluate_loss(model, dataloader, eval_steps, metrics)`,
which runs each batch through the System the way training does and hands the CE to the metrics, with gradients off:

```text
for each of `steps` batches from iter(loader):
    idx, kwargs = model.prepare_trunk_inputs(batch)             the translation training uses
    hidden = model.trunk(idx, **kwargs)                         compiled, if the orchestrator compiled it
    ce = model.head.loss_per_token(hidden, batch["targets"])    CE at every position, shaped like targets, 0 at IGNORE
    for each metric:
        numerator, denominator = metric(batch, ce)              added to two running sums, in float64
sum the running sums over the ranks                             one all-reduce
reading = sum of numerators / sum of denominators               inf when no batch on any rank had anything for it
```

What follows from this:

- **It measures the CE that the batch's `targets` define**, not `model.loss`. An auxiliary loss term, or targets a
  System builds inside its own `loss`, are not in it.
- **The loader is iterated afresh at every evaluation**, and whether that starts over is up to the loader. A list of
  batches starts over every time. A `MixedDataLoader` goes on from where its sources stand, so every evaluation would
  read different batches, and successive readings would differ by the data as well as by the model. Freeze the
  validation batches (build them once, as in the first example), or build fresh sources and a fresh loader inside
  each evaluation, seeded the same way each time, as the text modality's `TextEvaluator` does. A new
  `MixedDataLoader` around the same source objects is not enough: the sources go on from where they stand.
- **Batches must already be on the model's device.** A `MixedDataLoader` built with `device=` moves them; a frozen
  list is moved once by whoever builds it.
- **`steps` is the same on every rank, and the loader yields at least that many batches** (it raises if the loader
  runs out). Under FSDP every forward is a collective, so a rank that runs fewer forwards leaves the others waiting.
- **The head computes the CE at every position**, padding and positions without a target included. With few targets
  per row that is most of the cost; nano_world_model wraps the head so that it computes only where there are targets
  (`_AtTargets` in [exemplars/nano_world_model/evaluator.py](../../exemplars/nano_world_model/evaluator.py)).
- **Precision.** Under `LossEvaluator` the forward runs in the Trainer's bf16 autocast, and the per-position CE is
  whatever the head's CE implementation computes (`build_system(head_ce=...)`; the Liger one serves this entry only
  with gradients off, as here). Only the running totals are float64. Called directly, `evaluate_loss` runs in the
  caller's autocast context and mode.

## Metrics

A metric is a function of the batch and the per-position CE that returns two numbers for the batch, a numerator and
a denominator:

```python
metric(batch, ce) -> (numerator, denominator)       # ce is shaped like batch["targets"]
```

Each is a Python number or a 0-dim tensor on `ce`'s device; a batch with nothing for the metric returns `(0, 0)`. A
metric reads only the batch and `ce`: it keeps no state from batch to batch and does not communicate between ranks,
since `evaluate_loss` does the sums. Every rank must build the same metrics, under the same names (the same code and
config see to it). core's three:

| Metric | Numerator / denominator for one batch |
|---|---|
| `mean_ce()` | CE summed over the targets that are not IGNORE / their count |
| `type_ce(t)` | the same over the targets whose `target_types` is `t`: the type of the token being predicted, not of the input (the batch must carry `target_types`) |
| `bits_per_byte(token_bytes)` | CE in bits summed over the targets that carry bytes / their bytes (below) |

The sums run over every batch and rank before the one division, so `mean_ce` weights every target equally. Take two
batches, one with 3 targets whose CE adds up to 6 and one with a single target of CE 4: they read
(6 + 4) / (3 + 1) = 2.5, not the mean of the two batch means, (2 + 4) / 2 = 3. A metric that wants the mean of batch
means returns `(batch mean, 1)`. A reading that is not a ratio of two sums, such as perplexity (exp of the global
CE), is computed from the result afterwards.

To write one, select positions with a boolean mask and index with it. Do not multiply by a 0/1 mask: a NaN at a
position meant to be left out stays NaN after the multiplication by 0. This metric reads the CE of the predictions
made with more than k tokens of their own sample as context. It uses the batch's `doc_pos`, the position inside the
sample: with next-token prediction under causal attention, position p predicts the sample's token p + 1 from its
tokens 0 to p.

```python run
from core.tokenization.vocab_layout import VocabLayout

IGNORE = VocabLayout.IGNORE_INDEX                    # -1

def late_ce(k):
    def metric(batch, ce):
        sel = (batch["targets"] != IGNORE) & (batch["doc_pos"] >= k)
        return ce[sel].sum(), sel.sum()
    return metric

# row 0 holds two samples (positions 0-1 and 2) and one padding position; row 1 one sample and padding
batch = {"targets": torch.tensor([[7, 8, 9, IGNORE], [5, 6, IGNORE, IGNORE]]),
         "doc_pos": torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]])}
ce = torch.tensor([[1.0, 3.0, 2.0, 0.0], [4.0, 6.0, 0.0, 0.0]])
print(mean_ce()(batch, ce))
print(late_ce(1)(batch, ce))
```
```text output
(tensor(16.), tensor(5))
(tensor(9.), tensor(2))
```

### Bits per byte

`bits_per_byte(token_bytes)` takes a table of UTF-8 byte counts with one entry per token ID of the assembled
vocabulary (`[vocab_size]`), 0 for control tokens and for the tokens of other modalities. Every target is looked up in
it, so an ID past its end raises an index error. The text modality's `get_token_bytes()`
(`modalities/text/tokenizer.py`) returns the table of the text tokenizer's own vocabulary: that is the whole table
only when the vocabulary is the text tokenizer's. With other modalities in the vocabulary, build the full table: the
text tokenizer's counts at the IDs where the vocabulary puts its tokens, 0 everywhere else. Over the targets that
carry bytes, the metric sums their CE, converted to bits, and their bytes:

```text
bpb = (sum of CE in nats over the targets that carry bytes) / (ln 2 x their bytes)
```

It is not `mean_ce / ln 2`: that is bits per token, and it includes the control targets, which carry no bytes. On a
fixed evaluation set the bytes are a constant, so bpb moves with the CE of the text targets; what it adds is a scale
that does not depend on the tokenizer, for comparing models whose tokenizers differ. Tokens without bytes (motion
codes, video tokens) never count in it; report their CE with `mean_ce` or `type_ce`.

## Evaluating on several GPUs

Every rank runs every evaluation. With `LossEvaluator` the reading is global: `evaluate_loss` sums the numerators
and the denominators of all ranks before the division. (The Trainer itself reduces nothing; see "Writing an
Evaluator".) There are two ways to give the ranks their batches.

- **Each rank reads its own share of a stream.** The documents in the window then depend on the number of ranks, so a
  reading taken on 8 GPUs and one taken on 1 measure different text: compare readings taken on the same number of
  processes. The text modality's `TextEvaluator` works this way.
- **Every rank builds the same frozen batches and evaluates its share of them**, batches rank, rank + N, ... The
  reading is then the same on any number of GPUs, up to rounding in the compute path. exemplars/nano_motion does this
  ([train_t2m.py](../../exemplars/nano_motion/train_t2m.py)). When the batches do not divide evenly, a rank that is
  one short runs a copy of a batch with every target IGNORE, which adds nothing to a metric whose denominator comes
  from the targets (core's three). Five frozen batches on two ranks:

```text
frozen batches    b0   b1   b2   b3   b4         steps = ceil(5 / 2) = 3 on every rank
rank 0            b0        b2        b4
rank 1                 b1        b3        b0'   b0' = b0 with every target IGNORE: adds nothing to the sums
```

```python
me = process_identity()                                      # core.utils
shard = frozen[me.rank::me.world_size]
steps = -(-len(frozen) // me.world_size)                     # the same on every rank
if len(shard) < steps:
    shard.append({**frozen[0], "targets": torch.full_like(frozen[0]["targets"], IGNORE)})
evaluator = LossEvaluator(shard, steps, {"val/ce": mean_ce()})
```

## Writing an Evaluator

Subclass `Evaluator`, give it a schedule (`interval_steps` or `eval_at`), and return a dict of numbers from
`evaluate(model, autocast_ctx)`.

- Turn gradients off yourself (`torch.no_grad()`) around the forward passes. The Trainer switches the System to eval
  mode and nothing more; `LossEvaluator` has them off because `evaluate_loss` does. With gradients on, every forward
  keeps its activations for a backward that never comes, and a trunk given a BlockMask refuses to run uncompiled.
- Enter `autocast_ctx` around whatever should run in the training precision, and only once: `torch.amp.autocast`
  keeps the state it restores on the object, so entering the same object a second time inside the first leaves
  autocast on after the outer exit. An evaluator that delegates to a `LossEvaluator` leaves the context to it.
- Run the same collectives on every rank; under FSDP, the same number of forwards.
- Reduce across ranks yourself where a reading should cover every rank's data (`torch.distributed.all_reduce`). The
  Trainer merges each rank's dict on that rank, and rank 0's values are the ones printed and logged.
- Make `should_eval(step)` a function of the step alone, the same on every rank. The Trainer may call it more than
  once for the same step (`eval_is_due` asks, then `_evaluate` asks again), so it cannot be a trigger that fires once.
- Do not change the model. The Trainer switches eval and train mode around the call.
- An evaluator's own state (the best reading below) is not saved in checkpoints; a resumed run starts it afresh.

This one wraps a `LossEvaluator` and adds the best reading so far, as exemplars/nano_motion's evaluator does:

```python run
from core.evaluation import Evaluator

class CEWithBest(Evaluator):
    def __init__(self, batches, interval_steps):
        self.interval_steps = interval_steps
        self.best = float("inf")
        self.loss = LossEvaluator(batches, len(batches), {"val/ce": mean_ce()})

    def evaluate(self, model, autocast_ctx):
        ce = self.loss.evaluate(model, autocast_ctx)["val/ce"]     # LossEvaluator enters the context
        self.best = min(self.best, ce)
        return {"val/ce": ce, "val/ce_best": self.best}

result = CEWithBest(val_batches, interval_steps=500).evaluate(system, contextlib.nullcontext())
print({name: round(value, 4) for name, value in result.items()})
```
```text output
{'val/ce': 4.1589, 'val/ce_best': 4.1589}
```

Other measurements, such as sampling and scoring generations or a probe on the hidden states, are evaluators of the
same shape that do not go through `evaluate_loss`.

## Reference

| | |
|---|---|
| `Evaluator` | attributes `interval_steps` (int, default 50) and `eval_at` (set of steps or None; overrides the interval); `should_eval(step) -> bool`; `evaluate(model, autocast_ctx) -> dict[str, float]`, which subclasses implement |
| `LossEvaluator(dataloader, eval_steps, metrics)` | an `Evaluator`: `dataloader` any iterable of batches carrying `targets` and whatever the metrics read; `eval_steps` batches per evaluation, the same on every rank; `metrics` `{name: metric}` |
| `evaluate_loss(model, loader, steps, metrics)` | needs `model.prepare_trunk_inputs`, `model.trunk`, `model.head.loss_per_token`; raises `ValueError` for `steps <= 0` and `RuntimeError` when the loader runs out before `steps` batches |
| `mean_ce()`, `type_ce(type_id)`, `bits_per_byte(token_bytes)` | metric factories; `type_ce` raises `ValueError` when the batch has no `target_types` or its shape differs from `targets` |
| `eval_is_due(evaluators, step, last_step)` | in `core.training.trainer`: False with no evaluators, else `last_step` or any `should_eval(step)` |
