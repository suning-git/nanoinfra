# Data pipeline

How samples get from a data source into the batches the Trainer hands to the System. Four parts in a row, each with
one job, and a loader that wires them and keeps the data's resume position:

```text
DataSource            SampleMixer           build_supervised_sample_fn     batcher (layout + collate)
one raw sample   ->   which source     ->   one sample's inputs and   ->   samples placed into a [B, T] batch
at its true length    comes next            targets (e.g. NTP's shift)     padding or packing, + doc_id, doc_pos
                                                                                       |
MixedDataLoader: wires the four, adds the data snapshot (state_dict), moves the batch to the device
                                                                                       |
                                                    the Trainer -> system.loss(batch) -> prepare_trunk_inputs
```

| Piece | Code |
|---|---|
| `DataSource` | [data_source.py](data_source.py) |
| `SequenceRecipe` (fields -> a sequence and its supervision) | [sequence_recipe.py](sequence_recipe.py) |
| `VocabLayout` (the token-ID bands and token types) | [../tokenization/vocab_layout.py](../tokenization/vocab_layout.py) |
| `build_NTP_supervised_sample` | [supervision.py](supervision.py) |
| `SampleMixer` | [sample_mixer.py](sample_mixer.py) |
| `PaddingBatcher`, `PackingBatcher` | [batchers.py](batchers.py) |
| `padding_layout`, `sequential_layout`, `collate_samples` | [layout.py](layout.py) |
| `MixedDataLoader` | [mixed_dataloader.py](mixed_dataloader.py) |
| `ResumableDistributedSampler` | [dist_sampler.py](dist_sampler.py) |
| converting an old list-format snapshot | [../compatibility/loader_snapshot.py](../compatibility/loader_snapshot.py) |
| `synthetic_token_loader` (debugging) | [synthetic_loader.py](synthetic_loader.py) |

Supervision happens one sample at a time, before batching. So a packed row never has "the last token of sample A
predicts the first token of sample B", and padding or packing never changes a sample's targets. How a batch then
becomes the trunk's inputs (the attention mask, the positions) is the System's, in
[../model/model.md](../model/model.md). The concrete sources live with their modality (`TextDataSource` in
`modalities/text/`, `MotionDataSource` and `T2MDataSource` in `modalities/motion/`) or in a project. The examples on
this page run on the CPU, in order, and share their names.

## A running example

A source of four samples, A to D, packed into batches of 2 rows × 6 positions:

```python run
from functools import partial
import torch
from core.data import DataSource, MixedDataLoader, PackingBatcher, build_NTP_supervised_sample

class ListSource(DataSource):
    """A fixed list of samples, read once, in order (one rank only: it has no merge_rank_states)."""
    def __init__(self, token_lists):
        self.samples = [torch.tensor(t) for t in token_lists]
        self.next = 0
    def __iter__(self):
        while self.next < len(self.samples):
            i = self.next
            self.next += 1                  # the state moves before the yield: get_state() right after is current
            yield {"tokens": self.samples[i], "meta": {"item": i}}
    def get_state(self):
        return {"next": self.next}
    def set_state(self, state):
        self.next = state["next"]

source = ListSource([[10, 11, 12, 13], [20, 21, 22], [30, 31, 32, 33, 34, 35], [40, 41, 42]])    # A, B, C, D
loader = MixedDataLoader({"toy": (source, 1.0)}, build_supervised_sample_fn=build_NTP_supervised_sample,
                         create_batcher_fn=partial(PackingBatcher, shape=(2, 6), pad_values={"idx": 0, "targets": -1}))
for batch in loader:
    print("idx    ", batch["idx"].tolist())
    print("targets", batch["targets"].tolist())
    print("doc_id ", batch["doc_id"].tolist())
    print("doc_pos", batch["doc_pos"].tolist())
    print("meta", batch["sample_meta"], "state", batch["state_dict"])
```
```text output
idx     [[10, 11, 12, 20, 21, 0], [30, 31, 32, 33, 34, 0]]
targets [[11, 12, 13, 21, 22, -1], [31, 32, 33, 34, 35, -1]]
doc_id  [[0, 0, 0, 1, 1, -1], [2, 2, 2, 2, 2, -1]]
doc_pos [[0, 1, 2, 0, 1, 0], [0, 1, 2, 3, 4, 0]]
meta [{'item': 0}, {'item': 1}, {'item': 2}] state {'source_states': {'toy': {'next': 3}}}
idx     [[40, 41, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
targets [[41, 42, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
doc_id  [[0, 0, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
doc_pos [[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
meta [{'item': 3}] state {'source_states': {'toy': {'next': 4}}}
```

What happened, sample by sample:

```text
A = 10 11 12 13 -> idx 10 11 12, targets 11 12 13       (n tokens make n - 1 positions: the next-token shift)
B = 20 21 22    -> idx 20 21,    targets 21 22
C = 30 .. 35    -> 5 positions;  D = 40 41 42 -> 2 positions

batch 1  row 0 | A A A B B . |    A and B fit in one row; C does not fit after them and starts row 1
         row 1 | C C C C C . |    D does not fit after C and there is no row 2: D is pending (pulled, not delivered)
         snapshot: the source's state after C, {"next": 3}, though D has been pulled
batch 2  row 0 | D D . . . . |    D first; then the source ends, so this tail batch goes out padded
         row 1 | . . . . . . |
         snapshot: after D, {"next": 4}
```

`doc_id` numbers the samples of a batch in arrival order and is -1 at padding, so `doc_id >= 0` marks the real
positions. `doc_pos` is the position inside the sample. A checkpoint after batch 1 resumes at D, which no batch has
yet carried into training.

## Data sources

A source yields raw samples, one at a time, each at its own true length L:

```python
{"tokens": LongTensor[L],              # required; NTP needs at least 2
 "token_types": LongTensor[L],         # optional: the type of each token (VocabLayout below)
 "supervision_mask": BoolTensor[L],    # optional: True where the token is a prediction target; absent = all
 "meta": {...}}                        # optional: the sample's identity; kept once per sample, never sliced
```

No mechanical padding and no `attention_mask`: batching pads or packs later. A source on the CPU yields CPU tensors;
the loader moves whole batches to the device. Every source in one mix yields the same fields with the same dtypes
(all with `token_types` or all without, ...): collate refuses a batch whose samples differ. `build_NTP_supervised_sample`
fixes this raw form; with a project's own sample function the source and that function agree on whatever form they
like.

```python
class DataSource(ABC):
    def __iter__(self): ...                      # yields raw samples
    def get_state(self): ...                     # this rank's position, or None for a source that keeps none
    def set_state(self, state): ...              # restore the run's state, read under this rank and world size
    def merge_rank_states(self, states): ...     # optional: every rank's state -> the run's state
    def budget_tokens(self): return None         # optional: a token budget, for weight: auto
```

- **`get_state()`** is a position to come back to, also before the first sample: the loader records it when it is
  built and after every sample it pulls. Make the state move before the `yield`, as `ListSource` does: code after a
  `yield` runs only at the next pull, so a state taken right after a pull would be one sample behind.
- **The state goes into `meta.json`** and comes back through `set_state`, so it must survive JSON unchanged: dicts
  with string keys (an int key comes back a string), lists rather than tuples, plain ints and floats (no NumPy
  numbers, no tensors). Nothing checks this when a checkpoint is saved; test it in the source's tests:
  `json.loads(json.dumps(state)) == state` after a few samples.
- **`merge_rank_states`** makes a resume exact on any number of ranks (see "Resume"). Without it, a checkpoint keeps
  rank 0's state, which is exact only when every rank consumed the same number of the source's samples.
- **An indexable dataset** gets all three from `ResumableDistributedSampler` (below), which shards it over the ranks.
- **Built from a config**, a source is constructed as `source_types[type](config, tokenizers)`, where `config` is its
  entry in `data.sources` without `type`, `weight` and `name`, and `tokenizers` is whatever dict the orchestrator
  passes (the text entry passes the tokenizer, the vocabulary layout and the control resolver). So a source meant
  for configs takes `__init__(self, config, tokenizers)`.

The shipped sources count lengths in raw tokens, before the shift: `TextDataSource` and `MotionDataSource` take
`sample_len` (every sample exactly that long), `T2MDataSource` takes `max_sample_len` (an upper bound). To fill rows of
T positions, give T + 1. The motion sources read `rank` and `world_size` from their config (default 0 and 1): an
orchestrator running on several GPUs passes them, or every rank reads the same data.

## Sequences from fields: `SequenceRecipe` and `VocabLayout`

A source with structured samples (a caption and a motion clip, say) assembles each sequence from named fields with a
`SequenceRecipe`, and the recipe's choice of supervised segments becomes the `supervision_mask`. Token types come from
the `VocabLayout`: the shared token-ID space cut into one band per type.

```python run
from core.data import SequenceRecipe
from core.tokenization.vocab_layout import VocabLayout

layout = VocabLayout()
layout.add_range(0, 0, 60)                 # type 0: text, IDs 0-59
layout.add_range(1, 60, 64)                # type 1: control tokens, IDs 60-63
print(layout.vocab_size, layout.n_token_types)

class Controls:                            # stands in for the control modality's resolver: a name -> its ID, or None
    ids = {"bos": 60, "text_start": 61, "text_end": 62, "eos": 63}
    def resolve(self, name):
        return self.ids.get(name)

recipe = SequenceRecipe(template=["bos", "text_start", "text_tokens", "text_end"],
                        supervise=["text_tokens", "text_end"])
raw = recipe.assemble({"text_tokens": [5, 6, 7]}, layout, Controls())
print(raw["tokens"].tolist(), raw["token_types"].tolist(), raw["supervision_mask"].tolist())
sample = build_NTP_supervised_sample(raw)
print(sample["data"]["idx"].tolist(), sample["data"]["targets"].tolist())
```
```text output
64 2
[60, 61, 5, 6, 7, 62] [1, 1, 0, 0, 0, 1] [False, False, True, True, True, True]
[60, 61, 5, 6, 7] [-1, 5, 6, 7, 62]
```

**The template** is a list of names, and each is one of three kinds: a control token (the resolver knows the name) ->
its single ID; a constant (a key of the recipe's `constants`) -> that literal text, encoded once by the modality's
tokenizer (`content_tokenizer`); otherwise a field of the sample -> its token list. A name has exactly one kind: one
that would be two kinds is refused, not settled by precedence.

**What is supervised**, three ways to say it:

| | | |
|---|---|---|
| `supervise: all` (default) | every segment | |
| `supervise: [text_tokens, text_end]` | the segments named | the tags default to the segment names |
| `supervise_tags: [0, 0, 1, 1]` and `supervise: [1]` | the segments whose tag is listed | one tag per template entry, written in a row under the template |

A single string other than `all` is refused (a string would quietly become a set of its letters). For samples that all
have the same structure, as text windows do, `build_fixed_layout(field_lengths, layout, control_resolver)` assembles the
template once and returns the token template, types, mask and each field's slice, so a source only fills the field
per sample. The types are computed once, from a placeholder ID that stands for every position of a field: ID 0 by
default, which in most layouts is a text ID. A field of another band names an ID inside its own band,
`field_dummy_ids={"motion_tokens": layout.offset(1)}` say (the motion source does); filling in the real tokens later
does not change the precomputed types. Each field must appear once in the template, and all of a field's positions
must be of one type. `overhead_tokens(control_resolver)` is the length of everything that is not a field.

**The layout** is pure integer structure, assembled by the orchestrator from the modalities: `add_range(type, start,
end)` per band (bands may not overlap), `vocab_size` (the end of the last band), `n_token_types` (the largest type
plus one, which sizes the trunk's type embedding), `classify_token_types(ids)` (-1 outside every band),
`to_global(type, local_ids)` and `to_local(global_ids)` (each modality numbers its tokens from 0; the layout places
them). Control-token names are not in it: the control modality's resolver owns them (`modalities/control/`), and the
text modality's `TEXT_RECIPE` is its standard recipe.

## Supervision: one sample at a time

`build_NTP_supervised_sample` is next-token prediction for one sample. It takes the raw dict, or the tensors:

```python
build_NTP_supervised_sample(raw_sample)
build_NTP_supervised_sample(tokens, supervision_mask=None, token_types=None, meta=None)   # types, meta: keyword-only
# -> {"data": {"idx": tokens[:-1], "targets": tokens[1:] (not supervised -> IGNORE = -1),
#              "token_types": types[:-1], "target_types": types[1:]},           # the types only when given
#     "meta": {...}}
```

- L raw tokens make L - 1 positions: to fill a row of T positions, a sample needs T + 1 tokens.
- An IGNORE target's input is still real context. After this step the supervision lives only in `targets`.
- A sample with no supervised target left after the shift is refused (`ValueError`, which the loader re-raises
  naming the source and the pull). So every micro-batch has a target, as `LMSystem.loss` requires; a source that may
  produce such a sample filters it.
- The raw form refuses the old pipeline's fields (`loss_mask`, `loss_weights`, `attention_mask`) instead of
  reinterpreting them. Inputs are never modified.

The loader's `build_supervised_sample_fn` is this function, a project function of the same shape, or None when the
sources already yield prepared samples. A prepared sample is `{"data": {...}, "meta": {...}}` in which every `data`
value is a tensor whose first dimension is the sample's L (later dimensions may differ, `[L, D]` say): a two-stream
video objective, for instance, builds its own aligned targets and per-position fields there.

## Mixing: `SampleMixer`

```python
SampleMixer({"text": (text_source, 2), "t2m": (t2m_source, 1)})      # source name -> (source, sample weight)
```

The next sample comes from the active source with the smallest `samples drawn / weight`, ties going to the
mapping's order. With weights 2 and 1 the order is text, t2m, text, text, t2m, text, ...: two text samples per t2m
sample, whatever their lengths. A zero-weight source is never iterated; a source that ends ends the mix (a source that
should loop manages its own epochs). The counts restart on every restore and are not saved: they only shape the order
within one stretch of training, while each source keeps its own position.

Mixing counts samples. A project that wants a token ratio converts it to sample weights itself: weight_i ∝ token
share_i / mean sample length_i. `MixedDataLoader.from_config` derives sample weights from each source's `weight`:

| Config | Sample weights |
|---|---|
| every source `weight: auto` (an omitted weight is `auto`) | proportional to the sources' token budgets (`budget_tokens()`); the total budget is their sum |
| every weight explicit | the weights, normalized |
| some of each | the explicit ones are absolute shares, summing to less than 1; the auto ones share the rest by budget |

An explicit weight on a source that also has a budget is refused as ambiguous. The text and motion sources read their
budget from their `tokens` key; `T2MDataSource` has none. A budget only sets the mixing weights: how long a run
trains is the Trainer's `max_steps` (or its rule for `max_steps: -1`), and the loader prints the total budget for
comparison.

## Batching

Two batchers with the same use and the same output:

```python
PaddingBatcher(samples, shape=(B, T), pad_values=pad_values)     # one sample per row, from column 0
PackingBatcher(samples, shape=(B, T), pad_values=pad_values)     # whole samples one after another, rows filled in order
```

- **`PaddingBatcher`** takes up to B samples per batch, one per row, and never holds one back.
- **`PackingBatcher`** fills the rows in order with whole samples: never split, never across two rows, no going back
  to fill a hole. A sample that does not fit the current row starts the next; one that fits in no row of this batch
  becomes `pending` and is the first sample of the next batch (D in the running example). An exactly full batch goes
  out without pulling one more sample.
- Both refuse a sample longer than a row (cutting a sample is the source's decision, which knows what it means), and
  when the stream ends they hand out the non-empty remainder as a last, padded batch, never an all-padding one.

Underneath, a *layout* reads only the lengths and returns placements `(sample index, row, start column)`:
`padding_layout(lengths, shape)` or `sequential_layout(lengths, shape)`. `collate_samples(samples, placements, *,
shape, pad_values)` copies every data field of every sample to its place, fills the rest from `pad_values`, and adds
the structure. A project may compute its own placements and collate with core's function; collate checks that every
sample is placed exactly once, inside the batch and without overlap, because a wrong placement would otherwise
overwrite or drop a sample silently.

| Batch key | Shape | |
|---|---|---|
| each data field (`idx`, `targets`, `token_types`, `target_types`, ...) | `[B, T, ...]` | the samples' values; padding from `pad_values` |
| `doc_id` | `[B, T]` int64 | the sample's index in this batch (arrival order), -1 at padding |
| `doc_pos` | `[B, T]` int64 | the position inside the sample, 0 .. L - 1; 0 at padding |
| `sample_meta` | list | `sample_meta[i]` is the meta of the sample with `doc_id == i` |
| `layout_facts` | dict | `{"one_sample_per_row_left_aligned": bool}`, from the placements, on the host |
| `state_dict` | dict | the data snapshot, added by `MixedDataLoader` (see "Resume") |

`pad_values` names exactly the data fields: no guessed default for a field someone forgot. Pad `targets` with IGNORE
(-1); the text modality's `ntp_pad_values(layout, control_resolver)` is an example. The structure keys and
`state_dict` are reserved: a data field with one of those names is refused, and so is a batch whose samples carry
different fields or dtypes (a later sample's int64 field cast into a first sample's float one would round
2^24 + 1 to 2^24).

## `MixedDataLoader`

```python
loader = MixedDataLoader({"t2m": (source, 1.0)}, build_supervised_sample_fn=..., create_batcher_fn=..., device=None)
loader = MixedDataLoader.from_config(loader_config, tokenizers, source_types, build_supervised_sample_fn=...,
                                     create_batcher_fn=..., device=None)
# build_supervised_sample_fn, create_batcher_fn and device are keyword-only
```

The direct entry takes the sources by name with their sample weights. `from_config` builds them from
`loader_config["data"]["sources"]`, a list of entries with `type` (a key of `source_types`, the orchestrator's own
mapping, not a global registry), an optional `name` and `weight`, and the source's own keys. A source is named by its
`name`, or by its type when it is the only source of that type; a missing or repeated name is refused.
`create_batcher_fn` carries the batch shape (B rows of T positions, T the model's row length) and the pad values,
written once there, and `device` is where each batch goes after collate (None: it stays where collate put it). Both
`build_supervised_sample_fn` and `create_batcher_fn` are required.

Every `iter(loader)` starts a new batcher, and a sample the old batcher was holding back is then lost unless the
loader is restored first: the Trainer iterates its loader once. The loader also checks every batch on its way out:
each sample carries a record of which source it came from and its pull number, and per source the delivered pulls must
run 1, 2, 3, ... with no gap. A batcher may reorder samples within a batch, but one that delivers a later sample of a
source while holding back an earlier one is refused, because a snapshot taken after the later one would skip the
earlier one on resume. core's batchers never do that; the check is for a project's own batcher. The record rides on
each sample under the top-level key `loader_record` (`{"source", "pull", "state_after"}`), and core's
`collate_samples` gathers the records into `batch["loader_records"]`, which the loader takes out again; a batcher that
builds its batches without core's collate must hand the records back the same way, or the loader fails with a
`KeyError`.

## Resume

`batch["state_dict"]` is this rank's data snapshot, `{"source_states": {name: state}}`: each source's state right after
its last sample that was delivered in a batch (`{"next": 3}` after batch 1 above, though D had been pulled). A source
not drawn since the last restore keeps the state it was restored to; a source not delivered yet keeps the state it had
when the loader was built. So every entry is a position a restore can return to. Each entry is a copy taken at the
pull, which later pulls cannot change.

When the Trainer saves a checkpoint, every rank hands in its snapshot of the last micro-batch of the step;
`save_checkpoint_if_needed` gathers them, and the loader merges them source by source with the source's
`merge_rank_states`. A merged state has two parts:

- the **front**: the furthest rank's position. Nothing from it on was trained by any rank.
- the **leftover**: the samples before the front that no rank trained: what a rank that read less has not read yet
  (packing gives ranks different numbers of samples per step), a pending sample that lies before the front, and
  leftover not yet handed out since the last resume. A pending sample at or after the front is simply read again
  from the main stream.

On restore, the source splits the leftover over the ranks of the new run, hands its share out first, then reads on from
the front. Nothing is skipped or repeated, however the samples were batched and however many GPUs the resume runs on
(within what the source supports: the text source needs every rank of the new run to get a row group of its own);
the order of the samples and the mixing phase differ from an uninterrupted run. The sampler section below works it
through. Two sources merge today: `ResumableDistributedSampler` (and the sources built on it: the motion sources,
nano_multimodal's video rows) and the text modality's `TextDataSource`, whose positions go down to the document and
the token within it. What it keeps exact is every token position: its leftover items are pieces of row groups, read
and tokenized again on resume, so the windows may be cut at other places than in an uninterrupted run; and it assumes
the files and the tokenizer did not change.

Approximate, by contrast: a source without `merge_rank_states` (the checkpoint keeps rank 0's state, so a rank that
consumed more or fewer of its samples replays or skips the difference); the first resume from a checkpoint written
before 0.3.0, which holds rank 0's position and no leftover; the text source's synthetic `data_path` stream. A
project that saves on several GPUs through `save_checkpoint_dcp` itself, without merging the snapshots
(`merge_data_snapshots`, in [../model/checkpoints.md](../model/checkpoints.md)), writes rank 0's own state. A source
that can merge accepts it only when it holds nothing that is rank 0's alone, and then resumes approximately; otherwise
it refuses it, since restoring one rank's part of the reading would lose the others'. For the sampler that is a state
with leftover items, for the text source one that has entered a row group or holds leftover. The model, the
optimizer state and the step resume whatever the data does; what a resume does not restore at all is in
[../model/checkpoints.md](../model/checkpoints.md), "Starting from a checkpoint".

`set_state` gives each source its entry back by name (the names must match the loader's), starts a new mixing phase
and new pull numbers, and keeps no pending sample; a loader that has already read can be restored too. A `None` entry,
from a source that keeps no state, leaves that source where it stands. A snapshot in the list format of nanoinfra
before 0.3.0 (states matched to sources by position) is refused with an error naming the converter,
`core.compatibility`: `accept_list_snapshot(loader, source_order)` converts at load time and leaves the checkpoint as
it is; `upgrade_list_snapshot(checkpoint_dir, source_order)`, or `python -m core.compatibility.loader_snapshot`,
rewrites the checkpoint's `meta.json` once, keeping a backup.

## `ResumableDistributedSampler`

```python
sampler = ResumableDistributedSampler(dataset, seed=0)      # any map-style dataset: len() and indexing
```

An endless stream of dataset indices: the epochs' permutations laid end to end (each from `seed + epoch`, so nothing
random is saved), global position g being the item at place g mod n of epoch g div n's permutation. Rank r of N takes
the item at its position + r, then moves N on, so every item of every epoch goes to exactly one rank and nothing is
dropped at an epoch's end. Each rank's state is `{"seed", "epoch", "index", "leftover", "shard"}`: its position
(epoch × n + index), the leftover items it still has to hand out (each `[epoch, item]`), and which rank of how many it
is. The run's state, from `merge_rank_states`, has no `shard`. Its rank and world size come from the launcher's
environment when it is built; a source that shards otherwise (every rank reading the whole validation set) sets
`sampler.rank` and `sampler.world_size` before restoring.

Two ranks over a dataset of ten items, where rank 1 read fewer samples than rank 0 (longer samples under packing,
say), then a resume on three ranks:

```python run
from core.data import ResumableDistributedSampler

items = list(range(10))
ranks = [ResumableDistributedSampler(items, seed=0) for _ in range(2)]
for r, sampler in enumerate(ranks):
    sampler.rank, sampler.world_size = r, 2               # what torchrun's environment would say
streams = [iter(s) for s in ranks]
print("epoch 0:", ranks[0]._perm(0))
print("rank 0 trained", [next(streams[0]) for _ in range(3)], " rank 1 trained", [next(streams[1])])
run = ranks[0].merge_rank_states([s.state_dict() for s in ranks])
print("the run's state:", run)

def rest_of_epoch(sampler):          # what this rank reads before it reaches epoch 1
    out, stream = [], iter(sampler)
    while sampler.leftover or sampler.epoch * len(items) + sampler.index + sampler.rank < len(items):
        out.append(next(stream))
    return out

for r in range(3):
    sampler = ResumableDistributedSampler(items, seed=0)
    sampler.rank, sampler.world_size = r, 3
    sampler.load_state_dict(run)
    print(f"resume rank {r} of 3 reads", rest_of_epoch(sampler))
```
```text output
epoch 0: [4, 1, 7, 5, 3, 9, 0, 8, 6, 2]
rank 0 trained [4, 7, 3]  rank 1 trained [1]
the run's state: {'seed': 0, 'epoch': 0, 'index': 6, 'leftover': [[0, 5], [0, 9]]}
resume rank 0 of 3 reads [5, 0, 2]
resume rank 1 of 3 reads [9, 8]
resume rank 2 of 3 reads [6]
```

```text
position     0  1  2  3  4  5  6  7  8  9
item         4  1  7  5  3  9  0  8  6  2
rank 0       *     *     *                  positions 0, 2, 4 trained; next at 6
rank 1          *                           position 1 trained; next at 3 (3 and 5 still unread)
merged       front = 6; leftover = positions 3 and 5 = items 5 and 9
resume on 3  rank 0: leftover 5, then positions 6 (0) and 9 (2)
             rank 1: leftover 9, then position 7 (8)
             rank 2: position 8 (6)
```

The rest of epoch 0, items 5, 9, 0, 8, 6 and 2, is read once each; 4, 7, 3 and 1 are not read again.
`load_state_dict` refuses a state whose seed differs, a position beyond the dataset, and one rank's state with
leftover items from a run of several ranks (it holds only that rank's share; restore the merged state).

## `synthetic_token_loader`

A debugging loader with no tokenizer and no data: endless batches `{idx, token_types, targets, target_types,
state_dict}` of random tokens, or of a file's bytes, with the next-token shift and `token_types` all 0. Passing
a batch's `state_dict` (step, offset, the generator's state) back as `resume_state_dict` starts a new generator where
that one stopped; it has no `set_state`, so the Trainer cannot restore it from a checkpoint. It has no `doc_id` or `layout_facts`, so use it with the unpacked
input preparation. [../training/training.md](../training/training.md) trains on it in its smallest example.

## Reference

| | |
|---|---|
| `DataSource` | abstract: `__iter__`, `get_state`, `set_state`; optional `merge_rank_states`, `budget_tokens` |
| `SequenceRecipe(template, supervise="all", supervise_tags=None, constants=None)` | `assemble(fields, layout, control_resolver, content_tokenizer=None)`, `build_fixed_layout(...)`, `overhead_tokens(...)` |
| `VocabLayout()` | `add_range`, `vocab_size`, `n_token_types`, `offset`, `classify_token_types`, `to_global`, `to_local`; `IGNORE_INDEX = -1` |
| `build_NTP_supervised_sample(tokens, supervision_mask=None, *, token_types=None, meta=None)` | or one raw dict; `ValueError` for fewer than 2 tokens, a non-integer tensor, a misshaped mask or types, no target after the shift, old fields |
| `SampleMixer(weighted_sources)` | iterating yields `(name, raw sample)`; `reset_phase()` |
| `PaddingBatcher(stream, *, shape, pad_values)`, `PackingBatcher(...)` | iterating yields collated batches; `ValueError` for a sample longer than a row |
| `padding_layout(lengths, shape)`, `sequential_layout(lengths, shape)` | -> placements `(sample, row, start)` |
| `collate_samples(samples, placements, *, shape, pad_values)` | -> a batch; `ValueError` for bad placements, reserved names, `pad_values` not naming exactly the fields, mixed fields or dtypes |
| `MixedDataLoader(weighted_sources, *, build_supervised_sample_fn, create_batcher_fn, device=None)` | `from_config(...)`, `get_state()`, `set_state(state)`, `merge_rank_snapshots(snapshots)` |
| `ResumableDistributedSampler(dataset, seed=0)` | `state_dict()`, `merge_rank_states(states)`, `load_state_dict(state)`; attributes `rank`, `world_size` |
| `synthetic_token_loader(B, T, split="train", device="cuda", resume_state_dict=None, data_path=None, vocab_size=65536, seed=0)` | an endless generator of batches |
