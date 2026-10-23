"""Placement and collate — WHERE samples go in a [B, T] batch, and moving every field there.

A prepared sample is {"data": {...}, "meta": {...}}: every `data` value is a tensor whose first dimension is the
sample's shared, nonempty position dimension L (trailing dims may differ: [L], [L, D], ...); `meta` is per-sample
information kept once, never sliced. Collate does not need idx or targets to exist.

    padding_layout(lengths, shape=(B, T))     # one sample per row, left aligned
    sequential_layout(lengths, shape=(B, T))  # whole samples in order; a sample that does not fit the rest of a row
                                              # starts the next row; never truncates
    # both return placements: a list of (sample_index, row, start)

    collate_samples(samples, placements, *, shape=(B, T), pad_values={...})

A layout only reads lengths; collate moves EVERY data field by the same placements and adds the structure facts:
    doc_id       [B, T] int64  the sample's index in `samples` (its order of arrival in this batch) at real
                               positions, -1 at padding — `doc_id >= 0` IS input validity
    doc_pos      [B, T] int64  offset inside the prepared sample, 0..L-1 (0 at padding; validity is doc_id)
    sample_meta  list          sample_meta[i] is samples[i]["meta"] (the sample with doc_id == i)
    layout_facts dict          {"one_sample_per_row_left_aligned": bool}, from the placements on the host (no
                               device access); prepare_unpacked_AR_trunk_inputs reads it to refuse a packed batch
    loader_records list        loader_records[i] is samples[i]["loader_record"], present only when the samples
                               carry the loader's records; the loader takes the list out
                               again before the batch goes on
These names and the loader's `state_dict` are reserved: a data field named like one of them is refused before
anything is written. `pad_values` must cover exactly the data fields (no guessed zero for a field someone forgot).
Every sample is placed exactly once, without overlap or overflow, and this is checked here: a project may compute its
own placements and collate with this function, and a wrong placement would otherwise overwrite, split or drop a
sample without an error. The samples of one batch carry the same data fields with the same dtypes (checked: a field
only later samples carry would be dropped, and a later sample's int64 field under a first sample's float one would be
rounded into it, 2**24 + 1 -> 2**24 in float32); a field whose device or trailing shape differs between samples fails
in torch's own cat. Inputs and meta are only read.
"""
import torch

RESERVED = {"doc_id", "doc_pos", "sample_meta", "layout_facts", "loader_records", "state_dict"}


def sample_length(sample):
    data = sample["data"]
    if not data or any(not isinstance(x, torch.Tensor) or x.ndim < 1 for x in data.values()):
        raise ValueError("data must contain tensors with a position dimension")
    lengths = {x.shape[0] for x in data.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise ValueError("all data fields must share a nonempty position dimension")
    return next(iter(lengths))


def padding_layout(lengths, shape):
    """One sample per row, left aligned."""
    B, T = shape
    if B < 1 or T < 1 or not lengths or len(lengths) > B or any(n < 1 or n > T for n in lengths):
        raise ValueError("samples do not fit padding layout")
    return [(i, i, 0) for i in range(len(lengths))]


def sequential_layout(lengths, shape):
    """Whole samples in input order; a sample that does not fit the rest of a row starts the next row; never
    truncates."""
    B, T = shape
    if B < 1 or T < 1 or not lengths:
        raise ValueError("invalid shape or empty sample group")
    row, start, placements = 0, 0, []
    for i, n in enumerate(lengths):
        if n < 1 or n > T:
            raise ValueError("sample length outside row capacity")
        if start + n > T:
            row, start = row + 1, 0
        if row >= B:
            raise ValueError("sample group exceeds batch capacity")
        placements.append((i, row, start))
        start += n
    return placements


def layout_facts(placements):
    """Host-side structure facts derived from the placements."""
    rows = [row for _, row, _ in placements]
    one_per_row = len(rows) == len(set(rows)) and all(start == 0 for _, _, start in placements)
    return {"one_sample_per_row_left_aligned": bool(one_per_row)}


def collate_samples(samples, placements, *, shape, pad_values):
    placements = list(placements)
    lengths = [sample_length(s) for s in samples]
    if not lengths:
        raise ValueError("cannot collate an empty group")
    keys = set(samples[0]["data"])
    if keys & RESERVED:
        raise ValueError(f"reserved data key in sample: {sorted(keys & RESERVED)}")
    if set(pad_values) != keys:
        raise ValueError(f"pad_values must cover exactly the data fields: have {sorted(pad_values)}, "
                         f"data has {sorted(keys)}")
    B, T = shape
    ref = samples[0]["data"]
    device = next(iter(ref.values())).device
    for sample in samples:
        if set(sample["data"]) != keys:                # a field only later samples carry would be dropped
            raise ValueError("samples must have the same data keys in one batch")
        for key, x in sample["data"].items():
            if x.dtype != ref[key].dtype:              # cat would promote, index_copy_ round into sample 0's dtype
                raise ValueError(f"field {key}: dtype {x.dtype}, but {ref[key].dtype} in the batch's first sample")
    seen, occupied = set(), [[False] * T for _ in range(B)]
    for i, row, start in placements:
        if i in seen or not 0 <= i < len(samples) or not 0 <= row < B:
            raise ValueError("duplicate/invalid sample placement")
        end = start + lengths[i]
        if start < 0 or end > T or any(occupied[row][start:end]):
            raise ValueError("out-of-bounds/overlapping sample placement")
        seen.add(i)
        occupied[row][start:end] = [True] * lengths[i]
    if seen != set(range(len(samples))):
        raise ValueError("every supplied sample must be placed exactly once")

    # One flat index, doc ids and in-sample offsets built on the host, then ONE cat + ONE index_copy_ per field:
    # a batch of B samples costs a few kernels, not B x fields.
    order = [i for i, _, _ in placements]
    flat = torch.cat([torch.arange(start, start + lengths[i]) + row * T for i, row, start in placements])
    doc_vals = torch.cat([torch.full((lengths[i],), i, dtype=torch.long) for i in order])
    pos_vals = torch.cat([torch.arange(lengths[i], dtype=torch.long) for i in order])
    indices = torch.stack([flat, doc_vals, pos_vals]).to(device, non_blocking=True)
    flat_d, doc_d, pos_d = indices[0], indices[1], indices[2]

    batch = {}
    for key, r in ref.items():
        values = torch.cat([samples[i]["data"][key] for i in order])
        out = r.new_full((B * T, *r.shape[1:]), pad_values[key])
        out.index_copy_(0, flat_d, values)
        batch[key] = out.view(B, T, *r.shape[1:])
    doc_id = torch.full((B * T,), -1, dtype=torch.long, device=device)
    doc_id.index_copy_(0, flat_d, doc_d)
    doc_pos = torch.zeros((B * T,), dtype=torch.long, device=device)
    doc_pos.index_copy_(0, flat_d, pos_d)
    batch["doc_id"] = doc_id.view(B, T)
    batch["doc_pos"] = doc_pos.view(B, T)
    batch["sample_meta"] = [s.get("meta", {}) for s in samples]
    batch["layout_facts"] = layout_facts(placements)
    if samples[0].get("loader_record") is not None:
        batch["loader_records"] = [s["loader_record"] for s in samples]
    return batch
