"""PaddingBatcher / PackingBatcher — a stream of prepared samples in, [B, T] batches out.

Two classes with the same use, the same batch contract (layout.collate_samples) and no shared base class:

    batches = PaddingBatcher(samples, shape=(B, T), pad_values=pad_values)   # one sample per row
    batches = PackingBatcher(samples, shape=(B, T), pad_values=pad_values)   # whole samples, next-fit

PackingBatcher fills rows in order with whole samples (never split, never across rows, no looking back for holes);
a sample that does not fit the current row starts the next; when the last row cannot take it either it becomes
`pending` — pulled from upstream, not yet delivered — and is the first sample of the next batch. A batch that is
exactly full is handed out without pulling one more sample. PaddingBatcher pulls at most B samples per batch, one
per row, and so never holds one back. Both: a sample longer than a row is an error (cutting belongs upstream, with
the sample's meaning); an upstream that ends yields its non-empty tail batch (padded) and then stops — never an
all-padding batch. A batcher keeps no record of what it delivered: the loader's records ride on the samples and come
back inside the batch (collate's `loader_records`), so a batcher that collates with core's collate_samples hands
them back without doing anything.
"""
from .layout import collate_samples, padding_layout, sample_length, sequential_layout


class PackingBatcher:
    def __init__(self, sample_stream, *, shape, pad_values):
        self.stream, self.shape, self.pad_values = sample_stream, tuple(shape), dict(pad_values)
        self.pending = None

    def __iter__(self):
        it = iter(self.stream)
        B, T = self.shape
        exhausted = False
        while True:
            group, lengths, row, col = [], [], 0, 0
            while True:
                if self.pending is not None:
                    s = self.pending
                elif exhausted:
                    s = None
                else:
                    try:
                        s = next(it)
                    except StopIteration:
                        exhausted, s = True, None
                if s is None:
                    break
                L = sample_length(s)
                if L > T:
                    raise ValueError(f"sample is longer than a row ({L} > {T}); split or truncate upstream")
                if col + L > T:
                    row, col = row + 1, 0
                if row >= B:
                    self.pending = s
                    break
                self.pending = None
                group.append(s)
                lengths.append(L)
                col += L
                if row == B - 1 and col == T:          # exactly full: do not pull one more
                    break
            if not group:
                return                                   # nothing left; never an all-padding batch
            yield collate_samples(group, sequential_layout(lengths, self.shape),
                                  shape=self.shape, pad_values=self.pad_values)


class PaddingBatcher:
    def __init__(self, sample_stream, *, shape, pad_values):
        self.stream, self.shape, self.pad_values = sample_stream, tuple(shape), dict(pad_values)

    def __iter__(self):
        it = iter(self.stream)
        B, T = self.shape
        while True:
            group, lengths = [], []
            for _ in range(B):
                try:
                    s = next(it)
                except StopIteration:
                    break
                L = sample_length(s)
                if L > T:
                    raise ValueError(f"sample is longer than a row ({L} > {T}); split or truncate upstream")
                group.append(s)
                lengths.append(L)
            if not group:
                return
            yield collate_samples(group, padding_layout(lengths, self.shape),
                                  shape=self.shape, pad_values=self.pad_values)
            if len(group) < B:
                return                                   # a finite stream's tail batch was the last
