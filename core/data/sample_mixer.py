"""SampleMixer — which source the next sample comes from.

Mixing counts SAMPLES: every successful draw counts one sample for its source, whatever its length; a project that
wants a token ratio converts it to sample weights itself (weight_i ∝ token share_i / mean sample length_i). The mixer
never reads lengths, YAML or batches — it only picks sources.

    mixer = SampleMixer({"text": (text_src, 2), "t2m": (t2m_src, 1)})
    for name, raw_sample in mixer: ...

One mapping, source name -> (source, weight): a source and its weight are written together, so they cannot disagree
on the names; the name, not a list position, is also what pairs the loader's snapshot with its source.

Rules:
  - the next source is the active one with the smallest samples_drawn / weight (ties: the mapping's order);
  - a zero-weight source is never iterated;
  - a source that ends ends the mix (no reweighting, no restart; a source that should loop manages its own epochs);
  - the counts are the phase of THIS scheduling run only: not written to a checkpoint, reset on every restore
    (`reset_phase`); sources keep their own resume state. A sample pulled but held back by a batcher (pending) was
    drawn and is counted once; delivering it later does not count again.
"""
import math


class SampleMixer:
    def __init__(self, weighted_sources):
        if not weighted_sources:
            raise ValueError("SampleMixer needs at least one source")
        for name, entry in weighted_sources.items():
            # unpacking a source given alone would iterate it: read up to three samples, then a misleading error
            if not (isinstance(entry, tuple) and len(entry) == 2):
                got = f"a tuple of {len(entry)}" if isinstance(entry, tuple) else f"a {type(entry).__name__}"
                raise TypeError(f"source {name!r}: each value is a (source, weight) pair, not {got}")
        # in the mapping's order: ties follow it
        self.sources = {name: source for name, (source, _) in weighted_sources.items()}
        self.weights = {name: float(weight) for name, (_, weight) in weighted_sources.items()}
        if any(not math.isfinite(x) or x < 0 for x in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("weights must be finite, nonnegative and not all zero")
        self.samples_drawn_per_source = {name: 0 for name in self.sources}
        self.active = [name for name, x in self.weights.items() if x > 0]

    def reset_phase(self):
        """Start a new scheduling phase (on restore): all counts back to zero."""
        self.samples_drawn_per_source = {name: 0 for name in self.sources}

    def next_source(self):
        drawn, w = self.samples_drawn_per_source, self.weights
        return min(self.active, key=lambda name: drawn[name] / w[name])   # min keeps the first of equal keys

    def __iter__(self):
        iters = {name: iter(self.sources[name]) for name in self.active}
        while True:
            name = self.next_source()
            try:
                raw = next(iters[name])
            except StopIteration:
                return
            self.samples_drawn_per_source[name] += 1
            yield name, raw
