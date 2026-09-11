"""
inference.py — one sampler, three lines.

    load(line)                      newest checkpoint -> a runnable System
    generate(...)                   band-masked autoregressive sampling
    Session                         the interactive video loop: one action -> one frame

BAND-MASKED SAMPLING. Over the shared vocabulary the model could emit any id, but
after `video_start` only video-band ids are legal, and after `motion_start` only
motion-band ids plus the closing tag. Masking the logits to the band ENFORCES the
grammar the row taught rather than hoping the model learned to stay inside it. The
band is the only per-line parameter: same function, three callers.

THE THREE SAMPLERS, ported from exemplars/nano_world_model/inference/sample.py:

    plain    recompute the whole prefix every token. O(n^2). The reference.
    cached   core's KVCache: prefill once, then one-token forwards.
    static   core's StaticKVCache: every shape pinned, so
             torch.compile(mode="reduce-overhead") captures CUDA graphs.

All three are kept, and that is the point. A decode path that is fast and subtly
wrong is worse than a slow one, and the only thing that makes the fast one
trustworthy is agreeing with the slow one — see tests/test_inference.py, which
checks the way that survives kernel changes (teacher-forced argmax and greedy
decode, not draw-for-draw identity at finite temperature, where bf16 noise flips
near-tied candidates either way).

WHAT IS NEW HERE. The interactive world-model server this is modelled on is mature
but BLOCK-DIFFUSION only; its own file header says the AR branch should be ported
from exemplars/nano_world_model "the day an AR model is worth eyeballing". This
project is that day, and joining the two halves is the work.

AND ONE CONSEQUENCE OF BEING AR. 256 codes per frame at one forward per token is
roughly half a second per frame; short of speculative decoding there is little to
reclaim. So the interaction is PULL, not push — the client asks for a frame and
watches it arrive. Session is written for that. The rate also buys something 40ms
cannot: it is long enough to stream a frame's 256 integers into the UI as they are
drawn, so a student watches a frame being written one token at a time.
"""

from pathlib import Path

import torch

from core.model.kv_cache import STATIC, KVCache, StaticKVCache

from projects.nano_multimodal import assembly, spec


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def checkpoints(line=None):
    """Loadable step_* directories under the project's models root, BEST first.

    Best, not newest, and that is not a nicety: a small model on a small corpus
    overfits, and its last checkpoint is its worst one. Measured on this project's
    own motion run — val CE 1.41 at its floor, 4.75 after 20k steps. Serving the
    newest one would show a student the worst model this project can produce and
    invite them to conclude the method does not work.

    "Best" is read from the run's metrics.jsonl (metrics.py writes it), matching each
    saved step to the val CE recorded at it. A run with no metrics falls back to
    newest-first, which is the only honest ordering when nothing was measured.

    Directories without a meta.json are skipped rather than returned: checkpoint
    retention deletes old steps WHILE a run continues, so a browser or a test can
    otherwise pick a path that stops existing between the listing and the load.
    """
    root = spec.MODELS_ROOT
    if not root.exists():
        return []
    out = []
    for run in sorted(root.glob(f"nmm_{line}_*" if line else "nmm_*")):
        val = _val_by_step(run / "metrics.jsonl")
        out += [(val.get(_step_of(p), float("inf")), -_step_of(p), p)
                for p in run.glob("step_*") if (p / "meta.json").exists()]
    # Sorted ACROSS runs, not within each. Sorting per run and concatenating puts the
    # alphabetically-first run's best checkpoint at the head of the list, which is not
    # the best checkpoint — and with two sizes of the same line on disk (a d6 and a
    # d12) that silently served the worse one.
    out.sort(key=lambda t: t[:2])
    return [p for _, _, p in out]


def _step_of(p):
    try:
        return int(p.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _val_by_step(path):
    """metrics.jsonl -> {step: val CE}. Any val/* metric; there is one per line."""
    import json
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        keys = [k for k in rec if k.startswith("val/") and not k.endswith("_best")]
        if keys and isinstance(rec.get("step"), int):
            out[rec["step"]] = float(rec[keys[0]])
    return out


_GRAPHED = set()


def _enable_graphs(system):
    """Compile the trunk for CUDA-graph capture — ONLY ever from the static path.

    `mode="reduce-overhead"` makes each replay write its output into the SAME buffer,
    which is exactly what makes it fast and exactly what makes it unusable anywhere
    else: the dynamic-cache path keeps a `logits` tensor alive across iterations, and
    on the next replay that tensor's storage is gone. torch says so — "accessing
    tensor output of CUDAGraphs that has been overwritten by a subsequent run" — but
    only at the moment of use, several layers away from the decision that caused it.

    So the decision lives HERE, next to the only path that may make it, rather than
    in a `fast=True` flag a caller could hand to the wrong sampler.
    """
    if id(system) in _GRAPHED:
        return
    system.trunk.compile(mode="reduce-overhead")
    _GRAPHED.add(id(system))


def load(line, ckpt=None, device="cuda", config=None):
    """A checkpoint -> (system, vocab). The checkpoint self-describes its
    architecture (core's load_system reads meta.json), so nothing here restates a
    geometry that could disagree with the weights.

    The VOCAB still comes from the line's config, and the two are locked together:
    a checkpoint trained on a different band assembly has a different vocab_size,
    and loading it against this one would be a silent mismatch of meaning rather
    than a shape error.
    """
    from core.training.model_setup import load_system

    candidates = [Path(ckpt)] if ckpt else checkpoints(line)
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint for line {line!r} under {spec.MODELS_ROOT} — train one "
            f"first: python -m projects.nano_multimodal.train --config-name {line}")
    config = config or assembly.load_config(line)
    vocab = assembly.assemble_vocab(line, config)

    # load_system hands back an EAGER trunk (assembly never compiles). The training
    # compile (`dynamic=True`) would be wrong here anyway: dynamic shapes are exactly
    # what stops CUDA graphs from being captured. The decode compile is a different
    # mode applied to a different path, and it happens in _enable_graphs, from the
    # static sampler only.
    # Retention can delete the chosen directory between the listing and the load
    # while a run is still going. Try the next candidate rather than failing a
    # browser request on a race the reader cannot see.
    last = None
    for cand in candidates[:4]:
        try:
            setup = load_system(str(cand), sequence_len=vocab.sequence_len,
                                head_ce="naive")
            ckpt = cand
            break
        except (FileNotFoundError, OSError) as e:
            last = e
    else:
        raise FileNotFoundError(f"no loadable checkpoint for {line!r}: {last}")
    system = setup["system"]
    if setup["gpt_config"].vocab_size != vocab.layout.vocab_size:
        raise ValueError(
            f"{ckpt} was trained with vocab_size={setup['gpt_config'].vocab_size} but "
            f"line {line!r} assembles {vocab.layout.vocab_size}. Same integers, "
            f"different meanings — the bands moved.")
    system.eval()
    return system, vocab


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def _band(vocab, name):
    from projects.nano_multimodal.assembly import TYPE_IDS
    return vocab.layout.ranges[TYPE_IDS[name]]


def _draw(logits, lo, hi, stop_id, temperature, top_k, g):
    """Mask to the vocab band (plus the stop id, if allowed), then temperature + top-k."""
    masked = torch.full_like(logits, float("-inf"))
    masked[lo:hi] = logits[lo:hi]
    if stop_id is not None:
        masked[stop_id] = logits[stop_id]
    masked = masked / max(temperature, 1e-6)
    if top_k:
        v, _ = torch.topk(masked, min(top_k, int((masked > float("-inf")).sum())))
        masked[masked < v[-1]] = float("-inf")
    return int(torch.multinomial(torch.softmax(masked, -1), 1, generator=g))


@torch.no_grad()
def generate(system, vocab, prefix_ids, band, *, stop=None, max_new=None,
             fixed_len=None, temperature=1.0, top_k=40, seed=0, device="cuda",
             sampler="cached", cache=None, on_token=None, graphs=True):
    """Band-masked AR generation. Returns (global ids generated, cache).

    band       band NAME ("text" / "motion" / "video"); the layout turns it into ids
    stop       control token NAME that ends generation, or None
    fixed_len  emit exactly this many and never stop early — right for a fixed-shape
               codec, where a latent frame is always the same number of codes
    on_token   callback(i, global_id) per token, for streaming the row into a UI
    graphs     static sampler only: compile the trunk for CUDA-graph replay. True is
               what production wants. False keeps the static SHAPES but leaves the
               kernels eager, which is the arm that must agree with the dynamic paths
               EXACTLY — compiling introduces inductor numerics, and separating the
               two is how a real disagreement stays distinguishable from a rounding
               one (tests/test_inference.py does exactly that).
    """
    lo, hi = _band(vocab, band)
    stop_id = vocab.resolver.resolve(stop) if stop else None
    seq_len = system.trunk.config.sequence_len
    n = fixed_len if fixed_len is not None else (max_new or seq_len - len(prefix_ids))
    allow_stop = stop_id is not None and fixed_len is None
    g = torch.Generator(device=device).manual_seed(seed)
    system.eval()

    if sampler == "static":
        if graphs:
            _enable_graphs(system)
        return _static(system, vocab, prefix_ids, lo, hi, n, temperature, top_k, g,
                       device, cache, seq_len, on_token)

    if id(system) in _GRAPHED:
        raise RuntimeError(
            f"this system was compiled for CUDA-graph replay (the static path) and "
            f"cannot also serve sampler={sampler!r}: a graph replay overwrites the "
            f"previous output buffer, which the dynamic paths hold across iterations. "
            f"Load a second copy, or use sampler='static'.")
    use_cache = sampler == "cached"
    if use_cache and cache is None:
        cache = KVCache.for_model(system.trunk.config, 1, seq_len)

    def fwd(ids):
        toks = torch.tensor([ids], dtype=torch.long, device=device)
        types = vocab.layout.classify_token_types(toks)
        return system.head(system.trunk(toks, token_types=types,
                                        kv_cache=cache if use_cache else None))[0, -1].float()

    seq, out = list(prefix_ids), []
    logits = fwd(seq) if use_cache else None
    for i in range(n):
        if not use_cache:
            logits = fwd(seq)
        nxt = _draw(logits, lo, hi, stop_id if allow_stop else None, temperature, top_k, g)
        if allow_stop and nxt == stop_id:
            break
        out.append(nxt)
        if on_token is not None:
            on_token(i, nxt)
        if use_cache:
            logits = fwd([nxt])
        else:
            seq.append(nxt)
    return out, cache


@torch.no_grad()
def _static(system, vocab, prefix_ids, lo, hi, n, temperature, top_k, g, device,
            cache, seq_len, on_token):
    """The static-shape twin: every shape and address pinned, so CUDA graphs capture.

    No stop token here, deliberately. A data-dependent stop is a host sync per
    token, which is the overhead this path exists to remove — and a fixed-shape
    codec has nothing to stop early for.
    """
    if cache is None:
        cache = StaticKVCache.for_model(system.trunk.config, 1, seq_len)
    system.trunk.attach_kv_cache(cache)
    tok_buf = torch.empty((1, 1), dtype=torch.long, device=device)   # fixed address

    def fwd(toks):
        types = vocab.layout.classify_token_types(toks)
        return system.head(system.trunk(toks, token_types=types, kv_cache=STATIC))[0, -1].float()

    logits = fwd(torch.tensor([list(prefix_ids)], dtype=torch.long, device=device))
    out = []
    for i in range(n):
        # The sampling ops REPLICATE the dynamic path exactly — multinomial over the
        # same category layout, so the same seed draws the same ids. Two differences
        # change no math: top-k count is the constant min(top_k, band width) (the
        # dynamic path recomputes the same number with a GPU->CPU sync per token), and
        # the drawn token stays a GPU tensor until the segment ends (one sync per
        # segment instead of n).
        mask = torch.full_like(logits, float("-inf"))
        mask[lo:hi] = logits[lo:hi]
        logits = mask / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, hi - lo))
            logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)   # [1], on GPU
        out.append(nxt)
        tok_buf.copy_(nxt.view(1, 1))
        logits = fwd(tok_buf)
    ids = [int(t) for t in torch.cat(out).cpu()]
    if on_token is not None:
        for i, t in enumerate(ids):
            on_token(i, t)
    return ids, cache


# --------------------------------------------------------------------------
# the three lines, as callers
# --------------------------------------------------------------------------

def sample_text(system, vocab, prompt, max_new=120, **kw):
    """prompt string -> (continuation ids, decoded text)."""
    tok = vocab.tokenizers["text"]
    r = vocab.resolver
    prefix = [r.resolve("bos"), r.resolve("text_start")] + tok.encode(prompt)
    ids, _ = generate(system, vocab, prefix, "text", stop="text_end",
                      max_new=max_new, **kw)
    return ids, tok.decode(ids)


def sample_motion(system, vocab, caption, max_new=None, max_text_tokens=32, **kw):
    """caption -> the full assembled row (prompt + generated motion), so the caller
    can hand it straight to decode.render — the same row shape the browser dissects."""
    tok, r = vocab.tokenizers["text"], vocab.resolver
    text_ids = tok.encode(caption)[:max_text_tokens]
    prefix = ([r.resolve("bos"), r.resolve("text_start")] + text_ids
              + [r.resolve("text_end"), r.resolve(spec.MOTION_START)])
    n = max_new or (vocab.sequence_len - len(prefix) - 2)
    ids, _ = generate(system, vocab, prefix, "motion", stop=spec.MOTION_END,
                      max_new=n, **kw)
    return prefix + ids + [r.resolve(spec.MOTION_END)]


class Session:
    """An ENDLESS interactive video rollout.

    Holds the KV cache across frames and owns the ROW GRAMMAR — after each latent
    frame come `td` action tokens, then the next frame's codes — so the server stays
    transport and never re-derives the row.

    RE-ANCHORING. The model was trained on rows holding `n_latent` latent frames, so
    a rollout cannot simply run forever inside one row. When the row fills, this
    starts a NEW row whose first `carry` latent frames are the last `carry` frames of
    the old one — their CODES copied verbatim, together with the actions that
    produced them. Two things follow, and both matter:

      * context SURVIVES the re-anchor. Restarting from a single frame would make the
        world Markov: everything older than one frame forgotten, every re-anchor a
        chance to forget the room you are standing in.
      * no codec round-trip. The obvious implementation re-encodes the last decoded
        PIXELS, which passes the state through decode->encode and accumulates that
        error every window. Copying codes has no such drift.

    Both decisions were settled on the block-diffusion research line before this
    project existed (its re-anchor is called reset_with_history); what follows is
    the autoregressive twin of that method.

    `carry` trades context against cost: carrying more leaves less room before the
    next re-anchor, and each re-anchor costs one prefill. At the default contract,
    carry=2 leaves room for 3 frames per window.
    """

    def __init__(self, system, vocab, seed_codes, seed=0, temperature=1.0, top_k=40,
                 device="cuda", sampler="static", carry=2):
        self.system, self.vocab = system, vocab
        self.shape = spec.video_shape()
        self.device, self.sampler = device, sampler
        self.temperature, self.top_k, self.seed = temperature, top_k, seed
        self.carry = max(1, min(int(carry), self.shape["n_blocks"]))
        r = vocab.resolver
        self.v_off = _band(vocab, "video")[0]
        self.a_off = _band(vocab, "action")[0]
        self.ctrl = [r.resolve("bos"), r.resolve("video_start")]
        self.frames_done = 0            # frames generated inside the CURRENT row
        self.total_frames = 0           # frames generated since the session began
        self.reanchors = 0
        self._cache = None
        self._used = 0                  # cache slots consumed in the CURRENT row
        # The rolling window of latent frames, as LOCAL codes, and the actions
        # between them. Only the last `carry` of each survive a re-anchor.
        self.window = [list(int(c) for c in seed_codes)]
        self.acts = []
        self._pending = self.ctrl + [self.v_off + int(c) for c in seed_codes]

    @property
    def room(self):
        """Frames this row can still take before a re-anchor is needed."""
        return self.shape["n_blocks"] - self.frames_done

    def _reanchor(self):
        """Start a new row from the tail of the old one. Codes copied verbatim."""
        n = min(self.carry, len(self.window))
        lat = self.window[-n:]
        acts = self.acts[-(n - 1):] if n > 1 else []
        prefix = list(self.ctrl)
        for k, codes in enumerate(lat):
            if k:
                prefix += [self.a_off + int(acts[k - 1])] * self.shape["td"]
            prefix += [self.v_off + int(c) for c in codes]
        self.window, self.acts = list(lat), list(acts)
        self._pending = prefix
        self._cache = None              # a new row means a new cache
        self.frames_done = n - 1        # the carried frames occupy their blocks
        self._used = 0                  # counted by step(), which feeds `prefix`
        self.reanchors += 1

    def step(self, action_id, on_token=None):
        """One action -> the next latent frame's LOCAL codes. Never runs out."""
        if self.room <= 0:
            self._reanchor()
        td, cpf = self.shape["td"], self.shape["codes_per_frame"]
        a = int(action_id) % spec.N_ACTIONS

        def plan():
            # Every token this step will INSERT into the cache: whatever has not been
            # fed yet (the re-anchor prefix, on the first step of a row) plus this
            # frame's actions, plus the codes about to be generated.
            p = self._pending + [self.a_off + a] * td
            return p, self._used + len(p) + cpf

        prefix, used = plan()
        # Counted on the HOST, before the forward. StaticKVCache asserts on the
        # DEVICE, which surfaces as an asynchronous "device-side assert triggered"
        # pointing at whatever kernel ran next — a useless post-mortem.
        if used > self.vocab.sequence_len:
            self._reanchor()
            prefix, used = plan()
            assert used <= self.vocab.sequence_len, (
                f"even a freshly re-anchored row needs {used} of "
                f"{self.vocab.sequence_len} slots — carry={self.carry} is too large "
                f"for this clip contract")
        self._used = used
        ids, self._cache = generate(
            self.system, self.vocab, prefix, "video",
            fixed_len=self.shape["codes_per_frame"],
            temperature=self.temperature, top_k=self.top_k,
            seed=self.seed + self.total_frames, device=self.device,
            sampler=self.sampler, cache=self._cache, on_token=on_token)
        # WHAT GOES IN NEXT TIME. With a cache, the generated codes are ALREADY in it
        # — every one was inserted by the single-token forward that followed it — so
        # only tokens the cache has not seen (next frame's actions) may be fed again.
        # Feeding the codes back would insert each of them twice, and the cache
        # overflows two frames later with a device-side assert that points at an
        # unrelated kernel.
        self._pending = [] if self._cache is not None else prefix + ids
        lo = self.v_off
        codes = [i - lo for i in ids]
        self.window.append(codes)
        self.acts.append(a)
        self.frames_done += 1
        self.total_frames += 1
        return codes
