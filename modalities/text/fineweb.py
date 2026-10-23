"""
FineWeb streaming loader — text's production data path.

Streams an EXPLICIT list of FineWeb **parquet** files, tokenizes documents on
the fly (BPE, bos-prepended, multi-threaded), packs the token stream into
``[B, T]`` windows. Used by ``TextDataSource`` and ``TextEvaluator``. The
``token_data_loader`` entry point also dispatches to core's modality-free
synthetic/byte-stream loader (``source="synthetic"`` / ``data_path=...``) so
the callers stay source-agnostic for smoke tests.

Which files constitute a split is NOT decided here — it is declared in config
and resolved by ``modalities.text.datasets.resolve_split``. This module only
consumes the resulting list. (The predecessor inferred ``val`` as "whichever
shard sorts last", so downloading a shard silently changed the ruler; see
``datasets.py`` for the full account.)

Two readings of the parquet files:

- ``exact_windows`` (TextDataSource): windows of text tokens, each with the exact
  position after it, down to the document and the token; the state merges over the
  ranks (``merge_exact_states``), so a resume on any number of ranks neither skips
  nor repeats a token. See the section "Exact positions" below.
- ``token_data_loader`` (the raw [B, T] stream of TextEvaluator's raw ruler, and the
  synthetic paths): state ``{files, pq_idx, rg_idx, epoch}``, resumed at the next
  row group (approximate; nothing resumes it in training).
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Optional

import torch

from core.data.synthetic_loader import synthetic_token_loader
from core.utils import process_identity


# ---------------------------------------------------------------------------
# Real on-the-fly FineWeb parquet source
# ---------------------------------------------------------------------------

def _default_data_dir() -> str:
    from core.utils import get_base_dir
    return os.environ.get("NANOINFRA_DATA_DIR") or os.path.join(get_base_dir(), "base_data")


def list_parquet_files(data_dir: Optional[str] = None) -> list[str]:
    """Sorted full paths to all parquet shards in ``data_dir`` (default base_data)."""
    data_dir = data_dir or _default_data_dir()
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"No FineWeb parquet directory at {data_dir}. "
            f"Download shards there (e.g. HuggingFaceFW/fineweb sample/10BT), "
            f"or pass source='synthetic'/data_path=... for offline tokens."
        )
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")
    return [os.path.join(data_dir, f) for f in files]


def _document_batches(parquet_paths, resume_state_dict, tokenizer_batch_size, rank, world_size):
    """
    Infinite iterator over document batches (lists of text strings) from parquet.

    Handles DDP sharding (row-groups strided by world_size, offset by rank) and
    approximate resume. Yields ``(text_batch, (pq_idx, rg_idx, epoch))``.

    Resume finds the file by its POSITION in ``parquet_paths`` (``pq_idx``), not by
    name: it assumes the list up to that position is the one the state was saved
    with. Appending shards (train = ``rest`` growing) keeps that true. Inserting or
    removing a shard that sorts earlier, or resuming on a machine that holds other
    shards, puts the position on another file: no error, some data read again or
    skipped. The state's ``files`` is recorded to diagnose that and deliberately not
    checked. Checked on the real shards for one process within one epoch (2026-10-10):
    with the list unchanged or appended to, the resumed stream continues the saved one
    with nothing read twice; it skips the rest of the current row group and the tokens
    still buffered. Not checked across ranks: every rank resumes from the position the
    saved state records, so a rank that was ahead of or behind it re-reads or skips up
    to about a row group; nor when the number of ranks changes.
    """
    import pyarrow.parquet as pq

    if not parquet_paths:
        raise ValueError("no parquet files for this split (see modalities/text/datasets.py)")

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict else 0
    resume_rg_idx = resume_state_dict.get("rg_idx") if resume_state_dict else None
    epoch = resume_state_dict.get("epoch", 1) if resume_state_dict else 1
    first_pass = True

    while True:  # multi-epoch, infinite
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            pf = pq.ParquetFile(parquet_paths[pq_idx])
            if first_pass and resume_rg_idx is not None and pq_idx == resume_pq_idx:
                # advance one stride past the resumed row-group so we don't repeat
                base = resume_rg_idx // world_size + 1
                rg_idx = base * world_size + rank
                resume_rg_idx = None
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
            else:
                rg_idx = rank
            while rg_idx < pf.num_row_groups:
                texts = pf.read_row_group(rg_idx, columns=["text"]).column("text").to_pylist()
                for i in range(0, len(texts), tokenizer_batch_size):
                    yield texts[i:i + tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def _fineweb_token_loader(
    B: int,
    T: int,
    files: list[str],
    device: torch.device,
    resume_state_dict: Optional[dict[str, Any]],
    tokenizer,
    tokenizer_threads: int,
    tokenizer_batch_size: int,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
):
    """
    Stream FineWeb parquet, tokenize on the fly, pack into [B, T] windows.

    Each document is bos-prepended and concatenated into a flat token stream;
    consecutive windows overlap by one token so (idx, targets) form a clean
    next-token pair. token_types come from the tokenizer's own two-band
    VocabLayout (text + trailing specials as control).
    """
    from core.tokenization.vocab_layout import VocabLayout

    if tokenizer is None:
        from modalities.text.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()

    if rank is None and world_size is None:      # the launcher's process: its share of the row groups
        identity = process_identity()
        rank, world_size = identity.rank, identity.world_size
    bos = tokenizer.get_bos_token_id()
    vocab_size = tokenizer.get_vocab_size()
    # Text-only path: the layout is just the trained tokenizer's own two bands
    # (the artifact self-describes its bundled control tail).
    # Orchestrators with more modalities assemble richer layouts themselves;
    # TextDataSource ignores these types and re-derives its own via its recipe.
    n_control = len(tokenizer.get_special_tokens())
    layout = VocabLayout()
    layout.add_range(0, 0, vocab_size - n_control)           # text
    layout.add_range(2, vocab_size - n_control, vocab_size)  # control

    batches = _document_batches(
        files, resume_state_dict, tokenizer_batch_size, rank, world_size,
    )

    needed = B * T + 1
    token_buffer: list[int] = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    while True:
        while len(token_buffer) < needed:
            text_batch, (pq_idx, rg_idx, epoch) = next(batches)
            token_lists = tokenizer.encode(
                text_batch, prepend=bos, num_threads=tokenizer_threads,
            )
            for tokens in token_lists:
                token_buffer.extend(tokens)

        seq = torch.tensor(token_buffer[:needed], dtype=torch.long)
        del token_buffer[:B * T]  # keep one-token overlap for the next window

        inputs = seq[:-1].view(B, T).to(device)
        targets = seq[1:].view(B, T).to(device)
        token_types = layout.classify_token_types(inputs)
        target_types = layout.classify_token_types(targets)

        yield {
            "idx": inputs,
            "token_types": token_types,
            "targets": targets,
            "target_types": target_types,
            "state_dict": {
                "files": [os.path.basename(f) for f in files],
                "pq_idx": pq_idx,
                "rg_idx": rg_idx,
                "epoch": epoch,
            },
        }


# ---------------------------------------------------------------------------
# Exact positions (TextDataSource)
# ---------------------------------------------------------------------------
#
# A position is a "segment" [epoch, file, row group, document, token]: that row group
# from that token of that document (counted in the document's tokens with its BOS) to
# the row group's end; [e, f, g, 0, 0] is the whole row group. Documents are tokenized
# one by one, deterministically (the same file contents and tokenizer give the same
# tokens), so a token position names the same token again on resume: the document is
# tokenized whole and cut, never its remaining characters.
#
# A rank reads its share of the run state's leftover segments, then its stride from the
# front (the "main stream"): in the front's file row groups g + r, g + r + N, ...; in
# every later file, and epoch, r, r + N, ... . From the start, front (1, 0, 0), that is
# the old rule, row groups r mod N of every file, so a fresh reading is unchanged.
#
# The state after a window speaks of the CURSOR, the first token not yet in a yielded
# window, never of how far the reader has read ahead: row groups and leftover segments
# read into the buffer but not reached by the cursor count as not started.
#   this rank's: {files, next: front, taken: main-stream row groups the cursor has
#                 entered, shard: [rank, world_size], leftover: the cursor's segment
#                 from the cursor, then the rest of its share not yet entered}
#   the run's (merge_exact_states):  {files, next, leftover}


class RowGroups:
    """The files' row-group counts, from their parquet footers, each read once when first needed."""

    def __init__(self, files):
        self.files, self._counts = list(files), {}

    def count(self, f: int) -> int:
        if f not in self._counts:
            import pyarrow.parquet as pq
            self._counts[f] = pq.ParquetFile(self.files[f]).metadata.num_row_groups
        return self._counts[f]

    def successor(self, pos):
        """The row group after (epoch, file, row group) in reading order: the next file's first after a file's last,
        the next epoch's first after the last file's last."""
        e, f, g = pos
        g += 1
        while g >= self.count(f):
            f, g = f + 1, 0
            if f == len(self.files):
                e, f = e + 1, 0
        return e, f, g

    def stride(self, front, rank: int, world_size: int):
        """The main-stream row groups rank r of N reads from the front, endlessly, as (epoch, file, row group).
        Raises if a whole pass over the files gives this rank none (fewer row groups than ranks in every file),
        where reading would otherwise loop forever."""
        e, f0, g0 = front
        while True:
            given = False
            for f in range(f0, len(self.files)):
                for g in range((g0 if f == f0 else 0) + rank, self.count(f), world_size):
                    given = True
                    yield e, f, g
            if not given and (f0, g0) == (0, 0):
                raise ValueError(f"rank {rank} of {world_size} gets no row group in a whole pass over the "
                                 f"{len(self.files)} file(s): each holds fewer row groups than there are ranks")
            e, f0, g0 = e + 1, 0, 0


START = (1, 0, 0)                 # the front of a reading from the beginning: epoch 1, file 0, row group 0


def run_state(file_names, front, leftover):
    """The run's state: the front and the leftover segments, no rank in it."""
    return {"files": list(file_names), "next": list(front), "leftover": [list(x) for x in leftover]}


def rank_state(file_names, front, taken, rank, world_size, leftover):
    """One rank's state: the front its reading started from, the main-stream row groups its cursor has entered,
    which rank of how many, and its leftover segments from the cursor on."""
    return {"files": list(file_names), "next": list(front), "taken": taken, "shard": [rank, world_size],
            "leftover": [list(x) for x in leftover]}


def merge_exact_states(states, row_groups: RowGroups):
    """The run's state from every rank's state (each after its last DELIVERED window; the ranks of one run, every
    one once): the leftover segments of all ranks, then the new front and the main-stream row groups the slower
    ranks have not entered before it. Pure: reads only the states and the files' row-group counts.

    The new front follows the furthest main-stream row group any rank's cursor has entered; consuming leftover
    segments does not move it, and when no rank has entered the main stream it stays where it was. A rank's
    main-stream row groups before the new front that its cursor has not entered are its gaps, whole row groups,
    under THIS run's stride (the restore's front, this run's world size); the ranks of a resume read under theirs.
    Every row group from the front on belongs to exactly one rank under the stride, so each before the new front was
    either entered by its rank (trained, or its rest is in that rank's leftover) or is a gap."""
    world_size = len(states)
    if sorted(tuple(s["shard"]) for s in states) != [(r, world_size) for r in range(world_size)]:
        raise ValueError(f"merge_exact_states needs the state of every rank, ranks 0..{world_size - 1} of "
                         f"{world_size} once each (got shards {[s.get('shard') for s in states]}): a missing "
                         f"rank's text would be lost")
    fronts = {tuple(s["next"]) for s in states}
    if len(fronts) != 1:
        raise ValueError(f"the ranks' states start from different fronts {sorted(fronts)}: not one run's states")
    front = fronts.pop()
    by_rank = sorted(states, key=lambda s: s["shard"][0])
    leftover = [list(x) for s in by_rank for x in s["leftover"]]
    entered = []
    for s in by_rank:
        if s["taken"]:
            for j, pos in enumerate(row_groups.stride(front, s["shard"][0], world_size)):
                if j == s["taken"] - 1:
                    entered.append(pos)
                    break
    new_front = max(front, row_groups.successor(max(entered))) if entered else front
    for s in by_rank:
        for j, pos in enumerate(row_groups.stride(front, s["shard"][0], world_size)):
            if pos >= new_front:
                break
            if j >= s["taken"]:
                leftover.append([*pos, 0, 0])
    return run_state(by_rank[0]["files"], new_front, sorted(leftover))


def exact_windows(T: int, files: list[str], tokenizer, row_groups: RowGroups, *, front, taken: int, leftover,
                  rank: int, world_size: int, tokenizer_threads: int = 4, tokenizer_batch_size: int = 128):
    """One rank's windows of T text tokens, each yielded as (tokens: list[int] of T, pieces, state after it).

    It reads `leftover` (this rank's segments, in order), then its main-stream row groups from the front, skipping
    the first `taken` (a reading resumed by the same rank; 0 after a restore). Documents are tokenized with BOS in
    batches of tokenizer_batch_size, concatenated, and cut into consecutive windows of T tokens, as soon as the
    tokens are there: the same windows as token_data_loader's rows from the same start. `pieces` says where the
    window's tokens came from, in order: [epoch, file, row group, document, first token, count] per piece. The state
    is this rank's, at the window's end (see the section's comment)."""
    import pyarrow.parquet as pq

    names = [os.path.basename(f) for f in files]
    bos = tokenizer.get_bos_token_id()
    share = [tuple(x) for x in leftover]

    def segments():
        for i, seg in enumerate(share):
            yield ("leftover", i), seg
        for j, (e, f, g) in enumerate(row_groups.stride(tuple(front), rank, world_size)):
            if j >= taken:
                yield ("stream", j), (e, f, g, 0, 0)

    def state(ref, e, f, g, doc, tok):
        """This rank's state with the cursor at (ref's segment, doc, tok); doc None = at the segment's end."""
        kind, i = ref
        rest = [] if doc is None else [[e, f, g, doc, tok]]
        if kind == "leftover":
            return rank_state(names, front, taken, rank, world_size, rest + share[i + 1:])
        return rank_state(names, front, i + 1, rank, world_size, rest)

    buf, head = [], 0             # buffered tokens; buf[head:] not yet in a window
    pieces = deque()              # [ref, e, f, g, doc, tok, n] for buf[head:], in order
    pf, pf_file = None, None
    for ref, (e, f, g, doc0, tok0) in segments():
        if pf_file != f:
            pf, pf_file = pq.ParquetFile(files[f]), f
        texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
        for d in range(doc0, len(texts), tokenizer_batch_size):
            batch = texts[d:d + tokenizer_batch_size]
            for k, toks in enumerate(tokenizer.encode(batch, prepend=bos, num_threads=tokenizer_threads)):
                start = tok0 if d + k == doc0 else 0
                if start < len(toks):
                    buf.extend(toks[start:])
                    pieces.append([ref, e, f, g, d + k, start, len(toks) - start])
            next_doc = d + len(batch)
            while len(buf) - head >= T:
                window, head, need, from_ = buf[head:head + T], head + T, T, []
                while need:
                    p = pieces[0]
                    used = min(need, p[6])
                    from_.append([p[1], p[2], p[3], p[4], p[5], used])
                    need -= used
                    if used == p[6]:
                        pieces.popleft()
                    else:
                        p[5], p[6] = p[5] + used, p[6] - used
                if pieces:                                           # the cursor is the first buffered token
                    after = state(*pieces[0][:6])
                elif next_doc < len(texts):                          # buffer empty: the next document to read
                    after = state(ref, e, f, g, next_doc, 0)
                else:                                                # this segment is done: the next one's start
                    after = state(ref, e, f, g, None, None)
                yield window, from_, after
            del buf[:head]
            head = 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def token_data_loader(
    B: int,
    T: int,
    files: Optional[list[str]] = None,
    device: str | torch.device = "cuda",
    resume_state_dict: Optional[dict[str, Any]] = None,
    data_path: Optional[str] = None,
    vocab_size: int = 65536,
    seed: int = 0,
    source: Optional[str] = None,
    tokenizer: Any = None,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    **_unused: Any,
):
    """Yield infinite next-token-prediction batches with checkpointable state.

    Source selection:
      - ``data_path`` set        → deterministic byte stream (offline reproducibility).
      - ``source == "synthetic"``→ deterministic random tokens (smoke tests).
      - otherwise (default)      → real FineWeb parquet streaming over ``files``.

    ``files`` is an EXPLICIT list of parquet paths — resolve it with
    ``modalities.text.datasets.resolve_split(data_cfg, dataset, split)``. There is
    deliberately no ``split=`` shortcut any more: inferring the validation set
    from the directory listing is the bug this refactor removes.

    ``rank`` / ``world_size``: which share of the parquet row groups this stream
    reads (row groups ``rank, rank + world_size, ...`` of each file); give both or
    neither. Both unset: the launcher's process (``core.utils.process_identity``),
    what training wants.
    A stream every process reads whole, such as a validation set each rank freezes
    in full, passes ``rank=0, world_size=1``. The synthetic paths are not sharded.

    All paths yield the SAME dict shape:
      ``{idx:[B,T], token_types:[B,T], targets:[B,T], target_types:[B,T], state_dict}``
    so the callers (TextDataSource, TextEvaluator) are source-agnostic.
    """
    if data_path is not None or source == "synthetic":
        return synthetic_token_loader(
            B, T, split="train", device=device, resume_state_dict=resume_state_dict,
            data_path=data_path, vocab_size=vocab_size, seed=seed,
        )
    if not files:
        raise ValueError(
            "token_data_loader needs an explicit `files=[...]`. Resolve it via "
            "modalities.text.datasets.resolve_split(data_cfg, dataset, split)."
        )
    return _fineweb_token_loader(
        B, T, files, torch.device(device), resume_state_dict,
        tokenizer, tokenizer_threads, tokenizer_batch_size, rank, world_size,
    )
