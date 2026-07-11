"""
FineWeb streaming loader — text's production data path.

Streams FineWeb **parquet** shards from disk by ``split``, tokenizes documents
on the fly (BPE, bos-prepended, multi-threaded), packs the token stream into
``[B, T]`` windows. Used by ``TextDataSource`` and ``TextEvaluator``. The
``token_data_loader`` entry point also dispatches to core's modality-free
synthetic/byte-stream loader (``source="synthetic"`` / ``data_path=...``) so
the callers stay source-agnostic for smoke tests.

Resume state for the parquet path is ``{pq_idx, rg_idx, epoch}`` (parquet index +
row-group index).

Parquet layout on disk (nanochat convention): ``get_base_dir()/base_data/*.parquet``,
sorted; all-but-last shard = ``train``, last shard = ``val``. Override the directory
with env ``NANOINFRA_DATA_DIR`` or the ``data_dir`` argument.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from core.data.synthetic_loader import _resolve_device, synthetic_token_loader


# ---------------------------------------------------------------------------
# Real on-the-fly FineWeb parquet source
# ---------------------------------------------------------------------------

def _get_dist_info() -> tuple[int, int]:
    """(rank, world_size) from torch.distributed, or (0, 1) if not initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


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


def _split_paths(split: str, data_dir: Optional[str]) -> list[str]:
    """All-but-last shard = train, last shard = val (nanochat convention)."""
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    paths = list_parquet_files(data_dir)
    if len(paths) == 1:
        # Single shard available: use it for both (small-scale dev convenience).
        return paths
    return paths[:-1] if split == "train" else paths[-1:]


def _document_batches(split, data_dir, resume_state_dict, tokenizer_batch_size, rank, world_size):
    """
    Infinite iterator over document batches (lists of text strings) from parquet.

    Handles DDP sharding (row-groups strided by world_size, offset by rank) and
    approximate resume. Yields ``(text_batch, (pq_idx, rg_idx, epoch))``.
    """
    import pyarrow.parquet as pq

    parquet_paths = _split_paths(split, data_dir)

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
    split: str,
    resolved_device: torch.device,
    resume_state_dict: Optional[dict[str, Any]],
    data_dir: Optional[str],
    tokenizer,
    tokenizer_threads: int,
    tokenizer_batch_size: int,
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

    rank, world_size = _get_dist_info()
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
        split, data_dir, resume_state_dict, tokenizer_batch_size, rank, world_size,
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

        inputs = seq[:-1].view(B, T).to(resolved_device)
        targets = seq[1:].view(B, T).to(resolved_device)
        token_types = layout.classify_token_types(inputs)
        target_types = layout.classify_token_types(targets)

        yield {
            "idx": inputs,
            "token_types": token_types,
            "targets": targets,
            "target_types": target_types,
            "state_dict": {
                "split": split,
                "pq_idx": pq_idx,
                "rg_idx": rg_idx,
                "epoch": epoch,
            },
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def token_data_loader(
    B: int,
    T: int,
    split: str = "train",
    device: str | torch.device = "cuda",
    resume_state_dict: Optional[dict[str, Any]] = None,
    data_path: Optional[str] = None,
    vocab_size: int = 65536,
    seed: int = 0,
    source: Optional[str] = None,
    data_dir: Optional[str] = None,
    tokenizer: Any = None,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    **_unused: Any,
):
    """Yield infinite next-token-prediction batches with checkpointable state.

    Source selection:
      - ``data_path`` set        → deterministic byte stream (offline reproducibility).
      - ``source == "synthetic"``→ deterministic random tokens (smoke tests).
      - otherwise (default)      → real on-the-fly FineWeb parquet streaming.

    All paths yield the SAME dict shape:
      ``{idx:[B,T], token_types:[B,T], targets:[B,T], target_types:[B,T], state_dict}``
    so the callers (TextDataSource, TextEvaluator) are source-agnostic.
    Only the *content* and the ``state_dict`` keys differ.
    """
    if data_path is not None or source == "synthetic":
        return synthetic_token_loader(
            B, T, split=split, device=device, resume_state_dict=resume_state_dict,
            data_path=data_path, vocab_size=vocab_size, seed=seed,
        )
    return _fineweb_token_loader(
        B, T, split, _resolve_device(device), resume_state_dict, data_dir,
        tokenizer, tokenizer_threads, tokenizer_batch_size,
    )
