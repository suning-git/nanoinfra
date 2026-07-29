"""
Dataset registry for the text modality: resolve ``(dataset, split)`` to a
concrete, declared file list — plus a fingerprint so "which validation set did
this run use?" is an auditable fact rather than a guess.

WHY THIS FILE EXISTS
--------------------
The predecessor derived the split from the directory listing::

    paths = sorted(os.listdir(data_dir))
    return paths[:-1] if split == "train" else paths[-1:]   # val = whichever sorts last

so the *identity of the validation set* depended on what happened to be on
disk. Downloading one more shard silently changed the ruler and quietly
invalidated every historical number — with no error and no log line. That is
the bug class this module removes: **splits are declared, never inferred.**

Config shape (see ``configs/train_text.yaml``)::

    data:
      datasets:
        fineweb:
          kind: parquet_text
          dir: /path/to/shards          # optional; defaults to <base>/base_data
          splits:
            val:   {files: [shard_005_00000.parquet]}   # pinned, a list — any size
            train: {rest: true}                         # = the complement, per dataset

Design notes:
  - ``files`` is always a LIST, so a validation set spanning many shards is the
    same declaration, just longer. Nothing here assumes val is one file.
  - ``rest: true`` makes *adding data a safe operation*: train grows, val does
    not move, historical numbers stay comparable. Declaring it the other way
    round (val = rest) would put the ruler back at the mercy of the directory.
  - A dataset with a single file and a pinned val therefore has an EMPTY train
    complement -> hard error, which is exactly right. The predecessor silently
    used one shard as both train and val.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional


def _default_dir() -> str:
    from core.utils import get_base_dir
    return os.environ.get("NANOINFRA_DATA_DIR") or os.path.join(get_base_dir(), "base_data")


def dataset_dir(ds_cfg: Dict[str, Any]) -> str:
    d = ds_cfg.get("dir") or _default_dir()
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"dataset dir does not exist: {d}\n"
            f"(download shards there, e.g. "
            f"`python exemplars/text_pretrain/data/download_shards.py 000 005`)"
        )
    return d


def _all_files(d: str) -> List[str]:
    return sorted(f for f in os.listdir(d)
                  if f.endswith(".parquet") and not f.endswith(".tmp"))


def resolve_split(data_cfg: Dict[str, Any], dataset: str, split: str) -> List[str]:
    """``(dataset, split)`` -> absolute file paths. No inference, no silent fallback."""
    datasets = data_cfg.get("datasets")
    if not datasets:
        raise KeyError(
            "data.datasets is not declared. Splits are declared, never inferred — "
            "see modalities/text/datasets.py for the config shape."
        )
    if dataset not in datasets:
        raise KeyError(f"unknown dataset {dataset!r}; declared: {sorted(datasets)}")
    ds_cfg = datasets[dataset]
    splits = ds_cfg.get("splits") or {}
    if split not in splits:
        raise KeyError(
            f"dataset {dataset!r} declares no split {split!r}; declared: {sorted(splits)}"
        )

    d = dataset_dir(ds_cfg)
    present = _all_files(d)
    spec = splits[split] or {}

    if spec.get("rest"):
        if sum(1 for s in splits.values() if (s or {}).get("rest")) > 1:
            raise ValueError(
                f"dataset {dataset!r}: more than one split declares `rest: true` — ambiguous"
            )
        claimed = {f for name, s in splits.items() if name != split
                   for f in ((s or {}).get("files") or [])}
        names = [f for f in present if f not in claimed]
        if not names:
            raise ValueError(
                f"dataset {dataset!r} split {split!r} (rest) is EMPTY: directory {d} holds "
                f"{len(present)} file(s) and all of them are claimed by other splits "
                f"({sorted(claimed)}). Training and validation must not overlap — "
                f"download more shards, or narrow the pinned split."
            )
    elif spec.get("files"):
        names = list(spec["files"])
        missing = [f for f in names if f not in present]
        if missing:
            raise FileNotFoundError(
                f"dataset {dataset!r} split {split!r} declares files that are not in {d}: "
                f"{missing}\n  present: {present}"
            )
    else:
        raise ValueError(
            f"dataset {dataset!r} split {split!r} declares neither `files:` nor `rest: true`. "
            f"There is no default — declare it."
        )

    return [os.path.join(d, f) for f in names]


def fingerprint(paths: List[str]) -> str:
    """Short stable digest of a resolved split (name + size — no 2 GB re-read).

    Printed at startup so a log self-describes which data it measured on; a
    swapped or truncated shard changes the digest.
    """
    h = hashlib.sha256()
    for p in paths:
        h.update(f"{os.path.basename(p)}:{os.path.getsize(p)}\n".encode())
    return h.hexdigest()[:12]


def describe(data_cfg: Dict[str, Any], dataset: str, split: str,
             paths: Optional[List[str]] = None) -> str:
    """One-line, log-friendly description of a resolved split."""
    paths = paths if paths is not None else resolve_split(data_cfg, dataset, split)
    names = [os.path.basename(p) for p in paths]
    shown = ", ".join(names[:3]) + (f", … (+{len(names)-3})" if len(names) > 3 else "")
    return f"{dataset}/{split}: {len(names)} file(s) [{shown}] fp={fingerprint(paths)}"
