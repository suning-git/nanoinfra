"""
Synthetic / byte-stream token loader — modality-free debug mechanism.

Yields infinite next-token-prediction batches from deterministic random tokens
(smoke tests) or from a file's bytes (tiny offline reproducibility checks).
No tokenizer, no dataset, no modality knowledge. The production text loader
(FineWeb streaming) lives with the text modality and composes this one for
its offline fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _resolve_device(device: str | torch.device) -> torch.device:
    device_name = str(device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _load_token_stream(data_path: Optional[str], vocab_size: int) -> Optional[torch.Tensor]:
    """The whole file becomes an in-memory token stream (8 bytes/token as int64) —
    size the file accordingly (a debug mechanism, not a dataset loader)."""
    if data_path is None:
        return None

    payload = Path(data_path).read_bytes()
    if not payload:
        raise ValueError(f"data_path is empty: {data_path}")

    data = np.frombuffer(payload, dtype=np.uint8)
    if vocab_size < 256:
        data = data % np.uint8(vocab_size)   # byte values 0..255 fit any vocab >= 256
    return torch.from_numpy(data.astype(np.int64))


def synthetic_token_loader(
    B: int,
    T: int,
    split: str = "train",
    device: str | torch.device = "cuda",
    resume_state_dict: Optional[dict[str, Any]] = None,
    data_path: Optional[str] = None,
    vocab_size: int = 65536,
    seed: int = 0,
):
    """Yield infinite next-token-prediction batches from random/byte tokens.

    Batch shape matches the production loaders:
      ``{idx:[B,T], token_types:[B,T], targets:[B,T], target_types:[B,T], state_dict}``
    (token_types all 0 — there is no layout here to classify against).
    """
    resolved_device = _resolve_device(device)
    stream = _load_token_stream(data_path, vocab_size)

    state = dict(resume_state_dict or {})
    step = int(state.get("step", 0))
    offset = int(state.get("offset", 0))

    generator = torch.Generator(device="cpu")
    if "rng_state" in state:
        rng_state = state["rng_state"]
        if not isinstance(rng_state, torch.Tensor):   # restored from JSON metadata
            rng_state = torch.tensor(rng_state, dtype=torch.uint8)
        generator.set_state(rng_state.cpu())
    else:
        split_offset = 0 if split == "train" else 10_000
        generator.manual_seed(int(seed) + split_offset)

    while True:
        n = B * T + 1
        if stream is None:
            seq = torch.randint(0, vocab_size, (n,), generator=generator, dtype=torch.long)
        else:
            if offset + n <= stream.numel():
                seq = stream[offset:offset + n]
                offset = (offset + B * T) % stream.numel()
            else:
                parts = []
                remaining = n
                while remaining > 0:
                    take = min(remaining, stream.numel() - offset)
                    parts.append(stream[offset:offset + take])
                    remaining -= take
                    offset = (offset + take) % stream.numel()
                seq = torch.cat(parts)

        inputs = seq[:-1].view(B, T).to(resolved_device)
        targets = seq[1:].view(B, T).to(resolved_device)
        token_types = torch.zeros_like(inputs)
        step += 1

        yield {
            "idx": inputs,
            "token_types": token_types,
            "targets": targets,
            "target_types": token_types,
            "state_dict": {
                "split": split,
                "step": step,
                "offset": offset,
                # list, not tensor: checkpoint metadata is JSON (meta.json)
                "rng_state": generator.get_state().tolist(),
            },
        }
