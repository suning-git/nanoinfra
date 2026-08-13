"""
Training data sources — stream fixed-length token sequences to the (core) trainer.

Two concrete WS-B DataSources, both assembling sequences in the shared [text|control|motion]
vocab via a SequenceRecipe that is built FROM CONFIG (`config["recipe"]`) — recipes live in the
YAML, not in code.

- MotionDataSource: motion-only. Encodes feature clips -> code windows -> [bos, motion_start,
  <codes>, motion_end, eos].
- T2MDataSource: text->motion. Pairs a caption (text band) with precomputed motion codes
  (motion band) -> [bos, text_start, <text>, text_end, motion_start, <codes>, motion_end, eos], loss
  on the motion half only. End-padding is safe (causal attention + loss_weight 0 on pads).

(Merged from modality/motion_data_source.py + modality/t2m_data_source.py.)
"""

import os
import sys
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from modalities.motion.data import paths  # noqa: E402
from modalities.motion.data import dataset as md  # noqa: E402
from core.data.data_source import DataSource  # noqa: E402
from core.data.sequence_recipe import SequenceRecipe  # noqa: E402


def create_recipe(recipe_config):
    """Build a SequenceRecipe from a config dict (template/supervise[_tags]/constants).
    (Self-contained here — the modality's data leg does not depend on any project shim.)"""
    return SequenceRecipe(
        template=recipe_config["template"],
        supervise=recipe_config.get("supervise", "all"),
        supervise_tags=recipe_config.get("supervise_tags"),
        constants=recipe_config.get("constants"),
    )

MOTION_TYPE_ID = 1
TEXT_TYPE, MOTION_TYPE = 0, 1


class MotionDataSource(DataSource):
    def __init__(self, config: Dict[str, Any], tokenizers: Dict):
        self.motion_tokenizer = tokenizers["motion"]
        self._layout = tokenizers["layout"]
        self._resolver = tokenizers["control_resolver"]
        self.recipe = create_recipe(config["recipe"])

        self.sequence_len = config["sequence_len"]
        self.motion_len = self.sequence_len - self.recipe.overhead_tokens(self._resolver)
        self.split = config.get("split", "train")
        self.source = config.get("dataset", config.get("source", "lafan1"))
        self.code_stride = config.get("code_stride", self.motion_len // 2)
        self.seed = config.get("seed", 0)
        self.data_frac = float(config.get("data_frac", 1.0))   # clip-fraction (scaling-study data axis)
        self.data_seed = config.get("data_seed", 0)
        device_name = config.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self._budget_tokens = config.get("tokens")
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 1)

        motion_offset = self._layout.offset(MOTION_TYPE_ID)
        clips, _ = md.load_or_build(self.split, self.source, verbose=False)
        # Per-clip VQ codes are cached (encode once; reused across a sweep). Keyed by codebook
        # size, PLUS an optional `code_cache_tag` so different tokenizers of the SAME size (e.g. an
        # FSQ / 263 tokenizer vs the VQ baseline, both k512) don't collide — pre-encoded caches can
        # be dropped in under a tag and read here without re-encoding. No tag -> legacy name.
        k = self.motion_tokenizer.vocab_size
        tag = config.get("code_cache_tag")
        suffix = f"_k{k}" + (f"_{tag}" if tag else "")
        codes_cache = os.path.join(paths.PROCESSED_DIR, f"{self.source}_{self.split}_codes{suffix}.npz")
        if os.path.exists(codes_cache):
            clip_codes = list(np.load(codes_cache, allow_pickle=True)["codes"])
        else:
            clip_codes = [np.asarray(self.motion_tokenizer.encode(c), dtype=np.int64) for c in clips]
            np.savez(codes_cache, codes=np.array(clip_codes, dtype=object))
        # data-fraction: deterministic clip-level subsample (the scaling-study data axis).
        if self.data_frac < 1.0:
            rng = np.random.default_rng(self.data_seed)
            keep = rng.permutation(len(clip_codes))[:max(1, int(self.data_frac * len(clip_codes)))]
            clip_codes = [clip_codes[i] for i in keep]
        windows = []
        for codes in clip_codes:
            if len(codes) < self.motion_len:
                continue
            for s in range(0, len(codes) - self.motion_len + 1, self.code_stride):
                windows.append(codes[s:s + self.motion_len])
        self._global = (np.asarray(windows, dtype=np.int64) + motion_offset)
        self._budget = int(self._global.size)

        fixed = self.recipe.build_fixed_layout(
            {"motion_tokens": self.motion_len}, self._layout, self._resolver,
            content_tokenizer=self.motion_tokenizer,
            field_dummy_ids={"motion_tokens": motion_offset})
        self._token_template = fixed["token_template"].to(self.device)
        self._type_template = fixed["token_types"].to(self.device)
        self._loss_weights_template = fixed["loss_mask"].float().to(self.device)
        self._mask_template = torch.ones(self.sequence_len, dtype=torch.long, device=self.device)
        self._m0, self._m1 = fixed["field_slices"]["motion_tokens"]

        self._epoch = 0
        rs = config.get("resume_state")
        if rs:
            self._epoch = rs.get("epoch", 0)
        print(f"MotionDataSource[{self.source}/{self.split}]: {len(self._global)} windows of "
              f"{self.motion_len} codes (seq_len={self.sequence_len}), rank {self.rank}/{self.world_size}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        idx_all = np.arange(len(self._global))
        while True:
            order = np.random.default_rng(self.seed + self._epoch).permutation(idx_all)
            order = order[self.rank::self.world_size]
            for j in order:
                tokens = self._token_template.clone()
                tokens[self._m0:self._m1] = torch.from_numpy(self._global[j]).to(self.device)
                yield {"tokens": tokens, "token_types": self._type_template,
                       "attention_mask": self._mask_template,
                       "loss_weights": self._loss_weights_template}
            self._epoch += 1

    def get_state(self): return {"epoch": self._epoch}
    def set_state(self, state): self._epoch = state.get("epoch", 0)
    def __repr__(self): return f"motion:(ep={self._epoch})"

    def budget_tokens(self) -> Optional[int]:
        if self._budget_tokens is None or self._budget_tokens == "auto":
            return None
        return int(self._budget_tokens)


class T2MDataSource(DataSource):
    def __init__(self, config: Dict[str, Any], tokenizers: Dict):
        self.text_tokenizer = tokenizers["text"]
        self._layout = tokenizers["layout"]
        self._resolver = tokenizers["control_resolver"]
        self.recipe = create_recipe(config["recipe"])

        self.sequence_len = config["sequence_len"]
        self.split = config.get("split", "train")
        self.max_text = config.get("max_text_tokens", 32)
        self.seed = config.get("seed", 0)
        device_name = config.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 1)

        motion_offset = self._layout.offset(MOTION_TYPE)
        pad_id = self._resolver.resolve("eos")
        overhead = self.recipe.overhead_tokens(self._resolver)
        S = self.sequence_len

        # cache basename is configurable so t2m caches from different corpora coexist
        # (e.g. t2m_train.npz = AMASS/HumanML3D; t2m_bones_train.npz = Bones-SEED labels)
        cache = os.path.join(paths.PROCESSED_DIR,
                             config.get("cache", f"t2m_{self.split}.npz"))
        if not os.path.exists(cache):
            raise FileNotFoundError(f"{cache} missing — run tools/build_t2m.py (or "
                                    f"exemplars/nano_motion/data/encode.py first")
        d = np.load(cache, allow_pickle=True)
        codes_list, caps_list = list(d["codes"]), list(d["captions"])

        toks, lw, attn = [], [], []
        n_pairs = 0
        for codes, caps in zip(codes_list, caps_list):
            motion_ids_full = (np.asarray(codes, dtype=np.int64) + motion_offset).tolist()
            for cap in caps:
                text_ids = self.text_tokenizer.encode(cap)[:self.max_text]
                budget = S - overhead - len(text_ids)
                if budget < 1:
                    continue
                motion_ids = motion_ids_full[:budget]
                if len(motion_ids) < 1:
                    continue
                a = self.recipe.assemble({"text_tokens": text_ids, "motion_tokens": motion_ids},
                                         self._layout, self._resolver)
                t = a["tokens"].tolist()
                L = len(t)
                if L > S:
                    continue
                pad = S - L
                toks.append(t + [pad_id] * pad)
                lw.append(a["loss_mask"].float().tolist() + [0.0] * pad)
                attn.append([1] * L + [0] * pad)
                n_pairs += 1

        self._tokens = np.asarray(toks, dtype=np.int64)
        self._types = self._layout.classify_token_types(torch.from_numpy(self._tokens)).numpy()
        self._lw = np.asarray(lw, dtype=np.float32)
        self._attn = np.asarray(attn, dtype=np.int64)
        self._epoch = 0
        rs = config.get("resume_state")
        if rs:
            self._epoch = rs.get("epoch", 0)
        print(f"T2MDataSource[{self.split}]: {n_pairs} (text,motion) pairs from "
              f"{len(codes_list)} clips, seq_len={S}, rank {self.rank}/{self.world_size}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        idx = np.arange(len(self._tokens))
        while True:
            order = np.random.default_rng(self.seed + self._epoch).permutation(idx)[self.rank::self.world_size]
            for j in order:
                yield {"tokens": torch.from_numpy(self._tokens[j]).to(self.device),
                       "token_types": torch.from_numpy(self._types[j]).to(self.device),
                       "attention_mask": torch.from_numpy(self._attn[j]).to(self.device),
                       "loss_weights": torch.from_numpy(self._lw[j]).to(self.device)}
            self._epoch += 1

    def get_state(self): return {"epoch": self._epoch}
    def set_state(self, state): self._epoch = state.get("epoch", 0)
    def __repr__(self): return f"t2m:(ep={self._epoch})"
    def budget_tokens(self): return None
