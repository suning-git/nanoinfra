"""
Training data sources — stream fixed-length token sequences to the (core) trainer.

Two concrete WS-B DataSources, both assembling sequences in the shared [text|control|motion]
vocab via a SequenceRecipe that is built FROM CONFIG (`config["recipe"]`) — recipes live in the
YAML, not in code. Lengths count raw tokens, before the next-token shift: `sample_len` (the motion
window, every sample exactly this long) and `max_sample_len` (the T2M upper bound); an orchestrator
batching rows of T positions gives T + 1. Samples are CPU tensors by default (`device: cpu`): the
loader collates on the CPU and moves whole batches.

- MotionDataSource: motion-only. Encodes feature clips -> code windows -> [bos, motion_start,
  <codes>, motion_end, eos].
- T2MDataSource: text->motion. Pairs a caption (text band) with precomputed motion codes
  (motion band) -> [bos, text_start, <text>, text_end, motion_start, <codes>, motion_end, eos], loss
  on the motion half only. Each pair is yielded at its true length; batching pads or packs it.

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
from core.data.dist_sampler import ResumableDistributedSampler  # noqa: E402
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


class _Count:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def _index_sampler(n, seed, rank, world_size):
    """Both sources draw their windows/pairs through core's ResumableDistributedSampler (since 2026-10):
    state = (seed, epoch, index), a GLOBAL position in the endless sequence of epochs, plus the leftover items
    before it no rank has trained, re-sharded under the current rank / world_size on restore — so a run saved on
    two GPUs resumes on one, nothing skipped or repeated."""
    s = ResumableDistributedSampler(_Count(n), seed=seed)
    s.rank, s.world_size = int(rank), int(world_size)
    return s


def _restore(sampler, state, n, count_key):
    """Restore a sampler from a source state. The state written before 2026-10 was only {"epoch": e}; its meaning was
    "replay epoch e from its beginning", and that is what it still means here (no in-epoch position is invented)."""
    if state is None:
        return
    if "index" not in state:
        state = {"seed": sampler.seed, "epoch": int(state.get("epoch", 0)), "index": 0}
    elif state.get(count_key) is not None and int(state[count_key]) != n:
        raise ValueError(f"state was recorded over {state[count_key]} items, this source has {n}: the items were "
                         f"counted under another length. To resume a run saved before 2026-10, give this source "
                         f"that run's sequence_len as its length (sample_len / max_sample_len)")
    sampler.load_state_dict({k: v for k, v in state.items() if k != count_key})


def _merge(sampler, states, n, count_key):
    """The run's state from every rank's: the sampler merges, the item count rides along."""
    return {**sampler.merge_rank_states([{k: v for k, v in s.items() if k != count_key} for s in states]),
            count_key: n}


class MotionDataSource(DataSource):
    def __init__(self, config: Dict[str, Any], tokenizers: Dict):
        self.motion_tokenizer = tokenizers["motion"]
        self._layout = tokenizers["layout"]
        self._resolver = tokenizers["control_resolver"]
        self.recipe = create_recipe(config["recipe"])

        if "sequence_len" in config:
            raise KeyError("MotionDataSource: `sequence_len` is now `sample_len`, the raw tokens per window "
                           "(T + 1 for rows of T positions)")
        self.sample_len = config["sample_len"]
        self.motion_len = self.sample_len - self.recipe.overhead_tokens(self._resolver)
        self.split = config.get("split", "train")
        self.source = config.get("dataset", config.get("source", "lafan1"))
        self.code_stride = config.get("code_stride", self.motion_len // 2)
        self.seed = config.get("seed", 0)
        self.data_frac = float(config.get("data_frac", 1.0))   # clip-fraction (scaling-study data axis)
        self.data_seed = config.get("data_seed", 0)
        self.device = torch.device(config.get("device", "cpu"))
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
        self._supervision_template = fixed["supervision_mask"].to(self.device)
        self._m0, self._m1 = fixed["field_slices"]["motion_tokens"]

        self.sampler = _index_sampler(len(self._global), self.seed, self.rank, self.world_size)
        _restore(self.sampler, config.get("resume_state"), len(self._global), "n_windows")
        print(f"MotionDataSource[{self.source}/{self.split}]: {len(self._global)} windows of "
              f"{self.motion_len} codes (sample_len={self.sample_len}), rank {self.rank}/{self.world_size}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for j in self.sampler:
            tokens = self._token_template.clone()
            tokens[self._m0:self._m1] = torch.from_numpy(self._global[j]).to(self.device)
            yield {"tokens": tokens, "token_types": self._type_template,     # every window is full:
                   "supervision_mask": self._supervision_template,           # no padding, no attention_mask
                   "meta": {"window": int(j)}}

    def get_state(self): return {**self.sampler.state_dict(), "n_windows": len(self._global)}
    def set_state(self, state): _restore(self.sampler, state, len(self._global), "n_windows")
    def merge_rank_states(self, states): return _merge(self.sampler, states, len(self._global), "n_windows")
    def __repr__(self): return f"motion:(ep={self.sampler.epoch}, index={self.sampler.index}/{len(self._global)})"

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

        if "sequence_len" in config:
            raise KeyError("T2MDataSource: `sequence_len` is now `max_sample_len`, the raw tokens a pair may "
                           "take (T + 1 for rows of T positions)")
        self.max_sample_len = config["max_sample_len"]
        self.split = config.get("split", "train")
        self.max_text = config.get("max_text_tokens", 32)
        self.seed = config.get("seed", 0)
        self.device = torch.device(config.get("device", "cpu"))
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 1)

        motion_offset = self._layout.offset(MOTION_TYPE)
        pad_id = self._resolver.resolve("eos")
        overhead = self.recipe.overhead_tokens(self._resolver)
        S = self.max_sample_len

        # cache basename is configurable so t2m caches from different corpora coexist
        # (e.g. t2m_train.npz = AMASS/HumanML3D; t2m_bones_train.npz = Bones-SEED labels)
        cache = os.path.join(paths.PROCESSED_DIR,
                             config.get("cache", f"t2m_{self.split}.npz"))
        if not os.path.exists(cache):
            raise FileNotFoundError(f"{cache} missing — run tools/build_t2m.py (or "
                                    f"exemplars/nano_motion/data/encode.py first")
        d = np.load(cache, allow_pickle=True)
        codes_list, caps_list = list(d["codes"]), list(d["captions"])
        # `limit` caps how many CLIPS are read. Building every pair means BPE-encoding
        # every caption up front — 37 s for the 450k-pair Bones cache — which is right
        # for training and wrong for a browser or a test that needs a handful of real
        # rows. Absent, nothing changes.
        #
        # WHICH clips, and why there are two answers. By default it is the first N —
        # cheapest, and for a browser showing "some real rows" any N will do. But the
        # cache is NOT shuffled, so a prefix is a biased sample: the first eighth of
        # the Bones cache averages 60.5 codes per clip against 54.6 for a random
        # eighth. That bias is harmless for a preview and fatal for a DATA-SCALING
        # study, where the whole claim is that the arms differ only in how much data
        # they hold. So `limit_seed` switches to a deterministic random subsample.
        # The picks are re-sorted into on-disk order afterwards: which clips are drawn
        # is the experiment, the order they are read in is not.
        limit = config.get("limit")
        if limit:
            n = int(limit)
            if config.get("limit_seed") is None:
                sel = range(min(n, len(codes_list)))
            else:
                rng = np.random.default_rng(int(config["limit_seed"]))
                sel = sorted(rng.permutation(len(codes_list))[:n].tolist())
            codes_list = [codes_list[i] for i in sel]
            caps_list = [caps_list[i] for i in sel]

        toks, sm, lens = [], [], []
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
                toks.append(t + [pad_id] * pad)          # stored padded to S; yielded at the true length L
                sm.append(a["supervision_mask"].tolist() + [False] * pad)
                lens.append(L)
                n_pairs += 1

        self._tokens = np.asarray(toks, dtype=np.int64)
        self._types = self._layout.classify_token_types(torch.from_numpy(self._tokens)).numpy()
        self._sm = np.asarray(sm, dtype=np.bool_)
        self._len = np.asarray(lens, dtype=np.int64)
        self.sampler = _index_sampler(len(self._tokens), self.seed, self.rank, self.world_size)
        _restore(self.sampler, config.get("resume_state"), len(self._tokens), "n_pairs")
        print(f"T2MDataSource[{self.split}]: {n_pairs} (text,motion) pairs from "
              f"{len(codes_list)} clips, max_sample_len={S}, rank {self.rank}/{self.world_size}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for j in self.sampler:
            L = int(self._len[j])                        # the pair's true length: no mechanical padding
            yield {"tokens": torch.from_numpy(self._tokens[j, :L]).to(self.device),
                   "token_types": torch.from_numpy(self._types[j, :L]).to(self.device),
                   "supervision_mask": torch.from_numpy(self._sm[j, :L]).to(self.device),
                   "meta": {"pair": int(j)}}

    def get_state(self): return {**self.sampler.state_dict(), "n_pairs": len(self._tokens)}
    def set_state(self, state): _restore(self.sampler, state, len(self._tokens), "n_pairs")
    def merge_rank_states(self, states): return _merge(self.sampler, states, len(self._tokens), "n_pairs")
    def __repr__(self): return f"t2m:(ep={self.sampler.epoch}, index={self.sampler.index}/{len(self._tokens)})"
    def budget_tokens(self): return None
