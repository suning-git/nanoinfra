"""
sources.py — 世界 -> token. The encode side of the boundary.

Its mirror image is decode/ (token -> 世界). Between them sits the model, which
never sees anything but integers. That symmetry is the project, and it is meant to
be visible in the directory listing.

TWO of the three sources already exist and are used unchanged:

    text    modalities.text.TextDataSource      FineWeb parquet, BPE on the fly
    motion  modalities.motion...T2MDataSource   pre-encoded (caption, codes) pairs

ONE is new, and it is the only data code this project writes:

    video   VideoRowSource                      pre-encoded fixed-stride memmap

All three satisfy core's DataSource contract — an infinite iterator of raw samples
{tokens, token_types, supervision_mask}, each at its own length, no padding (core's
data pipeline since 2026-10) — so MixedDataLoader mixes, prepares (next-token
prediction), batches and checkpoints all three identically. The orchestrator passes
SOURCE_TYPES as a LOCAL mapping, never a global registry: switching by config is the
orchestrator's concern, not core's.

A consequence worth knowing: because all three are ordinary DataSources, listing
more than one under `data.sources` in a config trains ONE model on several
modalities at once. That is a later lesson, not this one, but it costs no code.
"""

from typing import Any, Dict, Iterator, Optional

import torch

from core.data.data_source import DataSource


class VideoRowSource(DataSource):
    """Pre-encoded VizDoom clips -> one flat token row each.

    The cache is two flat binaries plus a meta.json (built offline; students are
    given it): `{split}_codes.u16` of code_len ids per row and `{split}_actions.u8`
    of one action id per game frame. Every row of a given clip length has the same
    width, so `row i` is a slice and random access costs a page fault — which is
    what lets core's ResumableDistributedSampler drive it, and what makes a run
    checkpointed on 2 GPUs resume on 1 (its position {seed, epoch, index} is global,
    with no rank in it; at each save the ranks' states are merged, with the rows
    before it no rank has trained, so nothing is skipped or repeated).

    THE ROW. Codes and actions are interleaved into one flat sequence, then wrapped
    by the recipe exactly the way text and motion are wrapped:

        [bos, video_start, L0, a a a a, L1, a a a a, ..., L4, video_end, eos]
               ^ 256 codes per latent frame        ^ td=4 action ids per frame

    There is no RowLayout, no block mask and no 3D rope: this is a plain causal
    sequence, shifted by one, identical in every respect to the text line. The
    interleaving is a property of the DATA, not of the model.

    SUPERVISION. supervision_mask is False on:
      * the action tokens — they are the control INPUT; learning to predict which
        button a player presses is a different problem, and supervising it spends
        capacity on it;
      * the codes of L0 — frame 0 is the GIVEN observation, not a prediction.
    and True on the remaining code positions plus the closing tag. That is the whole
    mechanism: the loader's next-token preparation turns unsupervised tokens into
    IGNORE_INDEX targets, and the fused cross-entropy already honours ignore_index.
    The motion line expresses "supervise the motion half only" through the identical
    field.
    """

    def __init__(self, config: Dict[str, Any], tokenizers: Dict):
        if "sequence_len" in config:
            raise KeyError("VideoRowSource: `sequence_len` is now `sample_len`, the raw tokens per row "
                           "(T + 1 for rows of T positions)")
        import json
        from pathlib import Path

        import numpy as np

        from core.data.dist_sampler import ResumableDistributedSampler
        from core.data.sequence_recipe import SequenceRecipe

        from projects.nano_multimodal import spec

        self._layout = tokenizers["layout"]
        self._resolver = tokenizers["control_resolver"]
        self.split = config.get("split", "train")
        self.sample_len = int(config["sample_len"])

        cache = Path(config["cache"])
        meta = json.loads((cache / "meta.json").read_text())
        contract = meta.get("shape_contract", meta.get("geometry"))
        n_rows = meta["rows"][self.split]
        self.contract = contract
        self.codes = np.memmap(cache / f"{self.split}_codes.u16",
                               dtype=meta["code_dtype"], mode="r",
                               shape=(n_rows, contract["code_len"]))
        self.actions = np.memmap(cache / f"{self.split}_actions.u8",
                                 dtype=meta["action_dtype"], mode="r",
                                 shape=(n_rows, contract["n_action_tokens"]))

        cpf, n_lat, td = contract["codes_per_frame"], contract["n_latent"], contract["td"]
        n_blocks = n_lat - 1
        self.cpf, self.n_lat, self.td, self.n_blocks = cpf, n_lat, td, n_blocks

        # --- the interleave, laid out ONCE ---------------------------------
        # [L0, a*td, L1, a*td, ..., L4] as one flat field. Ported slot-for-slot
        # from exemplars/nano_world_model/row_layout.py: which action ids sit
        # between which frames is a fact of how the CACHE was built, and getting
        # it wrong misaligns the conditioning without misaligning anything a test
        # would notice.
        field_len = n_lat * cpf + n_blocks * td
        code_slots, action_slots, predicted = [], [], []
        p = 0
        for k in range(n_lat):
            if k > 0:                       # frame 0 is the given observation
                action_slots.extend(range(p, p + td))
                p += td
                predicted.extend(range(p, p + cpf))
            code_slots.extend(range(p, p + cpf))
            p += cpf
        assert p == field_len
        self._code_slots = np.asarray(code_slots)      # cache column order
        self._action_slots = np.asarray(action_slots)
        # The last cache action drives a frame past the window; only n_blocks*td used.
        self._action_cols = np.arange(n_blocks * td)

        recipe = SequenceRecipe(
            template=config["recipe"]["template"],
            supervise=config["recipe"].get("supervise", "all"),
            supervise_tags=config["recipe"].get("supervise_tags"),
            constants=config["recipe"].get("constants"),
        )
        self.recipe_name = config.get("recipe_name", "<inline>")
        self.v_off = self._layout.offset(spec.VIDEO_TYPE_ID)
        self.a_off = self._layout.offset(spec.ACTION_TYPE_ID)
        self.n_actions = spec.N_ACTIONS

        fixed = recipe.build_fixed_layout(
            {"video_tokens": field_len}, self._layout, self._resolver,
            field_dummy_ids={"video_tokens": self.v_off})
        row_len = int(fixed["token_template"].numel())
        if row_len != self.sample_len:
            raise ValueError(
                f"video row is {row_len} tokens but sample_len={self.sample_len}. The line's "
                f"sequence_len (row_len - 1 positions) is DERIVED from the clip "
                f"(spec.video_shape), not configured — leave sequence_len: null in the yaml.")
        f0, f1 = fixed["field_slices"]["video_tokens"]
        self._field = (f0, f1)

        # --- supervision, written once -------------------------------------
        # 1 on the code positions of the PREDICTED latent frames, plus the closing
        # video_end tag; 0 on the given frame, on every action token, and on the
        # delimiters. The reasons: the given frame is the observation, and the
        # actions are the control INPUT — learning which button a player presses is
        # a different problem that would spend capacity on it.
        sm = np.zeros(row_len, dtype=bool)
        sm[f0 + np.asarray(predicted)] = True
        sm[f1] = True                                  # video_end
        self.n_supervised = int(sm.sum())
        assert self.n_supervised == n_blocks * cpf + 1

        # The loader prepares and batches on the CPU and moves whole batches to the
        # training device (core's data pipeline): a source yields CPU
        # tensors unless its config asks otherwise.
        self.device = torch.device(config.get("device", "cpu"))
        self._template = fixed["token_template"].numpy().copy()
        # The field holds video codes AND action ids, so the fixed layout (one dummy ID per field, video here) types
        # the action slots as video. Type a representative row instead: an action id on every action slot.
        typing_row = fixed["token_template"].clone()
        typing_row[f0 + torch.from_numpy(self._action_slots)] = self.a_off
        self._types = self._layout.classify_token_types(typing_row).to(self.device)
        self._sm = torch.from_numpy(sm).to(self.device)

        self._sampler = ResumableDistributedSampler(self.codes,
                                                    seed=int(config.get("seed", 0)))
        if "rank" in config:
            # The share this source reads, when the config says (resolve_sources does: the process's own for
            # training, rank 0 of 1 for the validation set every rank freezes whole). Otherwise the sampler's
            # default, the launcher's process.
            self._sampler.rank, self._sampler.world_size = int(config["rank"]), int(config["world_size"])
        rs = config.get("resume_state")
        if rs:
            self._sampler.load_state_dict(rs)

        print(f"VideoRowSource[{self.split}]: {n_rows} rows x {contract['code_len']} "
              f"codes, {self.n_supervised} supervised/row, recipe[{self.recipe_name}], "
              f"cache={cache.name}")

    def _row(self, i):
        """Cache row -> one flat token row of GLOBAL ids."""
        import numpy as np
        row = self._template.copy()
        f0 = self._field[0]
        row[f0 + self._code_slots] = self.v_off + np.asarray(self.codes[i], dtype=np.int64)
        a = np.clip(np.asarray(self.actions[i], dtype=np.int64)[self._action_cols],
                    0, self.n_actions - 1)
        row[f0 + self._action_slots] = self.a_off + a
        return row

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        import numpy as np
        while True:
            for i in self._sampler:
                yield {                                # every row is full: no padding, no attention_mask
                    "tokens": torch.from_numpy(self._row(i)).to(self.device),
                    "token_types": self._types,        # shared (read-only)
                    "supervision_mask": self._sm,      # shared (read-only)
                }

    def get_state(self) -> Optional[Dict[str, Any]]:
        return self._sampler.state_dict()

    def set_state(self, state: Dict[str, Any]) -> None:
        self._sampler.load_state_dict(state)

    def merge_rank_states(self, states):
        return self._sampler.merge_rank_states(states)

    def __repr__(self) -> str:
        s = self._sampler.state_dict()
        return f"video:(ep={s['epoch']}, i={s['index']})"


def source_types() -> Dict[str, type]:
    """The local string->class mapping MixedDataLoader resolves `type:` against.

    A LOCAL parameter, not a global registry: switching by config is the orchestrator's
    concern, and a project that mutates a framework-level global to get exactly this
    changes every other user of the framework without saying so.

    Imports are lazy and failures are tolerated per entry, so a text-only run never
    needs the motion package (numpy converters, and the SMPL body model on the
    decode side) and a motion run never needs the video cache reader.
    """
    types: Dict[str, type] = {}
    try:
        from modalities.text import TextDataSource
        types["text"] = TextDataSource
    except Exception:                                    # noqa: BLE001 - optional line
        pass
    try:
        from modalities.motion.data.sources import MotionDataSource, T2MDataSource
        types["t2m"] = T2MDataSource
        types["motion"] = MotionDataSource
    except Exception:                                    # noqa: BLE001 - optional line
        pass
    types["video"] = VideoRowSource
    return types
