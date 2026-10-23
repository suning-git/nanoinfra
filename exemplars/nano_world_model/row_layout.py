"""
row_layout.py — the row: where every token of a clip sits.

ONE class owns the layout. In the research code this knowledge was split between a
row BUILDER (which emitted the token sequence) and a row GEOMETRY (which described
where the blocks were) — two independent derivations of the same fact, kept in
agreement by a runtime `verify()` that re-read a built row and checked it against
the shape contract. That check existed because the two could drift, and it caught real
drift. Deriving both from one place removes the failure mode instead of detecting it.

THE ROW (interleaved action/frame layout, "FAFA"):

    [bos, vstart, L0 (given), a*td, L1, a*td, L2, ... , a*td, L_{n-1}, vend, eos, pad..]

L_k is one latent frame = `codes_per_frame` video tokens, and it is also one
diffusion BLOCK. The td action ids that DRIVE latent frame k sit immediately before
it, so the causal action->frame link is local rather than 700 tokens away; L0 is the
given initial observation, so no actions precede it, and the final action drives a
frame outside the window and is dropped.

The row is padded with eos to a multiple of 128 positions. Training lays two copies
of it end to end ([clean | noisy], block_diffusion.py), and flex attention works in
tiles of 128 positions: with the padding the noisy copy starts on a tile boundary, so
its blocks cut the tiles exactly where the clean copy's do. Laid back to back without
it, the noisy copy is shifted against the tile grid and more tiles are only partly
visible, which costs time and changes nothing else.

Who may see whom is not decided here. The attention rules live with the objective
(block_diffusion.py) and read per-position facts every prepared clip carries; `blk`
and `start_of` below are where those facts come from.
"""

import numpy as np
import torch


class RowLayout:
    """The token layout of one clip, derived once from the shape contract + vocab offsets.

    Args:
        contract: spec.shape_contract() dict
        video_offset / action_offset: band bases in the shared vocab
        control_ids: {"bos", "eos", "video_start", "video_end"} -> global id
    """

    ALIGN = 128   # flex attention's tile: the row is padded to a multiple of it (module docstring)

    def __init__(self, contract, video_offset, action_offset, control_ids, n_actions):
        self.contract = contract
        self.v_off = int(video_offset)
        self.a_off = int(action_offset)
        self.n_actions = int(n_actions)
        self.ctrl = {k: int(v) for k, v in control_ids.items()}

        cpf, n_lat, td = contract["codes_per_frame"], contract["n_latent"], contract["td"]
        self.cpf, self.n_lat, self.td = cpf, n_lat, td
        self.n_blocks = contract["n_blocks"]
        self.n_given_frames = contract["n_given"] // cpf
        assert self.n_given_frames * cpf == contract["n_given"], "n_given must be whole frames"

        # --- lay the row out once; everything below is an index into it ---------
        # content = [bos, vstart] + L0 + (td actions + L_k) * n_blocks + [vend, eos]
        self.content_len = 2 + cpf + self.n_blocks * (td + cpf) + 2
        self.row_len = -(-self.content_len // self.ALIGN) * self.ALIGN

        code_slots, action_slots, spans = [], [], []
        blk = np.zeros(self.row_len, dtype=np.int64)
        start_of = np.zeros(self.row_len, dtype=np.int64)
        p = 2                                        # after [bos, vstart]
        for k in range(n_lat):
            if k >= self.n_given_frames:
                action_slots.extend(range(p, p + td))
                p += td
                blk[p:p + cpf] = k
                start_of[p:p + cpf] = p
                spans.append((p, p + cpf))
            code_slots.extend(range(p, p + cpf))
            p += cpf
        self.end_pos = p                             # where vend sits
        assert p + 2 == self.content_len

        self.code_slots = np.asarray(code_slots)     # [n_lat * cpf], cache column order
        self.action_slots = np.asarray(action_slots)  # [n_blocks * td]
        self.spans = spans                            # per predicted block: (start, end)
        self.blk = torch.from_numpy(blk)              # block index per position, 0 = not in a predicted block
        self.start_of = torch.from_numpy(start_of)    # where that block starts (0 outside one)

        # The constant scaffolding: control tokens in place, eos everywhere else.
        template = np.full(self.row_len, self.ctrl["eos"], dtype=np.int64)
        template[0] = self.ctrl["bos"]
        template[1] = self.ctrl["video_start"]
        template[self.end_pos] = self.ctrl["video_end"]
        self._template = template

        # Which cache actions land in the row: the td driving each predicted frame.
        # (frames-1 actions exist; the last drives a frame past the window.)
        self.action_cols = np.arange(self.n_blocks * td)
        assert len(self.action_cols) == len(self.action_slots)

    # --- row assembly --------------------------------------------------------
    def assemble(self, codes, actions):
        """(codes [B, code_len], actions [B, n_action_tokens]) -> idx [B, row_len] int64.

        Pure fancy-indexing on the precomputed slot arrays — no per-row Python."""
        codes = np.asarray(codes)
        actions = np.asarray(actions)
        assert codes.shape[1] == self.contract["code_len"], "cache row width != contract"
        B = codes.shape[0]
        idx = np.broadcast_to(self._template, (B, self.row_len)).copy()
        idx[:, self.code_slots] = self.v_off + codes.astype(np.int64)
        a = np.clip(actions[:, self.action_cols].astype(np.int64), 0, self.n_actions - 1)
        idx[:, self.action_slots] = self.a_off + a
        return torch.from_numpy(idx)

    def __repr__(self):
        return (f"RowLayout(row_len={self.row_len}, content={self.content_len}, "
                f"blocks={self.n_blocks}x{self.cpf}, td={self.td})")
