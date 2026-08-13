"""
row_layout.py — the row: where every token of a clip sits, and who may see whom.

ONE class owns the layout. In the research code this knowledge was split between a
row BUILDER (which emitted the token sequence) and a row GEOMETRY (which described
where the blocks were) — two independent derivations of the same fact, kept in
agreement by a runtime `verify()` that re-read a built row and checked it against
the geometry. That check existed because the two could drift, and it caught real
drift. Deriving both from one place removes the failure mode instead of detecting it.

THE ROW (interleaved action/frame layout, "FAFA"):

    [bos, vstart, L0 (given), a*td, L1, a*td, L2, ... , a*td, L_{n-1}, vend, eos, pad..]

L_k is one latent frame = `codes_per_frame` video tokens, and it is also one
diffusion BLOCK. The td action ids that DRIVE latent frame k sit immediately before
it, so the causal action->frame link is local rather than 700 tokens away; L0 is the
given initial observation, so no actions precede it, and the final action drives a
frame outside the window and is dropped.

TRAINING SEQUENCE (BD3-LM's vectorized two-stream trick):

    [ clean row | noisy row ]        length 2n

so one forward pass gives every block both a clean prefix and its own noised
content. The attention mask (`train_block_mask`) grants:
    clean->clean   causal — an ordinary prefix-providing pass
    noisy_k->clean strictly before block k's start (its own clean block EXCLUDED,
                   which would otherwise hand the model its own labels)
    noisy_k->noisy its own block, bidirectionally — mask tokens co-denoise
    noisy_cond     the duplicated conditioning positions, causal; outputs unused
The mask depends only on geometry, so ONE mask serves every row and every noise level.

Rope positions are MIRRORED (0..n-1 twice) so the second stream sits in the same
positional range the model was trained on — a table swap, no core changes.
"""

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_FLEX_PATCHED = False


def use_compiled_flex_attention():
    """Point core's flex_attention symbol at the COMPILED kernel. Call once, early.

    Not an import side effect on purpose (core has no magic globals) — but
    also not optional: eager flex_attention silently runs a debug path with WRONG
    outputs (causal propagation dead; torch itself says "SOLUTION: Use
    torch.compile(flex_attention)"). `RowLayout.train_block_mask` asserts this ran,
    so forgetting it is an error rather than a quietly untrained model.
    """
    global _FLEX_PATCHED
    if not _FLEX_PATCHED:
        import core.model.gpt as gpt_module
        gpt_module.flex_attention = torch.compile(flex_attention, dynamic=False)
        _FLEX_PATCHED = True


class RowLayout:
    """The token layout of one clip, derived once from geometry + vocab offsets.

    Args:
        geom: spec.clip_geometry() dict
        video_offset / action_offset: band bases in the shared vocab
        control_ids: {"bos", "eos", "video_start", "video_end"} -> global id
    """

    ALIGN = 128   # flex_attention requires Q_LEN % 128 == 0

    def __init__(self, geom, video_offset, action_offset, control_ids, n_actions):
        self.geom = geom
        self.v_off = int(video_offset)
        self.a_off = int(action_offset)
        self.n_actions = int(n_actions)
        self.ctrl = {k: int(v) for k, v in control_ids.items()}

        cpf, n_lat, td = geom["codes_per_frame"], geom["n_latent"], geom["td"]
        self.cpf, self.n_lat, self.td = cpf, n_lat, td
        self.n_blocks = geom["n_blocks"]
        self.n_given_frames = geom["n_given"] // cpf
        assert self.n_given_frames * cpf == geom["n_given"], "n_given must be whole frames"

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
        self.blk = torch.from_numpy(blk)              # block index per position, 0 = conditioning
        self.start_of = torch.from_numpy(start_of)

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
        assert codes.shape[1] == self.geom["code_len"], "cache row width != geometry"
        B = codes.shape[0]
        idx = np.broadcast_to(self._template, (B, self.row_len)).copy()
        idx[:, self.code_slots] = self.v_off + codes.astype(np.int64)
        a = np.clip(actions[:, self.action_cols].astype(np.int64), 0, self.n_actions - 1)
        idx[:, self.action_slots] = self.a_off + a
        return torch.from_numpy(idx)

    # --- attention ------------------------------------------------------------
    def train_block_mask(self, device, batch=None):
        """BlockMask over [clean | noisy] (length 2n). True = may attend."""
        assert _FLEX_PATCHED, (
            "call row_layout.use_compiled_flex_attention() before building a mask — "
            "eager flex_attention returns silently wrong values")
        n = self.row_len
        blk = self.blk.to(device)
        start = self.start_of.to(device)

        def mask_mod(b, h, q, kv):
            qs, qm = q // n, q % n
            ks, km = kv // n, kv % n
            qb = blk[qm]
            clean_causal = (qs == 0) & (ks == 0) & (km <= qm)
            noisy_prefix = (qs == 1) & (qb > 0) & (ks == 0) & (km < start[qm])
            noisy_self = (qs == 1) & (qb > 0) & (ks == 1) & (blk[km] == qb)
            noisy_cond = (qs == 1) & (qb == 0) & (ks == 0) & (km <= qm)
            return clean_causal | noisy_prefix | noisy_self | noisy_cond

        return create_block_mask(mask_mod, B=batch, H=None,
                                 Q_LEN=2 * n, KV_LEN=2 * n, device=device)

    def sample_block_mask(self, k, seq_len, device):
        """Mask for SAMPLING block k on a single-stream sequence: positions inside
        block k see the clean prefix plus the whole block bidirectionally;
        everything else is causal. seq_len must be a multiple of 128."""
        assert _FLEX_PATCHED, "call use_compiled_flex_attention() first"
        s, e = self.spans[k - 1]
        s_t = torch.tensor(s, device=device)
        e_t = torch.tensor(e, device=device)

        def mask_mod(b, h, q, kv):
            in_q = (q >= s_t) & (q < e_t)
            in_kv = (kv >= s_t) & (kv < e_t)
            return (in_q & ((kv < s_t) | in_kv)) | ((~in_q) & (kv <= q))

        return create_block_mask(mask_mod, B=None, H=None,
                                 Q_LEN=seq_len, KV_LEN=seq_len, device=device)

    def install_mirror_rope(self, trunk):
        """Swap the trunk's rope tables so a 2n sequence gets positions
        [0..n-1, 0..n-1] — stream B mirrors stream A, keeping every relative offset
        inside the range the model was trained on. Returns the originals."""
        orig = (trunk.cos, trunk.sin)
        pos = torch.arange(2 * self.row_len, device=trunk.cos.device) % self.row_len
        trunk.cos = orig[0][:, pos].contiguous()
        trunk.sin = orig[1][:, pos].contiguous()
        return orig

    def __repr__(self):
        return (f"RowLayout(row_len={self.row_len}, content={self.content_len}, "
                f"blocks={self.n_blocks}x{self.cpf}, td={self.td})")
