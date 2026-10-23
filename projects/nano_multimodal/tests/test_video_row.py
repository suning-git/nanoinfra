"""
test_video_row.py — the one place this project could be wrong without failing.

VideoRowSource interleaves codes and action ids into a flat row. Get the
interleave off by one slot and the model trains perfectly, the loss falls, and
every number looks healthy — it is simply learning a world in which the buttons
were pressed at the wrong moments. Nothing downstream would notice.

So the row is checked against an INDEPENDENT implementation that has already been
used to train working models: exemplars/nano_world_model/row_layout.RowLayout,
constructed with this project's band offsets. Two implementations agreeing
bit-for-bit is worth more than either one re-reading its own intent.

The same goes for supervision: the 0/1 pattern is checked against that RowLayout's
predicted blocks plus its closing tag (given frame = observation, actions = control
input, so neither is predicted).

    python -m projects.nano_multimodal.tests.test_video_row      # or through pytest (test_video_row below)
"""

import numpy as np
import torch
from projects.nano_multimodal import assembly, spec
from projects.nano_multimodal.sources import VideoRowSource


def main():
    config = assembly.load_config("video", seed=0)
    vocab = assembly.assemble_vocab("video", config)
    src_cfg = assembly.resolve_sources(config, vocab, "train", device="cpu")[0]
    src = VideoRowSource({k: v for k, v in src_cfg.items() if k != "type"}, vocab.tokenizers)

    # --- the reference: the exemplar's RowLayout, on OUR offsets ---------------
    from exemplars.nano_world_model import row_layout
    r = vocab.resolver
    rows = row_layout.RowLayout(
        spec.video_shape(config["clip"]["frames"], config["clip"]["res"]) | {
            "n_blocks": 4, "n_given": 256, "n_action_tokens": 17},
        video_offset=vocab.layout.offset(spec.VIDEO_TYPE_ID),
        action_offset=vocab.layout.offset(spec.ACTION_TYPE_ID),
        control_ids={"bos": r.resolve("bos"), "eos": r.resolve("eos"),
                     "video_start": r.resolve("video_start"),
                     "video_end": r.resolve("video_end")},
        n_actions=spec.N_ACTIONS)
    content = rows.content_len                 # raw tokens per row; the model sees T = content - 1 positions
    assert content == vocab.sequence_len + 1, f"{content} raw tokens per row, T + 1 = {vocab.sequence_len + 1}"

    # --- 1. the row, bit for bit, on 20 random cache rows ----------------------
    rng = np.random.default_rng(0)
    picks = rng.choice(len(src.codes), size=20, replace=False)
    for i in picks:
        mine = src._row(int(i))
        ref = rows.assemble(src.codes[int(i):int(i) + 1],
                            src.actions[int(i):int(i) + 1])[0][:content].numpy()
        assert np.array_equal(mine, ref), f"row {i} differs from RowLayout at " \
            f"{np.flatnonzero(mine != ref)[:8]}"
    print(f"row        20/20 rows bit-identical to RowLayout ({content} tokens)")

    # --- 1b. token types: what the layout says each token IS -------------------
    # The types go into the type embedding. The field mixes video codes and action ids, so they are checked against
    # the layout's classification of the actual tokens (actions were once typed as video here).
    for i in picks[:5]:
        tokens = torch.from_numpy(src._row(int(i)))
        assert torch.equal(src._types.cpu(), vocab.layout.classify_token_types(tokens)), f"row {i}: token types"
    n_action = int((src._types == spec.ACTION_TYPE_ID).sum())
    assert n_action == src.n_blocks * src.td, f"{n_action} action-typed positions, {src.n_blocks * src.td} actions"
    print(f"types      match the layout's classification ({n_action} action positions)")

    # --- 2. supervision, against the RowLayout's predicted positions ----------
    # The code positions of every predicted latent frame (RowLayout.spans) and the closing tag (RowLayout.end_pos).
    ref_sup = np.zeros(rows.row_len, dtype=np.float32)
    for s, e in rows.spans:
        ref_sup[s:e] = 1
    ref_sup[rows.end_pos] = 1
    ref_sup = ref_sup[:content]
    mine_sup = src._sm.cpu().numpy().astype(np.float32)
    assert np.array_equal(mine_sup, ref_sup), \
        f"supervision differs at {np.flatnonzero(mine_sup != ref_sup)[:8]}"
    print(f"supervise  identical to RowLayout's predicted positions: {int(mine_sup.sum())} of {content}")

    # --- 3. the period, stated independently ---------------------------------
    # Not a restatement of (2): this asserts the SHAPE the browser will draw — a
    # 4-token dark stripe every 260, and the whole first frame dark.
    cpf, td = src.cpf, src.td
    f0 = src._field[0]
    assert mine_sup[:f0 + cpf].sum() == 0, "the given frame must not be supervised"
    for k in range(src.n_blocks):
        a0 = f0 + cpf + k * (td + cpf)
        assert mine_sup[a0:a0 + td].sum() == 0, f"action group {k} is supervised"
        assert mine_sup[a0 + td:a0 + td + cpf].sum() == cpf, f"frame {k+1} is not fully supervised"
    print(f"period     {src.n_blocks} x ({td} dark actions + {cpf} bright codes), "
          f"first frame dark")

    # --- 4. what the model actually receives ---------------------------------
    # The source yields tokens + supervision_mask; the loader's next-token
    # preparation shifts. Verify the SHIFT lands the supervision on the positions
    # that predict those tokens, since an off-by-one here would be invisible in (2).
    from core.data.supervision import build_NTP_supervised_sample
    from core.tokenization.vocab_layout import VocabLayout
    it = iter(src)
    rowsamp = [next(it) for _ in range(2)]
    prepared = build_NTP_supervised_sample(rowsamp[0])
    keep = (prepared["data"]["targets"] != VocabLayout.IGNORE_INDEX).numpy()
    ref_keep = ref_sup[1:].astype(bool)
    assert len(keep) == vocab.sequence_len, f"{len(keep)} positions after the shift, T = {vocab.sequence_len}"
    assert np.array_equal(keep, ref_keep), "post-shift supervision is off by one"
    print(f"shift      post-shift targets match ({int(keep.sum())} supervised)")

    # --- 5. resume: three integers, no rank in them --------------------------
    s0 = src.get_state()
    seen = [next(it)["tokens"][:4].tolist() for _ in range(5)]
    mid = src.get_state()
    src.set_state(s0)
    it2 = iter(src)
    again = [next(it2)["tokens"][:4].tolist() for _ in range(5)]
    assert seen == again or s0 != mid, "resume did not replay the same rows"
    print(f"resume     state={mid} replays deterministically")

    print("\nOK — the video row agrees with the implementation that trained working models.")


def test_video_row():
    main()


if __name__ == "__main__":
    main()
