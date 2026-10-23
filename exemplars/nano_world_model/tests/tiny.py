"""A tiny world-model assembly for the CPU tests: control + video + action bands (no text tokenizer needed), a 9-frame
16 px clip (4 codes per latent frame, 2 predicted blocks), the row's pad values, and a small real GPT."""

import torch

import modalities.control
from modalities.assembler import Modality, build_layout
from modalities.control import make_control_resolver

from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead

from exemplars.nano_world_model import spec
from exemplars.nano_world_model.block_diffusion import pad_values
from exemplars.nano_world_model.row_layout import RowLayout

N_VIDEO = 24


def tiny_assembly():
    """-> (layout, resolver, rows, pads, mask_id)."""
    control = modalities.control.manifest()
    layout = build_layout([control, Modality(name="video", type_id=spec.VIDEO_TYPE_ID, vocab_size=N_VIDEO,
                                             tokenizer=None),
                           Modality(name="action", type_id=spec.ACTION_TYPE_ID, vocab_size=spec.N_ACTIONS,
                                    tokenizer=None)])
    resolver = make_control_resolver(control, layout)
    rows = RowLayout(spec.shape_contract(frames=9, res=16), video_offset=layout.offset(spec.VIDEO_TYPE_ID),
                     action_offset=layout.offset(spec.ACTION_TYPE_ID),
                     control_ids={"bos": resolver.resolve("bos"), "eos": resolver.resolve("eos"),
                                  "video_start": resolver.resolve(spec.VIDEO_START),
                                  "video_end": resolver.resolve(spec.VIDEO_END)},
                     n_actions=spec.N_ACTIONS)
    eos = resolver.resolve("eos")
    pads = pad_values(eos, int(layout.classify_token_types(torch.tensor([eos]))[0]))
    return layout, resolver, rows, pads, resolver.resolve(spec.MASK_SLOT)


def tiny_gpt(layout, rows, seed=0):
    """A 2-layer GPT and head with every weight random (init_weights zeroes the output projections, under which
    attention would not reach the output)."""
    torch.manual_seed(seed)
    cfg = GPTConfig(sequence_len=2 * rows.row_len, vocab_size=layout.vocab_size, n_layer=2, n_head=2, n_kv_head=2,
                    n_embd=32, n_token_types=layout.n_token_types)
    trunk = GPT(cfg)
    trunk.init_weights()
    with torch.no_grad():
        for p in trunk.parameters():
            p.normal_(0, 0.2)
    head = LMHead(cfg.n_embd, cfg.vocab_size)
    torch.nn.init.normal_(head.lm_head.weight, std=0.5)
    return trunk, head


def random_clips(rows, n, seed=1):
    """n assembled rows of random codes and actions [n, row_len]."""
    g = torch.Generator().manual_seed(seed)
    c = rows.contract
    return rows.assemble(torch.randint(0, N_VIDEO, (n, c["code_len"]), generator=g).numpy(),
                         torch.randint(0, spec.N_ACTIONS, (n, c["n_action_tokens"]), generator=g).numpy())
