"""
train_wm — the Orchestrator: assemble a discrete video world model, then train it.

This is the whole assembly, in one readable file, in the order it happens:

    vocab  ->  row layout  ->  model  ->  data  ->  objective  ->  evaluator  ->  Trainer

Read it top to bottom and you know what this run is. Nothing is discovered, nothing
is registered, nothing is injected: components are imported and wired here, which is
core's "orchestrators own assembly" principle and its library-over-framework stance.
The Trainer executes; it does not decide.

WHAT THE MODEL IS. A decoder-only transformer over a shared vocabulary whose bands
are [text | control | video | action]. A clip of 17 game frames becomes 5 latent
frames of 256 discrete codes each (Cosmos DV4x8x8), interleaved with the action ids
that drive them. Frame 0 is given; the other 4 latent frames are predicted by masked
(absorbing-state) diffusion, one latent frame per diffusion block — so the model
learns "given what I have seen and the buttons pressed, what does the world look
like next", which is what makes it a world model rather than a video generator.

Usage:
    python -m exemplars.nano_world_model.train_wm
    python -m exemplars.nano_world_model.train_wm max_steps=200 use_compile=false
    torchrun --nproc_per_node=2 --standalone \
        -m exemplars.nano_world_model.train_wm parallel=ddp

Config: configs/train_wm.yaml (co-located; inherits mechanism defaults from
core/configs/train_base.yaml).
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from exemplars.nano_world_model import spec

spec.pin_tokenizer()            # MUST precede the modalities.text import (see spec.py)

import modalities.control                                          # noqa: E402
import modalities.text                                             # noqa: E402
from modalities.assembler import Modality, build_layout            # noqa: E402
from modalities.control import make_control_resolver               # noqa: E402

from core.model.gpt import GPT, GPTConfig                          # noqa: E402
from core.training.trainer import Trainer, create_optimizers        # noqa: E402
from core.utils import print0                                      # noqa: E402

from core.parallel import NanoDDP, block_buckets                   # noqa: E402
from core.training.model_setup import (                            # noqa: E402
    build_system, compile_blocks)

from exemplars.nano_world_model import row_layout             # noqa: E402
from exemplars.nano_world_model.autoregressive import (       # noqa: E402
    Autoregressive, AutoregressiveSystem)
from exemplars.nano_world_model.block_diffusion import (      # noqa: E402
    BlockDiffusion, BlockDiffusionSystem)
from exemplars.nano_world_model.rope3d import install_rope3d  # noqa: E402
from exemplars.nano_world_model.dataset import (              # noqa: E402
    VideoRowDataset, VideoRowLoader)
from exemplars.nano_world_model.evaluator import (            # noqa: E402
    NELBOEvaluator, NLLEvaluator)


def assemble_vocab():
    """[text | control | video | action] -> (layout, control_resolver).

    The video band is declared INLINE as a manifest, exactly as motion once was
    before it earned a modalities/ package: a Modality is just a name, a type id,
    and a size. The size is a FACT about the frozen Cosmos codec, stated in spec.py
    rather than read off a live tokenizer — training needs the number, not the
    600MB of TorchScript that can produce codes.

    vocab_size and n_token_types are facts of THIS assembly. They are never config
    constants, because a config constant can disagree with the artifact.
    """
    tokenizer = modalities.text.get_tokenizer()
    bands = [
        modalities.text.manifest(tokenizer),
        modalities.control.manifest(),
        Modality(name="video", type_id=spec.VIDEO_TYPE_ID,
                 vocab_size=spec.CODEC_VOCAB, tokenizer=None),
        Modality(name="action", type_id=spec.ACTION_TYPE_ID,
                 vocab_size=spec.N_ACTIONS, tokenizer=None),
    ]
    layout = build_layout(bands)
    resolver = make_control_resolver(bands[1], layout)

    # Protocol lock: the delimiters this project reserves must still resolve.
    for name in (spec.VIDEO_START, spec.VIDEO_END, spec.MASK_SLOT):
        assert resolver.resolve(name) is not None, f"control slot {name} vanished"
    return layout, resolver


def build_row_layout(layout, resolver, contract):
    """The row: where codes, actions and delimiters sit, and the block spans."""
    return row_layout.RowLayout(
        contract,
        video_offset=layout.offset(spec.VIDEO_TYPE_ID),
        action_offset=layout.offset(spec.ACTION_TYPE_ID),
        control_ids={"bos": resolver.resolve("bos"), "eos": resolver.resolve("eos"),
                     "video_start": resolver.resolve(spec.VIDEO_START),
                     "video_end": resolver.resolve(spec.VIDEO_END)},
        n_actions=spec.N_ACTIONS,
    )


@hydra.main(version_base=None, config_path="configs", config_name="train_wm")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    obj_name = config["objective"]
    assert obj_name in ("diffusion", "ar"), f"objective must be diffusion or ar, got {obj_name!r}"
    print0("=" * 80)
    print0(f"train_wm — discrete video world model ({obj_name}) on core")
    print0("=" * 80)

    # --- vocabulary + shape contract -----------------------------------------
    layout, resolver = assemble_vocab()
    contract = spec.shape_contract(config["clip"]["frames"], config["clip"]["res"])
    rows = build_row_layout(layout, resolver, contract)
    row_layout.use_compiled_flex_attention()
    print0(f"\nvocab: ranges={dict(layout.ranges)} -> {layout.vocab_size} ids, "
           f"{layout.n_token_types} token types")
    print0(f"clip:  {contract}")
    print0(f"row:   {rows}")

    # Block diffusion feeds the model BOTH streams — [clean | noisy] — so its sequence
    # is twice the row; autoregression reads one row causally. Everything downstream
    # (rope table, MFU, tokens/s) is sized from this, so throughput counts positions
    # truly computed rather than positions nominally present.
    sequence_len = 2 * rows.row_len if obj_name == "diffusion" else rows.row_len
    config["sequence_len"] = sequence_len
    config["total_batch_size"] = config["total_batch_rows"] * sequence_len

    # --- model ----------------------------------------------------------------
    model_config = config["model"]
    gpt_config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=layout.vocab_size,
        n_layer=model_config["depth"],
        n_head=model_config["n_head"],
        n_kv_head=model_config["n_kv_head"],
        n_embd=model_config["dim"],
        n_token_types=layout.n_token_types,
    )
    setup = build_system(GPT, gpt_config, use_compile=False,     # compiled per block below
                         seed=config["seed"], parallel=config["parallel"])
    base, rank, world_size = setup["system"], setup["rank"], setup["world_size"]
    device = setup["device"]

    # The objective goes INSIDE the System, which is how core says to train something
    # other than next-token CE (core/model/system.py: "write your own System
    # satisfying the same loss(batch) contract — core does not change"). Consequence:
    # core's Trainer runs this unmodified and this project has no Trainer subclass.
    if obj_name == "diffusion":
        objective = BlockDiffusion(rows, layout, mask_id=resolver.resolve(spec.MASK_SLOT),
                                   t_min=config["diffusion"]["t_min"],
                                   t_max=config["diffusion"]["t_max"], device=device)
        system = BlockDiffusionSystem(base.trunk, base.head, objective)
    else:
        objective = Autoregressive(rows)
        system = AutoregressiveSystem(base.trunk, base.head, objective)
    system.arch = base.arch          # assembly fact for checkpoint self-description

    # Position encoding, in two moves. First the 3D tables — every objective
    # trains on (t, y, x) coordinates (rope3d.py; a checkpoint load does not
    # touch them, the buffers are non-persistent). Then, for diffusion only,
    # the mirror: it makes the noisy copy of a position carry the SAME rotary
    # phase as its clean twin. An autoregressive row has no twin.
    # Both swapped BEFORE compile so the compiled blocks see the final tables;
    # the swap itself is a trunk attribute, outside the block graphs.
    install_rope3d(system.trunk, rows)
    if obj_name == "diffusion":
        rows.install_mirror_rope(system.trunk)
    if config.get("use_compile", True):
        compile_blocks(system.trunk)

    # --- data -----------------------------------------------------------------
    cache = spec.cache_dir(contract["frames"], contract["res"])
    train_set = VideoRowDataset(cache, "train")
    val_set = VideoRowDataset(cache, "val")
    assert train_set.contract == contract, (
        f"cache contract {train_set.contract} != run contract {contract} — rebuild the "
        f"cache with build_cache.py for this clip length")
    dataloader = VideoRowLoader(train_set, rows, config["device_batch_size"],
                                seed=config["seed"], device=device)
    print0(f"\ndata:  {train_set}\n       {val_set}\n       {dataloader}")

    # --- the ruler -------------------------------------------------------------
    if obj_name == "diffusion":
        evaluators = [NELBOEvaluator(
            objective, val_set, rows, n_rows=config["evaluation"]["val_rows"],
            t_grid=config["evaluation"]["t_grid"], batch=config["evaluation"]["batch"],
            interval_steps=config["evaluation"]["interval_steps"], device=device)]
    else:
        evaluators = [NLLEvaluator(
            objective, val_set, rows, n_rows=config["evaluation"]["val_rows"],
            batch=config["evaluation"]["batch"],
            interval_steps=config["evaluation"]["interval_steps"], device=device)]
    print0(f"eval:  {evaluators[0].describe()}")

    # --- gradient synchronization (explicit; the Trainer calls it in the step) --
    optimizers = create_optimizers(system, config["optimizer"], world_size=world_size)
    ddp = None
    if world_size > 1:
        assert config["parallel"] == "ddp", (
            "multi-GPU with parallel=fsdp is handled inside build_system; NanoDDP is "
            "for parallel=ddp only")
        # Bucket order = gradient completion order: the head finishes first, then the
        # blocks from the last inwards, embeddings last (see block_buckets).
        ddp = NanoDDP([list(system.head.parameters())] + block_buckets(system.trunk),
                      module=system)
        print0(f"ddp:   NanoDDP over {len(ddp.buckets)} buckets")

    trainer = Trainer(
        system=system, optimizers=optimizers, dataloader=dataloader, config=config,
        rank=rank, world_size=world_size, evaluators=evaluators, ddp=ddp)
    # This file DECIDES to synchronize (constructing NanoDDP above is that decision);
    # the Trainer is where it HAPPENS, as one `with sync_gradients(...)` around the
    # backward — so the loop that reads as "one optimizer step" also shows where the
    # replicas exchange gradients.

    print0("\nStarting training...\n")
    trainer.train()
    print0("\n" + "=" * 80)
    print0(f"✓ train_wm completed — best {evaluators[0].metric} {evaluators[0].best:.4f}")
    print0("=" * 80)


if __name__ == "__main__":
    main()
