"""
train.py — THE ORCHESTRATOR. One file, three models.

    python -m projects.nano_multimodal.train --config-name motion
    python -m projects.nano_multimodal.train --config-name video
    python -m projects.nano_multimodal.train --config-name text
    torchrun --nproc_per_node=2 --standalone \
        -m projects.nano_multimodal.train --config-name video parallel=ddp

Read this file and the three configs side by side and you have the whole course:
the configs differ, this file does not. What the three lines share is everything
below the data source — one model class, one loss, one loop:

    vocab (this line's active bands) -> model -> data -> ruler -> Trainer

Nothing is discovered, nothing is registered, nothing is injected. Components are
imported and wired here, which is core's "orchestrators own assembly" and its
library-over-framework stance. The Trainer executes; it does not decide.

WHY THERE IS NO PER-LINE BRANCH IN THIS FILE. The three lines differ in four
things and all four are declarations: which bands to activate (spec.LINES), which
DataSource class to build (a string, resolved through a LOCAL mapping), how the
row is laid out (a SequenceRecipe template in yaml), and which positions carry
loss (loss_weights, written by the data source). Everything else — GPT trunk,
LMHead, next-token CE, MixedDataLoader, Trainer — is literally the same object.
If you ever find yourself adding `if line == ...` here, that is the signal the
difference belongs in a data source or a config instead.
"""

import os

import hydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig, OmegaConf


class _CoreConfigs(SearchPathPlugin):
    """Make core/configs findable no matter WHICH of the three line configs is
    primary. Declaring `hydra.searchpath` in a yaml only works while that yaml is
    the one invoked — hydra ignores the key in any config reached through a
    defaults list, and our three entry configs all inherit base.yaml, which is what
    needs train_base. A plugin applies either way."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="nano_multimodal", path="pkg://core.configs")


Plugins.instance().register(_CoreConfigs)

import torch                                                  # noqa: E402

from projects.nano_multimodal import spec                     # noqa: E402

spec.pin_tokenizer()                                          # MUST precede modalities.text

from core.model.gpt import GPT                                # noqa: E402
from core.parallel import NanoDDP, block_buckets              # noqa: E402
from core.training.model_setup import build_system, compile_blocks, compile_system_trunk  # noqa: E402
from core.training.trainer import create_optimizers           # noqa: E402
from core.utils import print0                                 # noqa: E402

from projects.nano_multimodal import assembly                 # noqa: E402

assembly.register_resolvers()   # ${eval:} / ${repo:} — shared with the browser and tests
from projects.nano_multimodal.evaluator import SupervisedCE   # noqa: E402
from projects.nano_multimodal.metrics import TrainerWithJsonl  # noqa: E402


def batch_plan(config, sequence_len, world_size):
    """Rows -> tokens, with the divisibility checked HERE, in a sentence a student
    can act on.

    core's Trainer asserts `total_batch_size % (device_batch_size * sequence_len *
    world_size) == 0` and its message talks about tokens without ever mentioning
    GPUs — so a group with 3 or 5 cards dies at startup on a number none of them
    chose. Batch is therefore declared in ROWS here, and the awkward arithmetic is
    done and explained before core ever sees it.
    """
    dbs = config["device_batch_size"]
    rows = config["total_batch_rows"]
    per_step = dbs * world_size
    if rows % per_step:
        fixed = max(per_step, (rows // per_step) * per_step)
        raise SystemExit(
            f"\ntotal_batch_rows={rows} is not divisible by device_batch_size={dbs} "
            f"x world_size={world_size} (= {per_step} rows per micro-step).\n"
            f"  Pick a multiple of {per_step} — e.g. total_batch_rows={fixed} — or "
            f"change device_batch_size.\n"
            f"  (This is core's tokens-per-step assertion, caught here where it can "
            f"still name the GPU count.)\n")
    return {"grad_accum": rows // per_step, "total_batch_size": rows * sequence_len}


@hydra.main(version_base=None, config_path="configs", config_name="text")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    line = config["line"]
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Pin this process to its GPU BEFORE anything allocates: the data sources put
    # their tensors on "cuda", which means device 0 on every rank unless the current
    # device has already been set — a mismatch that only surfaces at the first
    # forward, hundreds of lines later.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    print0("=" * 80)
    print0(f"nano_multimodal — line: {line}")
    print0("=" * 80)

    # --- 1. vocabulary: this line's ACTIVE bands, stacked --------------------
    # vocab_size and n_token_types are FACTS OF THE ASSEMBLY, absent from every
    # yaml on purpose: a config constant can disagree with the artifact, and a band
    # offset that is wrong by one trains perfectly and decodes into nonsense.
    vocab = assembly.assemble_vocab(line, config)
    print0(f"\nvocab   {vocab.describe()}")
    print0(f"        canonical table = {vocab.canonical.vocab_size} ids across "
           f"{len(spec.BAND_ORDER)} bands; this line pays for "
           f"{vocab.layout.vocab_size}")

    plan = batch_plan(config, vocab.sequence_len, world_size)
    config["sequence_len"] = vocab.sequence_len
    config["total_batch_size"] = plan["total_batch_size"]
    print0(f"batch   {config['total_batch_rows']} rows/step x {vocab.sequence_len} tokens "
           f"= {plan['total_batch_size']:,} tokens; grad_accum={plan['grad_accum']} "
           f"(dbs {config['device_batch_size']} x world {world_size})")

    # --- 2. model: the same GPT + LMHead on every line -----------------------
    gpt_config = assembly.gpt_config_for(config, vocab)
    parallel = config.get("parallel", "ddp")
    setup = build_system(GPT, gpt_config, parallel=parallel,
                         head_ce=config.get("head_ce", "naive"),
                         seed=config.get("seed", spec.SEED))
    system, rank, world_size = setup["system"], setup["rank"], setup["world_size"]
    # Same rule as modalities/text/train_text.py, decided here with world_size known:
    # per block under NanoDDP (a whole-trunk compile makes AOTAutograd finalize every
    # gradient at the END of backward, pytorch#109774, so no all_reduce overlaps), the
    # whole trunk otherwise. NOTE (2026-09-10): on ONE device with parallel=ddp this
    # used to compile per block; it now takes the whole trunk like text — a deliberate
    # alignment, not a consequence of the API change.
    if config.get("compile_trunk", True):
        if world_size > 1 and parallel == "ddp":
            compile_blocks(system.trunk)
        else:
            compile_system_trunk(system, dynamic=True)

    ddp = None
    if world_size > 1 and parallel == "ddp":
        # Bucket ORDER is part of NanoDDP's contract: NCCL pairs collectives by
        # issue order, so every rank must issue them in the same sequence. The head
        # completes before any block, so it goes at the FRONT.
        ddp = NanoDDP([[p for p in system.head.parameters() if p.requires_grad]]
                      + block_buckets(system.trunk), module=system)
        print0(f"ddp     NanoDDP over {len(ddp.buckets)} buckets")

    # --- 3. data: the SAME call serve/browse.py makes ------------------------
    loader = assembly.build_loader(line, config, vocab, "train", device,
                                   config["device_batch_size"], rank, world_size)

    # --- 4. the ruler --------------------------------------------------------
    ecfg = config.get("evaluation") or {}
    val_loader = assembly.build_loader(line, config, vocab, "val", device,
                                       ecfg.get("batch", 8), rank=0, world_size=1)
    evaluators = [SupervisedCE(
        val_loader, n_batches=ecfg.get("n_batches", 20),
        interval_steps=ecfg.get("interval_steps", 500), eval_at=ecfg.get("eval_at"),
        metric=config.get("metric", f"val/{line}_ce"))]
    print0(f"\nruler   {evaluators[0].describe()}")

    # --- 5. run --------------------------------------------------------------
    optimizers = create_optimizers(system, config["optimizer"], world_size=world_size)
    ckpt_dir = config["checkpoint"].get("save_dir") or spec.ckpt_dir(line, config["model"]["depth"])
    config["checkpoint"]["save_dir"] = ckpt_dir

    trainer = TrainerWithJsonl(
        system=system, optimizers=optimizers, dataloader=loader, config=config,
        rank=rank, world_size=world_size, evaluators=evaluators, ddp=ddp,
        metrics_path=os.path.join(ckpt_dir, "metrics.jsonl"),
    )
    print0("\nStarting training...\n")
    trainer.train()
    print0("\n" + "=" * 80)
    print0(f"✓ nano_multimodal / {line} — best {evaluators[0].metric} "
           f"{evaluators[0].best:.4f}")
    print0("=" * 80)


if __name__ == "__main__":
    main()
