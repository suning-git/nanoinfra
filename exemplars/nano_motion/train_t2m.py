"""train_t2m.py — THE ORCHESTRATOR. Assemble, then hand the loop to core's Trainer.

    data/encode.py -> [train_t2m.py] -> generate.py -> render.py

Read this file first. It is where the pieces meet: a shared vocabulary spanning
text, control and motion; a data source that assembles rows in that vocabulary; a
plain GPT trunk from core; and core's Trainer running the loop. Nothing here
subclasses the Trainer or teaches core what motion is — the assembly is the
orchestrator's job and the loop is core's, which is the whole architectural point.

Two modes, chosen by whether your data has captions:

  t2m       [bos, text_start, <caption>, text_end, motion_start, <codes>, motion_end, eos]
            with loss on the MOTION half only — the model is not being taught to
            write captions, and supervising them would spend capacity on it.
  motion    [bos, motion_start, <codes>, motion_end, eos], loss on everything.
            LAFAN1 has no captions, so this is what it trains.

    python -m exemplars.nano_motion.train_t2m                     # mode from the data
    python -m exemplars.nano_motion.train_t2m max_steps=200       # a smoke run
    torchrun --nproc_per_node=2 --standalone \
        -m exemplars.nano_motion.train_t2m parallel=ddp

The vocabulary is assembled, not configured. `vocab_size` and `n_token_types` are
FACTS of the assembly — writing them in the config would let the config disagree
with the bands, and a band offset that is wrong by one silently retargets every
motion token.
"""

import os
import sys

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from exemplars.nano_motion import spec  # noqa: E402

spec.pin_tokenizer()                                                # noqa: E402

import modalities.control                                           # noqa: E402
import modalities.text                                              # noqa: E402
from modalities.assembler import Modality, build_layout             # noqa: E402
from modalities.control import make_control_resolver                # noqa: E402
from modalities.motion.data import paths as motion_paths            # noqa: E402
from modalities.motion.data.sources import (                        # noqa: E402
    MotionDataSource, T2MDataSource)

from core.evaluation.evaluator import Evaluator                      # noqa: E402
from core.tokenization.vocab_layout import VocabLayout               # noqa: E402
from core.model.gpt import GPT, GPTConfig                           # noqa: E402
from core.parallel import NanoDDP, block_buckets                    # noqa: E402
from core.training.model_setup import build_system, compile_blocks  # noqa: E402
from core.training.trainer import Trainer, create_optimizers        # noqa: E402
from core.utils import print0                                       # noqa: E402


def assemble_vocab(codec):
    """[text | control | motion] -> (layout, control_resolver).

    The motion band's size comes from the CODEC, not from a constant here: the codec
    is the thing that decides how many distinct codes exist, and a mismatch between
    what the band reserves and what the encoder emits puts motion tokens on top of
    whatever follows in the vocabulary. Reading it off the artifact makes that
    disagreement impossible rather than merely unlikely.
    """
    tokenizer = modalities.text.get_tokenizer()
    bands = [
        modalities.text.manifest(tokenizer),
        modalities.control.manifest(),
        Modality(name="motion", type_id=spec.MOTION_TYPE_ID,
                 vocab_size=codec.vocab_size, tokenizer=codec),
    ]
    layout = build_layout(bands)
    resolver = make_control_resolver(bands[1], layout)
    for name in (spec.MOTION_START, spec.MOTION_END, "bos", "eos"):
        assert resolver.resolve(name) is not None, f"control slot {name} vanished"
    return layout, resolver, tokenizer


def pick_mode(source, split="train"):
    """t2m if a caption cache exists for this source, else unconditional motion."""
    t2m = os.path.join(motion_paths.PROCESSED_DIR, f"t2m_{source}_{split}.npz")
    motion = os.path.join(motion_paths.PROCESSED_DIR, f"{source}_{split}_codes.npz")
    if os.path.exists(t2m):
        return "t2m", os.path.basename(t2m)
    if os.path.exists(motion):
        return "motion", os.path.basename(motion)
    raise SystemExit(f"neither {t2m} nor {motion} exists — run data/encode.py first")


def make_source(mode, cache, cfg, tokenizers, split, seq_len, rank, world):
    """One DataSource, configured for a split. The RECIPE (which tokens go where, and
    which are supervised) comes from the config — it is a declaration, not code."""
    src_cfg = {
        "recipe": cfg["recipe"][mode],
        "sequence_len": seq_len,
        "split": split,
        "dataset": cfg["source"],
        "cache": cache.replace("train", split),
        "max_text_tokens": spec.MAX_TEXT_TOKENS,
        "seed": cfg["seed"],
        "rank": rank, "world_size": world,
        # cuda:<local_rank>, not "cuda": see the note in main() about device 0.
        "device": (f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
                   if torch.cuda.is_available() else "cpu"),
    }
    cls = T2MDataSource if mode == "t2m" else MotionDataSource
    return cls(src_cfg, tokenizers)


class SupervisedCEEvaluator(Evaluator):
    """Mean next-token CE over the SUPERVISED tokens of the val split.

    Batches are assembled ONCE, at construction. A val set that is re-sampled every
    time it is used is a ruler that moves, and two runs measured with it are not
    comparable — which matters more here than the small cost of holding them.
    """

    def __init__(self, ecfg, source, layout):
        self.interval_steps = ecfg.get("interval_steps", 1000)
        self.eval_at = {int(s) for s in ecfg["eval_at"]} if ecfg.get("eval_at") else None
        self.best = float("inf")
        it = iter(source)
        self._batches = []
        for _ in range(ecfg.get("n_batches", 20)):
            rows = [next(it) for _ in range(ecfg.get("batch", 8))]
            toks = torch.stack([r["tokens"] for r in rows])
            lw = torch.stack([r["loss_weights"] for r in rows])
            self._batches.append((toks, lw, layout.classify_token_types(toks)))

    def describe(self):
        return (f"supervised-token CE on {len(self._batches)} frozen val batches, "
                f"every {self.interval_steps} steps")

    @torch.no_grad()
    def evaluate(self, system, autocast_ctx):
        tot = 0.0
        with autocast_ctx:
            for toks, lw, types in self._batches:
                inp, tgt, ity, w = toks[:, :-1], toks[:, 1:], types[:, :-1], lw[:, 1:]
                logits = system.head(system.trunk(inp, token_types=ity))
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     tgt.reshape(-1), reduction="none").reshape(tgt.shape)
                tot += (ce * w).sum().item() / w.sum().clamp(min=1).item()
        mean = tot / len(self._batches)
        self.best = min(self.best, mean)
        return {"val/motion_ce": mean, "val/motion_ce_best": self.best}


class SourceLoader:
    """The infinite, resumable iterator core's Trainer expects.

    It also does the one shaping step that matters: the DataSource marks which tokens
    carry loss with a per-token WEIGHT, and core's System takes a target tensor. So
    unsupervised positions become IGNORE_INDEX targets rather than a separate mask
    argument — which means the caption half costs nothing, the fused cross-entropy
    path still applies (it honours ignore_index), and core needs to know nothing
    about what "supervised" means here.

    Its state is the number of rows served, an integer with no rank in it, so a run
    checkpointed on two GPUs resumes on one.
    """

    def __init__(self, source, batch_size, layout):
        self.source, self.bs, self.layout = source, batch_size, layout
        self._it, self._n = iter(source), 0

    def __iter__(self):
        return self

    def __next__(self):
        rows = [next(self._it) for _ in range(self.bs)]
        self._n += len(rows)
        toks = torch.stack([r["tokens"] for r in rows])
        lw = torch.stack([r["loss_weights"] for r in rows])
        types = self.layout.classify_token_types(toks)
        targets = torch.where(lw[:, 1:] > 0, toks[:, 1:],
                              torch.full_like(toks[:, 1:], VocabLayout.IGNORE_INDEX))
        return {"idx": toks[:, :-1], "targets": targets, "token_types": types[:, :-1]}

    def state_dict(self):
        return {"rows": self._n}

    def load_state_dict(self, state):
        self._it = iter(self.source)
        for _ in range(state.get("rows", 0) // self.bs):
            next(self)


@hydra.main(version_base=None, config_path="configs", config_name="train_t2m")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_dist = world > 1

    # Pin this process to its GPU BEFORE anything allocates. The data sources below
    # are built first and put their tensors on "cuda", which means device 0 on every
    # rank unless the current device has already been set — a mismatch that only
    # surfaces at the first forward, several hundred lines later.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # --- the codec decides the motion band, so it is loaded first ---------------
    from modalities.motion.tokenizers._convae import MotionCodec
    import importlib
    if config.get("codec_ckpt"):
        codec = MotionCodec(config["codec_ckpt"], device="cuda")
        codec_name = os.path.basename(config["codec_ckpt"])
    else:
        mod = importlib.import_module(f"modalities.motion.tokenizers.{config['tokenizer']}")
        codec, codec_name = mod.load(device="cuda"), config["tokenizer"]

    layout, resolver, text_tok = assemble_vocab(codec)
    mode, cache = pick_mode(config["source"])
    seq_len = config["sequence_len"]

    print0("=" * 80)
    print0(f"nano_motion — {mode} on {config['source']}, codec {codec_name}")
    print0(f"  vocab {layout.vocab_size} across {layout.n_token_types} bands; "
           f"motion band at {layout.offset(spec.MOTION_TYPE_ID)} ({codec.vocab_size} codes)")
    print0(f"  cache {cache}, sequence_len {seq_len}")
    print0("=" * 80)

    tokenizers = {"layout": layout, "control_resolver": resolver,
                  "text": text_tok, "motion": codec}

    # --- data ------------------------------------------------------------------
    train_src = make_source(mode, cache, config, tokenizers, "train", seq_len, rank, world)
    loader = SourceLoader(train_src, config["device_batch_size"], layout)

    evaluators = []
    try:
        val_src = make_source(mode, cache, config, tokenizers, "val", seq_len, 0, 1)
        evaluators.append(SupervisedCEEvaluator(config["evaluation"], val_src, layout))
    except FileNotFoundError as e:
        print0(f"  no val cache ({e}) — training without evaluation")

    # --- model -----------------------------------------------------------------
    gpt_config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=layout.vocab_size,
        n_layer=config["model"]["depth"],
        n_head=config["model"]["n_head"],
        n_kv_head=config["model"]["n_kv_head"],
        n_embd=config["model"]["dim"],
        n_token_types=layout.n_token_types,
    )
    setup = build_system(GPT, gpt_config, use_compile=False,
                         parallel=("ddp" if is_dist and config["parallel"] == "ddp"
                                   else config["parallel"]))
    system = setup["system"]
    if config.get("use_compile", True):
        compile_blocks(system.trunk)

    ddp = None
    if is_dist and config["parallel"] == "ddp":
        # The LM head completes before any block, so it goes at the FRONT — bucket
        # order is part of NanoDDP's contract, not a detail (see core/parallel).
        ddp = NanoDDP([[p for p in system.head.parameters() if p.requires_grad]]
                      + block_buckets(system.trunk), module=system)

    optimizers = create_optimizers(system, config["optimizer"], world)

    trainer = Trainer(
        system=system, optimizers=optimizers, dataloader=loader, config=config,
        rank=rank, world_size=world, evaluators=evaluators, ddp=ddp,
    )
    trainer.train()
    print0(f"\n{'=' * 80}\n✓ nano_motion training complete\n{'=' * 80}")


if __name__ == "__main__":
    main()
