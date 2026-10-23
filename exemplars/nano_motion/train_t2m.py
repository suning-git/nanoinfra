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

from functools import partial                                       # noqa: E402

from core.data import MixedDataLoader, PackingBatcher, PaddingBatcher  # noqa: E402
from core.data.layout import collate_samples, padding_layout        # noqa: E402
from core.data.supervision import IGNORE, build_NTP_supervised_sample  # noqa: E402
from core.evaluation.evaluator import Evaluator, LossEvaluator       # noqa: E402
from core.evaluation.metrics import mean_ce                         # noqa: E402
from core.model.gpt import GPT, GPTConfig                           # noqa: E402
from core.model.system import (prepare_packed_AR_trunk_inputs,      # noqa: E402
                               prepare_unpacked_AR_trunk_inputs)
from core.parallel import NanoDDP, block_buckets                    # noqa: E402
from core.training.model_setup import build_system, compile_blocks  # noqa: E402
from core.training.optim import build_optimizers                    # noqa: E402
from core.training.trainer import Trainer                           # noqa: E402
from core.utils import print0, process_identity, set_torch_global_defaults  # noqa: E402


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
    which are supervised) comes from the config — it is a declaration, not code.
    `seq_len` is T, the positions per row; a sample is at most `sample_len` raw tokens
    (config, default T + 1: the next-token shift). The source hands out CPU tensors; the
    loader moves whole batches."""
    sample_len = cfg.get("sample_len") or seq_len + 1
    src_cfg = {
        "recipe": cfg["recipe"][mode],
        ("max_sample_len" if mode == "t2m" else "sample_len"): sample_len,
        "split": split,
        "dataset": cfg["source"],
        "cache": cache.replace("train", split),
        "max_text_tokens": spec.MAX_TEXT_TOKENS,
        "seed": cfg["seed"],
        "rank": rank, "world_size": world,
    }
    cls = T2MDataSource if mode == "t2m" else MotionDataSource
    return cls(src_cfg, tokenizers)


def pad_values(layout, resolver):
    """Pad values for prepared motion/T2M samples: eos (what the sources always padded with) and its type."""
    eos = resolver.resolve("eos")
    eos_type = int(layout.classify_token_types(torch.tensor([eos]))[0])
    return {"idx": eos, "targets": IGNORE, "token_types": eos_type, "target_types": eos_type}


def prepare_batch(rows, seq_len, pads):
    """Raw source samples -> one padded batch through core's per-sample path: each row prepared by
    build_NTP_supervised_sample (L + 1 raw tokens -> L positions; unselected targets IGNORE), then one row of
    seq_len = T positions per sample (padding_layout + collate_samples). A full motion window fills its row; a T2M
    pair, yielded at its true length, is padded after the shift."""
    samples = [build_NTP_supervised_sample(r) for r in rows]
    shape = (len(samples), seq_len)
    lengths = [s["data"]["idx"].shape[0] for s in samples]
    return collate_samples(samples, padding_layout(lengths, shape), shape=shape, pad_values=pads)


class SupervisedCEEvaluator(Evaluator):
    """Mean next-token CE over the SUPERVISED tokens of the val split: every supervised token counts once (core's
    mean_ce, the nats over all of them divided by their count), so the reading does not depend on how the rows are
    grouped into batches. (Before 2026-10-21 it averaged the batch means, which weights a token by its batch: the
    same on full LAFAN1 windows, not on T2M pairs, whose supervised lengths vary.)

    Batches are assembled ONCE, at construction. A val set that is re-sampled every
    time it is used is a ruler that moves, and two runs measured with it are not
    comparable — which matters more here than the small cost of holding them.

    One GPU or many, the same batches: every rank assembles the same batches (the val source is rank 0 of 1) and
    evaluates its shard of them, batches rank, rank + N, ...; core's LossEvaluator adds the nats and the token counts
    over the ranks before dividing, so every batch is evaluated once, on one GPU, and a GPU runs ceil(K / N) of the K
    forwards (two GPUs under NanoDDP took about half the time, 2026-10-21). The reading is the same on the same compute
    path (up to the order of the fp64 sums): one GPU and NanoDDP both compile block by block and read the same; FSDP
    casts the floating inputs of its sharded modules to bf16 and reads 4e-5 to 1.3e-4 away (measured 2026-10-21). It
    runs core's path (core/evaluation/eval_loss.py): the System's prepare_trunk_inputs, as in training, then
    head.loss_per_token.
    """

    def __init__(self, ecfg, source, layout, resolver, seq_len, device):
        self.interval_steps = ecfg.get("interval_steps", 1000)
        self.eval_at = {int(s) for s in ecfg["eval_at"]} if ecfg.get("eval_at") else None
        self.best = float("inf")
        pads = pad_values(layout, resolver)
        # The first n_batches * batch samples of the val source, or one pass over it when it holds fewer: the source
        # goes round again after a pass (its sampler is endless), and a sample taken twice would count twice
        # (LAFAN1's 159 val windows against the default 20 batches of 8). A sample is known by its meta ({"window":
        # j}, {"pair": j}); the first one seen again ends the pass, and the last batch may be short.
        rows_per_batch = ecfg.get("batch", 8)
        wanted = rows_per_batch * ecfg.get("n_batches", 20)
        rows, seen, it = [], set(), iter(source)
        while len(rows) < wanted:
            sample = next(it, None)
            if sample is None or repr(sample["meta"]) in seen:
                break
            seen.add(repr(sample["meta"]))
            rows.append(sample)
        self.whole_set = len(rows) < wanted
        frozen = [prepare_batch(rows[i:i + rows_per_batch], seq_len, pads)    # on the CPU
                  for i in range(0, len(rows), rows_per_batch)]
        self.n_samples, self.n_batches = len(rows), len(frozen)
        self.n_supervised = sum(int((b["targets"] != IGNORE).sum()) for b in frozen)

        # This rank's shard: batches rank, rank + N, ... Every rank runs ceil(K / N) of them, because under FSDP
        # every forward is a collective; a rank left one short also runs a copy of the first batch with every target
        # IGNORE, which mean_ce counts as nothing (a metric whose denominator does not come from the targets would
        # count it; this evaluator has none).
        identity = process_identity()
        self.world_size = identity.world_size
        shard = frozen[identity.rank::identity.world_size]
        steps = -(-len(frozen) // identity.world_size)
        if len(shard) < steps:
            shard.append({**frozen[0], "targets": torch.full_like(frozen[0]["targets"], IGNORE)})
        shard = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()} for b in shard]   # moved once
        self._eval = LossEvaluator(shard, steps, {"val/motion_ce": mean_ce()})

    def describe(self):
        shared = (f", {self._eval.eval_steps} forwards per rank on {self.world_size}"
                  if self.world_size > 1 else "")
        whole = ", the whole val set" if self.whole_set else ""
        return (f"supervised-token CE on {self.n_batches} frozen val batches of {self.n_samples} samples{whole} "
                f"({self.n_supervised:,} supervised tokens){shared}, every {self.interval_steps} steps")

    def evaluate(self, system, autocast_ctx):
        mean = self._eval.evaluate(system, autocast_ctx)["val/motion_ce"]   # LossEvaluator enters the context
        self.best = min(self.best, mean)
        return {"val/motion_ce": mean, "val/motion_ce_best": self.best}


@hydra.main(version_base=None, config_path="configs", config_name="train_t2m")
def main(cfg: DictConfig) -> None:
    # Process-wide PyTorch settings (CUDA allocator, TF32) are the entry's to make, before anything uses CUDA.
    set_torch_global_defaults()
    config = OmegaConf.to_container(cfg, resolve=True)
    batching = config.get("batching", "padding")
    if batching not in ("padding", "packing"):
        raise ValueError(f"batching must be padding or packing, got {batching!r}")
    identity = process_identity()
    world, rank, local_rank = identity.world_size, identity.rank, identity.local_rank
    is_dist = world > 1

    # Pin this process to its GPU BEFORE anything allocates. The codec below is loaded
    # onto "cuda", which means device 0 on every rank unless the current device has
    # already been set — a mismatch that only surfaces at the first forward.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

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
    # The training stream: core's loader. Each raw window / pair is prepared on its own (next-token shift,
    # unsupervised targets IGNORE), then placed by the chosen batcher in rows of seq_len = T positions:
    # `batching: padding` one sample per row (the old layout), `packing` whole samples back to back (useful when
    # samples are shorter than a row — T2M pairs; a full motion window fills its row either way). Its state is the
    # source's sampler position (seed, epoch, index) and, merged over the ranks at each save, the pairs before it no
    # rank has trained, so a resumed run continues the stream on any number of GPUs, nothing skipped or repeated; the
    # previous SourceLoader's {"rows": n} never reached a checkpoint (its batches carried no state_dict and it had no
    # set_state), so there is no older state to read.
    # The source hands out CPU samples; the loader collates on the CPU and moves each batch field to this
    # rank's GPU once.
    train_src = make_source(mode, cache, config, tokenizers, "train", seq_len, rank, world)
    loader = MixedDataLoader(
        {mode: (train_src, 1.0)}, build_supervised_sample_fn=build_NTP_supervised_sample,
        create_batcher_fn=partial(PackingBatcher if batching == "packing" else PaddingBatcher,
                                  shape=(config["device_batch_size"], seq_len),
                                  pad_values=pad_values(layout, resolver)),
        device=device)

    # An empty list is what the Trainer reads as "do not evaluate" — it is handed a
    # LIST, so an empty one already says it and core has no evaluation on/off key of
    # its own, by design. Two ways to end up with one here: asked for
    # (`evaluation.enabled=false`, for throughput benchmarks and smoke runs), or
    # forced (no val cache on disk).
    evaluators = []
    if not config["evaluation"].get("enabled", True):
        print0("  evaluation disabled (evaluation.enabled=false)")
    else:
        try:
            val_src = make_source(mode, cache, config, tokenizers, "val", seq_len, 0, 1)
            evaluators.append(SupervisedCEEvaluator(config["evaluation"], val_src, layout, resolver, seq_len,
                                                    device))
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
    # head_ce: compiled, the fastest arm (2026-10-22, one RTX 5090: 35 -> 23 ms a step against liger, peak memory
    # 3.50 -> 3.53 GiB; RESULTS.md was measured with liger). Training routes through LMSystem.loss -> head.loss, so
    # the arm is a recipe fact here; state it rather than inherit core's naive default. Its evaluator goes through
    # core's evaluate_loss, i.e. this arm's head.loss_per_token and this System's prepare_trunk_inputs.
    # padding: one sample per row, causal SDPA; packing: samples isolated by doc_id (flex), each from position 0.
    # Packed rows train through flex attention, which must be compiled: with compile_trunk=false the trunk refuses
    # the first training step (an uncompiled flex backward may be silently wrong).
    # param_dtype: what parameters, gradients and AdamW moments are stored in (train_base.yaml).
    setup = build_system(GPT, gpt_config, head_ce="compiled",
                         parallel=("ddp" if is_dist and config["parallel"] == "ddp"
                                   else config["parallel"]),
                         prepare_trunk_inputs_fn=(prepare_packed_AR_trunk_inputs if batching == "packing"
                                                  else prepare_unpacked_AR_trunk_inputs),
                         param_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[config["param_dtype"]])
    system = setup["system"]
    if config.get("compile_trunk", True):
        compile_blocks(system.trunk)

    ddp = None
    if is_dist and config["parallel"] == "ddp":
        # The LM head completes before any block, so it goes at the FRONT — bucket
        # order is part of NanoDDP's contract, not a detail (see core/parallel).
        ddp = NanoDDP([[p for p in system.head.parameters() if p.requires_grad]]
                      + block_buckets(system.trunk), module=system)

    optimizers = build_optimizers(system, config["optimizer"])

    trainer = Trainer(
        system=system, optimizers=optimizers, dataloader=loader, config=config,
        rank=rank, world_size=world, evaluators=evaluators, ddp=ddp,
    )
    trainer.train()
    print0(f"\n{'=' * 80}\n✓ nano_motion training complete\n{'=' * 80}")


if __name__ == "__main__":
    main()
