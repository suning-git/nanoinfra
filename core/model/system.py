"""LMSystem: the trunk+head composition the Trainer trains against.

A thin nn.Module container, assembled by the orchestrator. Its ONLY contract with
the Trainer is `loss(batch) -> scalar` plus being an nn.Module (for parameters() /
state_dict() / train()/eval()). Batch-dict unpacking lives HERE — the trunk and head
take tensors — so the Trainer stays ignorant of idx/token_types/targets.

Namespacing: registering `trunk` and `head` as submodules yields clean `trunk.*` /
`head.*` state_dict keys for free (checkpoint namespace, param-group selection,
freezing) — no separate bookkeeping needed.

FSDP: the System itself is NOT sharded (it owns no parameters); the trunk and the
head are each their own shard group (separate FSDP roots).

Compilation never replaces the trunk: the orchestrator compiles
it after assembly IN PLACE, as a whole (core/training/model_setup.compile_system_trunk,
nn.Module.compile) or per block (compile_blocks), never by build_system. So
`system.trunk` is always the one registered module: training (`loss`), teacher-forcing
evaluation and inference run it — compiled if the orchestrator compiled it — and its
state_dict keys stay `trunk.*` whether or not it is compiled. (A separate
torch.compile wrapper would be a second object: under the name `trunk` it puts
`_orig_mod.` into every key, and kept beside the registered module it needs a second
name and a rule for which caller takes which.)

Projects that want a different composition (e.g. a diffusion head, or multiple heads)
write their own System satisfying the same `loss(batch)` contract — core does not
change (examples over interfaces).
"""

import torch.nn as nn
from torch.nn.attention.flex_attention import and_masks

from core.model.masks import causal, default_mask_builder, same_sample

_NO_PREPARE = (
    "this System has no prepare_trunk_inputs: pass prepare_trunk_inputs_fn to LMSystem, build_system or load_system "
    "(core.model.system.prepare_unpacked_AR_trunk_inputs for one sample per row, prepare_packed_AR_trunk_inputs for "
    "several, or a project function batch -> (idx, kwargs)), or define the method in a subclass.")


class LMSystem(nn.Module):
    def __init__(self, trunk, head, *, prepare_trunk_inputs_fn=None):
        super().__init__()
        if prepare_trunk_inputs_fn is not None and type(self).prepare_trunk_inputs is not LMSystem.prepare_trunk_inputs:
            raise TypeError(f"{type(self).__name__} defines prepare_trunk_inputs and was also given "
                            f"prepare_trunk_inputs_fn: choose one, so that neither silently replaces the other")
        self.trunk = trunk
        self.head = head
        # a plain attribute kept off the module registry: a function, not a submodule
        object.__setattr__(self, "_prepare_trunk_inputs_fn", prepare_trunk_inputs_fn)

    def __setattr__(self, name, value):
        if name == "prepare_trunk_inputs":
            raise AttributeError("prepare_trunk_inputs is chosen when the System is built: "
                                 "LMSystem(trunk, head, prepare_trunk_inputs_fn=...), or the same argument to "
                                 "build_system / load_system, or a subclass method — not assigned afterwards")
        super().__setattr__(name, value)

    @property
    def config(self):
        """The trunk's config — the architectural facts of this System.

        Checkpoint self-description hangs off this: save_checkpoint_dcp records
        asdict(model.config) (tuples as lists) into meta.json, and _validate_model_config audits
        every load against it (both look the attribute up via `model.config`).
        """
        return self.trunk.config

    def prepare_trunk_inputs(self, batch):
        """batch -> (idx, trunk kwargs): the ONE translation from a batch (idx, token_types, doc_id, doc_pos, ...)
        to the trunk's arguments, shared by training (`loss`) and core's teacher-forcing LM evaluation. Not chosen
        here: the function comes in at construction (prepare_trunk_inputs_fn, taking the batch only), or a
        project's subclass defines this method. Without either it raises on first use (pure inference, which has
        no batch to translate, never calls it)."""
        if self._prepare_trunk_inputs_fn is None:
            raise RuntimeError(_NO_PREPARE)
        return self._prepare_trunk_inputs_fn(batch)

    def loss(self, batch):
        """Scalar training loss — the Trainer's entire view of the model world.

        The head's mean CE over the micro-batch's targets. Precondition: every micro-batch, on every rank, holds at
        least one supervised target (with none, the mean is 0/0). core's build_NTP_supervised_sample refuses a
        sample without one and the two standard batchers never hand out an empty micro-batch; a project with its own
        sample builder or batcher keeps this, and an objective that can mask every position (diffusion) defines its
        own loss for that case.
        """
        idx, kwargs = self.prepare_trunk_inputs(batch)
        hidden = self.trunk(idx, **kwargs)
        return self.head.loss(hidden, batch["targets"])

    def estimate_flops(self):
        """FLOPs/token for MFU: the trunk knows its own formula (architecture
        fact — core must not peek trunk internals); the head is a plain matmul
        (6 * params). Sum is numerically identical to the old system-level
        formula 6*(all_params - wte) + attention term.

        CONTRACT, and it is a real one: this returns ONE number, and the Trainer
        multiplies it by total_batch_size. That is only the truth if EVERY position
        counted in total_batch_size costs the same — i.e. every position goes through
        the trunk, attention is dense causal over sequence_len, and every position
        goes through the head.

        A System that breaks any of the three MUST override this. Two ways to break
        it, and the second does not shrink as the model grows:
          - the head runs on a SUBSET of positions (masked prediction, a sliced head)
            -> the head term is overcharged by the inverse of that fraction;
          - attention runs under a block/sparse mask -> the attention term is
            overcharged by the inverse of the mask density. This one gets WORSE with
            sequence length, because attention is a growing share of the total.
        A worked override is exemplars/nano_world_model's block-diffusion System,
        which breaks both. Left uncorrected there, MFU read 101.6% — above the card's
        physical ceiling, which is how the violation was noticed at all.

        Separately, and NOT something to correct: the trunk's attention term follows
        the PaLM/nanoGPT convention of charging the full sequence length, so it does
        not credit the causal half. That is ~3-5% here, it is shared by every dense
        causal model, and it is what makes these numbers comparable to published ones.
        See GPT.estimate_flops for the split between "shared constant, leave it" and
        "varies per model, correct it".
        """
        head_flops = 6 * sum(p.numel() for p in self.head.parameters())
        return self.trunk.estimate_flops() + head_flops


# ------------------------------------------------------------------------------------------------------------------
# The two ordinary autoregressive preparations. A project picks one
# explicitly (or writes its own); neither is chosen from what a batch happens to contain.

def prepare_unpacked_AR_trunk_inputs(batch):
    """At most one sample per row, starting at column 0, right padding allowed; ordinary causal attention (the
    trunk's causal SDPA path, no BlockMask) and the trunk's default in-row positions. A real query never reads the
    right padding under causal attention, so no padding mask is needed. Refuses a packed batch: several samples in a
    row would read each other. The check reads the host-side `layout_facts` collate leaves on the batch (no device
    sync); a batch without them was not made by core's collate (a hand-built batch, the text evaluation's raw-ruler
    bridge) and whoever built it vouches for its rows."""
    facts = batch.get("layout_facts")
    if facts is not None and not facts.get("one_sample_per_row_left_aligned"):
        raise ValueError("unpacked AR preparation needs one left-aligned sample per row; this batch is packed "
                         "(choose prepare_packed_AR_trunk_inputs)")
    return batch["idx"], {"token_types": batch.get("token_types")}


def prepare_packed_AR_trunk_inputs(batch, *, mask_builder=None):
    """Several independent samples per row, each ordinary causal AR inside itself: a BlockMask "same sample AND
    causal" built from this batch's doc_id (padding, doc_id -1, sees and is seen by nothing real), and doc_pos as
    position_ids so every sample starts at position 0. Also correct for a one-sample-per-row batch (use it to compare
    the SDPA and flex paths). `mask_builder` defaults to core.model.masks.default_mask_builder: a compiled
    create_block_mask on CUDA, torch's as it is on the CPU. Only building the mask is compiled here; compiling the
    trunk (and with it the flex call) stays the project's choice.

    doc_id and doc_pos are read as collate makes them and not validated here: a batch without either is a KeyError
    (not a fallback to in-row positions); doc_id is taken to be collate's int64 [B, T] on idx's device; the trunk
    checks position_ids (dtype, shape, device, and values against its rotary table) where it reads them
    (GPT.forward), which also covers a project's own preparation and a table installed later.

    For TRAINING the flex call must be compiled (compile_blocks or a whole-trunk compile): the trunk refuses an
    uncompiled flex call with gradient recording on (CausalSelfAttention.forward says why). Evaluation under
    torch.no_grad() may run it uncompiled."""
    idx, doc, pos = batch["idx"], batch["doc_id"], batch["doc_pos"]
    B, T = doc.shape
    build = mask_builder if mask_builder is not None else default_mask_builder(idx.device)
    block_mask = build(and_masks(same_sample(doc), causal), B=B, H=None, Q_LEN=T, KV_LEN=T, device=idx.device)
    return idx, {"token_types": batch.get("token_types"), "attention": block_mask, "position_ids": pos}

