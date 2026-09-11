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
The compiled trunk is held OUTSIDE the module registry (object.__setattr__), because
torch.compile wraps the SAME parameters as the registered trunk; registering it too
would double-count them in state_dict / parameters(). It is registered by the
orchestrator after assembly (core/training/model_setup.compile_system_trunk), never
by build_system.

Projects that want a different composition (e.g. a diffusion head, or multiple heads)
write their own System satisfying the same `loss(batch)` contract — core does not
change (examples over interfaces).
"""

import torch.nn as nn


class LMSystem(nn.Module):
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk
        self.head = head
        # held off the module registry (see module docstring); starts uncompiled.
        object.__setattr__(self, "_compiled_trunk", None)

    def set_compiled_trunk(self, compiled_trunk):
        """Register a torch.compile'd view of the trunk for the hot training path
        (`_run_trunk`); None puts the eager trunk back.

        Not stored as a submodule: it shares parameters with the already-registered
        raw trunk, so registering it would double-count params in state_dict().
        Refuses to overwrite a registered wrapper with another one: that is either
        wrapping the wrapper or two decision sites both compiling. Clear with None
        first when a switch is really meant (frontier_arch/scripts/nan_repro.py does).
        """
        if compiled_trunk is not None and self._compiled_trunk is not None:
            raise RuntimeError(
                "set_compiled_trunk: a compiled trunk is already registered; compile "
                "it once. Pass None first to deliberately replace it.")
        object.__setattr__(self, "_compiled_trunk", compiled_trunk)

    @property
    def config(self):
        """The trunk's GPTConfig — the architectural facts of this System.

        Checkpoint self-description hangs off this: save_checkpoint_dcp records
        asdict(model.config) into meta.json, and _validate_model_config audits
        every load against it (both look the attribute up via `model.config`).
        """
        return self.trunk.config

    def _run_trunk(self, idx, token_types=None):
        trunk = self._compiled_trunk if self._compiled_trunk is not None else self.trunk
        return trunk(idx, token_types=token_types)

    def loss(self, batch):
        """Scalar training loss — the Trainer's entire view of the model world."""
        hidden = self._run_trunk(batch["idx"], token_types=batch.get("token_types"))
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
