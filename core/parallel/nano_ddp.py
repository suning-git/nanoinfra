"""
nano_ddp.py — replicated data parallelism: per-bucket gradient all_reduce that
overlaps the backward pass. A deliberately minimal DDP; see "WHAT THE NANO MEANS"
below for exactly what it does not do.

Every rank holds a full copy of the model and a different slice of the batch. After a
step's backward, the copies must agree on one averaged gradient. Doing that as one
collective after backward finishes is correct but leaves the link idle throughout
backward and the compute idle throughout the collective. This module instead splits
the parameters into BUCKETS and reduces each one the moment its last gradient lands,
on a side stream, so most of the traffic happens while the rest of the backward is
still computing.

Use it when the model FITS on one device. When it does not, shard instead
(core.training.model_setup.build_system(parallel="fsdp")). Measured on 2xRTX 5090, 233M params:
replicated 106.2k tok/s / 19GB vs FSDP 93.6k tok/s / 30GB — at that size the memory
bottleneck is activations, not parameters, so sharding pays an all-gather per block
per step and buys nothing.

    ddp = NanoDDP([head_params] + block_buckets(trunk), module=system)
    for step in ...:
        opt.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            with sync_gradients(ddp, enabled=(micro == grad_accum - 1)):
                compute_loss(next(batches)).backward()
        opt.step()

Only the LAST micro-backward synchronizes. Reducing every micro-step would also be
correct — averaging partial sums then adding equals adding then averaging — but costs
grad_accum times the communication. `sync_gradients` is a no-op when ddp is None, so
one training loop serves one GPU and many. (FSDP spells this same idea
`set_requires_gradient_sync`; the information — which backward is the last one — is
only available to the training loop either way.)

WHY THE BRACKET IS A CONTEXT MANAGER, not two calls. What `all_reduce` writes into is
a TRANSIENT flat buffer, not `p.grad`; until the closing half copies it back, `p.grad`
still holds this rank's local gradient. Forgetting the close is therefore not "a
missed sync" — it is an optimizer step on unsynchronized gradients, with no error
anywhere. `reset()`/`finalize()` stay public for callers that need the halves apart,
and `reset()` refuses to open a backward whose predecessor was never closed.

WHY THE HOOKS ARE ACCEPTABLE. Firing from `register_post_accumulate_grad_hook` is
hidden state, which core/vision.md's "explicit over implicit" rightly dislikes: a bare
`loss.backward()` gives no sign that collectives are being issued. Two things make it
the right trade anyway. It is the only way to overlap at all — reducing after backward
returns is the explicit version, and it is the version that gives up the speedup. And
core already made this exact trade for FSDP, whose `fully_shard` likewise rewrites
modules and installs hooks. The mitigation is that the orchestrator constructs this
object explicitly and the training step brackets its own backward with a call named
`sync_gradients`, so the loop that reads as "one optimizer step" is also where the
reader learns that replicas exchange gradients.

WHAT THE "NANO" MEANS — what this deliberately does NOT do, against torch's
DistributedDataParallel. Read this before reaching for a feature and finding it absent.

  * It does not WRAP the model. You hand it buckets of parameters, not a module to
    replace. Consequence, and the reason for the choice: `state_dict()` keys are
    unchanged, so checkpoints stay interchangeable with single-device runs, where
    torch's DDP prefixes every key with `module.`.
  * It does not BROADCAST BUFFERS. Torch's DDP does, every forward, by default. This
    is a deliberate refusal, not an omission: broadcasting rank 0's buffer makes the
    replicas agree on a value computed from one rank's shard, which is agreement
    rather than correctness, and it hides the bug instead of surfacing it. Use
    `replica_divergence()` to find such buffers and fix them where they are updated.
  * It does not traverse the autograd graph to find unused parameters
    (`find_unused_parameters`). Bucket counters plus a canonical issue order cover the
    same ground more cheaply — see `_release()` and `strict`.
  * There is no communication-hook API, no gradient compression, no `join()` for
    uneven inputs (every rank must take the same number of steps), and no
    `static_graph` optimization.
  * `gradient_as_bucket_view` is not offered because it was measured and is worse
    here — see the memory note below.
  * Verified at WORLD SIZE 2 only. Nothing here is known to be wrong at larger world
    sizes, and nothing is known to be right either.

If you need any of the above, use `torch.nn.parallel.DistributedDataParallel`; it is
a fine choice and this module is not trying to replace it in general. What this one
buys is the two properties above it — untouched checkpoint keys, and a bucket
schedule tuned by measurement on this hardware — at roughly 5% less overhead and
~6 GB less peak, which is the difference between fitting a 2.8B model and not.

DESIGN NOTES — each one is a measurement, not a preference. All were taken on 2x RTX
5090 (PCIe, no NVLink) in 2026-07, on the models named beside each number. Treat them
as facts about that hardware and those models rather than as constants: on different
interconnect or at a different size the conclusions may hold and the figures will not.
(No path to the write-ups is given on purpose — core outlives the projects that
produced these, and a pointer into one of them would rot or ship broken.)

  * ONE bucket per transformer block, plus one for the root parameters. A dozen tiny
    per-parameter collectives would be launch-latency bound; one per-block collective
    saturates the link. Measured at 1.4B/seq8192: per-block comm 3.4 ms hides entirely
    inside the next block's 11.9 ms of backward, taking backward from 384 ms to 312 ms
    with the compute stream never stalling (inter-block gap 0.00 ms).

  * MEMORY COST, stated plainly: the buffers are allocated per fire and held in
    `_inflight` until `finalize()` copies them back, so from the end of backward until
    the bracket closes there is ONE EXTRA FULL COPY of the gradients live. Whether
    that moves the STEP's peak depends on the activation profile — on a model whose
    peak occurs mid-backward with activations resident, it does not (measured on
    1.4B/seq8192, where the peak did not move); on a small model whose peak is at
    end-of-backward it does, roughly one gradient copy (measured: +384 MiB on a
    392 MiB model). Do not assume it is free.

    Buffers are nonetheless transient rather than persistent, and that choice IS
    measured: a persistent buffer, or `gradient_as_bucket_view`-style gradient slices,
    cost a further +2.81 GB peak at 1.4B for no speedup, because they are held across
    forward too where nothing can reuse them. Both variants were checked (view mode and
    persistent-copy mode) and both cost the same, so the cost is the persistence rather
    than the views.

  * `ReduceOp.AVG` averages inside NCCL, so there is no separate division pass over
    the gradients — and no need for this class to know the world size at all.

  * WHAT "WORKS WITH MIXTURE-OF-EXPERTS" MEANS HERE, stated precisely because the
    phrase invites a much bigger claim than the one being made.

    Every rank holds a FULL copy of every expert. What is split across ranks is the
    BATCH: a rank routes its own tokens only to its own local experts, and every
    expert's gradient is all_reduced like any other parameter. The experts are
    REPLICATED, not distributed. What it took to make that correct is the issue-order
    guarantee in `_release()` plus `strict=False`, because per-rank routing makes the
    SET of parameters that receive a gradient differ from rank to rank.

    This is NOT expert parallelism, which splits the experts themselves — rank 0 owns
    experts 0..15, rank 1 owns 16..31 — and ships each token to the rank that owns its
    expert, two all-to-alls per MoE layer. There an expert's weights live on exactly
    one rank and its gradient needs no reduction at all. That changes the model's
    FORWARD, which this module does not touch.

    The distinction has a measured price. Communication scales with TOTAL parameters
    while compute scales only with ACTIVE ones: on a 451.8M MoE trunk, 904 MB of bf16
    gradients per rank per step, 35% of the step, most of it spent on experts that took
    no part in that step's forward. That is the structural cost of replicating a
    sparsely-activated model, and removing it is precisely what expert parallelism is
    for.

  * Communication as a share of a step depends on tokens per step, not on model size,
    because both scale with parameters: `comm/compute ~= 2087 / (tokens per step per
    rank)` on a dense trunk. A long-window video model at 34816 tokens/step spends 6%
    of the step communicating, so overlap buys little; a text model at 4096-8192 spends
    25-51%, where it buys a lot. Mixture-of-experts breaks the rule in the other
    direction — communication follows TOTAL parameters while compute follows only the
    ACTIVE ones (measured on a 451.8M MoE: 35% of the step) — so measure, do not
    extrapolate.

  * Under `torch.compile`, whether buckets can overlap at all depends on how many
    graphs the trunk compiles to. A dense trunk compiled whole becomes ONE
    AOTAutograd graph, and every gradient is finalized at the very end of backward
    (pytorch#109774), leaving nothing to overlap — compile per block instead
    (core.training.model_setup.compile_blocks). A trunk that already graph-breaks has the seams
    anyway: nano-dsv4 compiles to 20 graphs / 19 breaks and shows no difference
    between the two (78.0 vs 78.2 ms/step on two ranks). Check with
    `torch._dynamo.explain(trunk.forward)` rather than assuming either way.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors as _flatten
from torch._utils import _unflatten_dense_tensors as _unflatten


def block_buckets(trunk):
    """Default bucketing for a GPT-style trunk: one bucket per block, plus one for
    everything else (embeddings and so on).

    ORDER IS PART OF THE CONTRACT. Buckets are issued in list order, so the list must
    be in the order the gradients are expected to COMPLETE, or early buckets sit
    waiting for late ones and the overlap is lost. Backward runs from the output
    inwards, so that is the LAST block first and the root last — the order returned
    here. A caller adding its own buckets must place them accordingly; an LM head
    completes before any block and belongs at the FRONT:

        NanoDDP([head_params] + block_buckets(trunk), module=system)

    The root bucket carries the embedding gradient, which backward finalizes last and
    so cannot be overlapped by anything — that residual (~12 ms at 1.4B) is structural,
    not an implementation flaw.
    """
    blocks = list(trunk.blocks)
    buckets = [[p for p in b.parameters() if p.requires_grad] for b in reversed(blocks)]
    in_block = {id(p) for bucket in buckets for p in bucket}
    root = [p for p in trunk.parameters() if p.requires_grad and id(p) not in in_block]
    if root:
        buckets.append(root)
    return buckets


@contextmanager
def sync_gradients(ddp, enabled=True):
    """Bracket a backward that should (or should not) synchronize gradients.

    A no-op when `ddp` is None, so a training loop written once runs on one GPU and on
    many without branching. THE NAME IS THE POINT: this line is where a reader learns
    that gradients cross the network here, which the bare `loss.backward()` underneath
    it cannot say for itself.

    Args:
        ddp: a NanoDDP, or None for single-device training.
        enabled: whether THIS backward is the one that synchronizes. False accumulates
            locally (see the module docstring on gradient accumulation).
    """
    if ddp is None:
        yield
        return
    with ddp.sync_on(enabled):
        yield


@torch.no_grad()
def replica_divergence(module, group=None, include_buffers=True):
    """Which tensors differ between ranks? Returns {name: spread across ranks}, empty
    when the replicas agree.

    A BRING-UP TOOL, not something to call in a training loop: it issues two
    collectives per tensor. It is the cheapest way to turn a silent divergence into a
    visible one when a new model is first put on NanoDDP.

    NanoDDP synchronizes GRADIENTS, and nothing else. Parameters stay in step because
    every rank starts from the same seed and applies the same averaged gradient — but
    anything a model updates OUTSIDE the gradient path is on its own, and BUFFERS are
    the usual case: a load-balancing counter, a running mean, a moving threshold. Each
    rank computes those from its own shard of the batch, so the replicas quietly drift
    and the model stops being equivalent to single-device training on the same global
    batch. This is not hypothetical — a mixture-of-experts router's bias buffer in this
    repository diverged on every router from step 0, while the parameters stayed
    bitwise identical.

    Deliberately NOT auto-fixed by broadcasting rank 0's buffers, which is torch DDP's
    default. Broadcasting makes the replicas agree on a value computed from ONE rank's
    shard: agreement, not correctness. The real fix is to move the update somewhere it
    can see the whole step — reduce the inputs, then update once per optimizer step.
    Broadcasting would conceal exactly the bug this function exists to surface.
    """
    named = list(module.named_parameters())
    if include_buffers:
        named += list(module.named_buffers())
    diverged = {}
    for name, tensor in sorted(named):
        if not tensor.is_floating_point():
            continue
        lo = tensor.detach().float().clone()
        hi = lo.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
        spread = (hi - lo).max().item()
        # `not (spread <= 0)` rather than `spread > 0` so a NaN — which makes every
        # comparison False and would otherwise be reported as agreement — is caught.
        if not (spread <= 0):
            diverged[name] = spread
    return diverged


class NanoDDP:
    """Gradient synchronization for replicated data parallelism.

    Owns no parameters and wraps no module: it attaches a hook to each parameter it is
    given and reduces them bucket by bucket during backward. Because it does not wrap
    the model, `state_dict()` keys are unchanged and checkpoints stay compatible with
    single-device runs (torch's DistributedDataParallel prefixes them with `module.`).

    Args:
        buckets: list of lists of parameters. **Issued in list order** — see
            `block_buckets` for why that order matters and how to choose it.
        module: the module these buckets are supposed to cover, checked exactly.
            Required, and keyword-only, because the failure it prevents — a trainable
            parameter in no bucket, whose gradient is then never synchronized — is
            silent, and only the caller knows the whole model. Pass the smallest
            module that contains exactly the bucketed parameters. NOTE the check is a
            snapshot: a parameter unfrozen AFTER construction is in no bucket and
            nothing will say so.
        strict: what to do when a bucket does not receive all its gradients.
            True (default) raises on every rank, naming the buckets. False zero-fills
            and reduces them anyway, which is what mixture-of-experts routing needs
            and is exact: a parameter that got no gradient contributed exactly zero to
            this rank's loss, so averaging that zero with the other ranks' real
            gradients yields the true gradient of the globally-averaged loss.
            NOTE one behavioural difference of strict=False: a parameter that receives
            no gradient on ANY rank ends the step with `.grad` set to zeros rather than
            left as None, and AdamW skips None but still applies weight decay to a zero
            gradient. Such a parameter therefore decays under this path and would not
            on a single device. It is the price of reducing the bucket at all, since
            whether some other rank had a real gradient is not knowable locally.
        stream: CUDA stream for the collectives. Defaults to a private one; pass your
            own only if you are coordinating with other side-stream work.
        group: process group. Defaults to the world.

    Everything after `buckets` is keyword-only so that inserting an argument can never
    silently re-bind an existing caller's positional.

    Call `close()` when done. A NanoDDP is reachable from its own hooks, so it is never
    garbage collected, and a forgotten one keeps reducing alongside its replacement.
    """

    # Issuing buckets in completion order instead of list order is silently wrong
    # whenever ranks have differently-shaped backwards (see _release). This attribute
    # exists ONLY so the gate that catches that bug can prove it still catches it:
    # core/tests/integration/test_collective_order.py sets it to False for its positive control. It is not
    # a constructor argument, because nothing in production should ever set it.
    canonical_order = True

    _OWNER = "_nano_ddp_owner"               # marker left on parameters we hook

    def __init__(self, buckets, *, module, strict=True, stream=None, group=None):
        self.buckets = [list(b) for b in buckets]
        self.strict = strict
        self.group = group
        self.comm_stream = stream or torch.cuda.Stream()
        self.sync = True
        self._inflight = []                  # (flat buffer, bucket index) awaiting copy-back
        self._handles = []
        self._validate(module)
        try:
            for bucket_index, bucket in enumerate(self.buckets):
                for param in bucket:
                    self._handles.append(
                        param.register_post_accumulate_grad_hook(self._hook(bucket_index)))
                    setattr(param, self._OWNER, self)
        except Exception:
            # Half-registered hooks would fire into a half-built object and break every
            # later backward on this model, with no way to undo it.
            self.close()
            raise
        self._arm()

    def close(self):
        """Detach every hook and release any in-flight buffers.

        Not optional bookkeeping: a NanoDDP is reachable from the hooks it installed,
        so it is never garbage collected, and __del__ would never run. Dropping the
        reference is NOT enough — the old object keeps counting every backward.
        """
        # The buffers in _inflight may still be being written by NCCL on the comm
        # stream. Returning them to the caching allocator without joining first would
        # let the compute stream hand that memory to something else mid-collective;
        # this torch build defaults to TORCH_NCCL_AVOID_RECORD_STREAMS, so nothing
        # underneath protects them. Reachable via the no-try/finally path in sync_on(),
        # where close() is the recovery from an exception mid-backward.
        if self._inflight and torch.cuda.is_available():
            torch.cuda.current_stream().wait_stream(self.comm_stream)
            torch.cuda.current_stream().synchronize()
        for handle in self._handles:
            handle.remove()
        for bucket in self.buckets:
            for param in bucket:
                if getattr(param, self._OWNER, None) is self:
                    delattr(param, self._OWNER)
        self._handles.clear()
        self._inflight.clear()
        self.sync = False

    def _validate(self, module):
        """Reject bucket lists that would fail quietly rather than loudly."""
        seen = {}
        for bucket in self.buckets:
            for param in bucket:
                owner = getattr(param, self._OWNER, None)
                if owner is not None:
                    # Without this the symptom appears a step later as "received
                    # gradients twice inside one bracket", which points the reader at
                    # their own backward() instead of at the NanoDDP nobody closed.
                    raise RuntimeError(
                        "NanoDDP: these parameters already belong to a NanoDDP that "
                        "was never close()d. Both would count every backward and both "
                        "would reduce, doubling the traffic and then failing with a "
                        "misleading error. Call close() on the old one first.")
        for bucket_index, bucket in enumerate(self.buckets):
            if not bucket:
                raise ValueError(
                    f"NanoDDP: bucket {bucket_index} is empty. An empty bucket can "
                    f"never complete, so it would stall every bucket behind it until "
                    f"finalize(). Drop it instead.")
            dtypes = {p.dtype for p in bucket}
            if len(dtypes) > 1:
                # _flatten_dense_tensors does not reject mixed dtypes; it promotes to
                # the widest one. The result is numerically fine but silently doubles
                # the bytes on the wire for every bf16 parameter in the bucket, which
                # is the sort of thing nobody notices for months.
                raise ValueError(
                    f"NanoDDP: bucket {bucket_index} mixes dtypes {sorted(map(str, dtypes))}. "
                    f"Flattening would promote them all to the widest, silently "
                    f"inflating this bucket's traffic. Split it by dtype.")
            for param in bucket:
                if not param.requires_grad:
                    raise ValueError(
                        f"NanoDDP: bucket {bucket_index} holds a parameter with "
                        f"requires_grad=False. It can never receive a gradient, so the "
                        f"bucket would never complete. Filter frozen parameters out "
                        f"(block_buckets already does).")
                if id(param) in seen:
                    raise ValueError(
                        f"NanoDDP: a parameter appears in both bucket {seen[id(param)]} "
                        f"and bucket {bucket_index}; its gradient would be reduced twice. "
                        f"Tied weights (a head sharing the embedding, a shared expert) "
                        f"hit this — put the shared parameter in exactly one bucket.")
                seen[id(param)] = bucket_index

        # THE hole this class exists to close: a parameter that is in no bucket is
        # never reduced, and nothing anywhere says so. Only the caller knows the whole
        # model, so coverage can only be checked if the caller hands it over — which
        # is why `module` is required rather than optional, and why passing None
        # explicitly is refused instead of quietly skipping the check.
        if module is None:
            raise ValueError(
                "NanoDDP: module=None would skip the coverage check, which is the one "
                "thing standing between a mis-bucketed model and gradients that are "
                "never synchronized with nothing to say so. Pass the smallest module "
                "containing exactly the bucketed parameters.")
        want = {id(p): n for n, p in module.named_parameters() if p.requires_grad}
        missing = [n for i, n in want.items() if i not in seen]
        extra = len(seen) - (len(want) - len(missing))
        if missing or extra:
            raise ValueError(
                f"NanoDDP: buckets do not cover the module. "
                f"{len(missing)} trainable parameter(s) are in no bucket "
                f"(e.g. {missing[:3]}), and {extra} bucketed parameter(s) are not "
                f"in the module. Unbucketed gradients are NEVER synchronized and "
                f"nothing would report it.")

    def _arm(self):
        """Per-backward state: how many gradients each bucket still awaits, which
        buckets are ready, and how far down the issue order we have got."""
        self._pending = [len(b) for b in self.buckets]
        self._ready = [False] * len(self.buckets)
        self._next = 0

    # --- the bracket ---------------------------------------------------------

    @contextmanager
    def sync_on(self, enabled=True):
        """Bracket ONE backward: arm the counters, then close out afterwards.

        Deliberately no try/finally. If the body raises, `finalize()` is skipped and
        the original exception propagates unmasked — a secondary failure inside
        finalize() would otherwise replace the real error. The step is dead either way,
        and the next `reset()` says loudly that a backward was never closed.
        """
        self.reset(sync=enabled)
        yield
        if enabled:
            self.finalize()

    def reset(self, sync=True):
        """Open a backward. Prefer `sync_gradients(ddp, enabled)`, which pairs this
        with `finalize()` so the two cannot come apart; this is the raw half.

        sync=False makes the hooks inert for that backward — gradient accumulation,
        where only the final micro-backward reduces.
        """
        if self._inflight:
            raise RuntimeError(
                f"NanoDDP.reset(): {len(self._inflight)} reduced bucket(s) from the "
                f"previous backward were never written back into .grad. Either "
                f"finalize() was not called, or it raised and this object was reused. "
                f"Those gradients are still the LOCAL ones, so continuing would train "
                f"unsynchronized replicas with no error anywhere. Use "
                f"`with sync_gradients(ddp, ...)` so the pair cannot come apart.")
        self._arm()
        self.sync = sync

    def finalize(self):
        """Close the backward. Must run before gradient clipping or the optimizer step,
        both of which read `.grad`.

        The ORDER below is the subtle part — it is what an earlier version of this
        function got wrong — so it is stated rather than left to be inferred:
        issue whatever buckets are still waiting, in order; then the strict flag's
        collective, which must come after them; then join the comm stream; then raise
        if any rank reported an incomplete bucket; then copy the reduced gradients back
        into `.grad`; then re-arm, cold.
        """
        if not self.sync:
            raise RuntimeError(
                "NanoDDP.finalize() after reset(sync=False) — the last micro-backward "
                "of a step must run with sync=True, or this step's gradients never "
                "synchronize.")
        incomplete = self.missing()

        # Drain the tail IN ORDER: buckets that never completed are zero-filled by
        # _fire, and buckets that completed early but were waiting their turn go out
        # here. Every rank issues 0..n-1 exactly once, whatever its routing did.
        if self.canonical_order:
            while self._next < len(self.buckets):
                self._fire(self._next)
                self._next += 1
        else:
            for bucket_index, _ in incomplete:
                self._fire(bucket_index)

        # The strict check must be COLLECTIVE: whether a bucket completed is a per-rank
        # fact — that is the whole point of the guard — so a purely local raise lets the
        # detecting rank exit while the others issue their drain alone and die minutes
        # later on an NCCL watchdog timeout, burying the useful message.
        #
        # It must come AFTER the drain. Collectives match by POSITION, and the number
        # issued during backward varies per rank (evening that out is exactly what the
        # drain is for), so a flag reduced before the drain would sit at a different
        # index on different ranks — reintroducing, inside the guard, the very bug the
        # guard reports. After the drain every rank has issued the same n buckets, so
        # this is position n everywhere.
        #
        # COST, measured rather than assumed: strict=True adds one small collective
        # per step. Interleaved A/B medians, two independent measurements on this box:
        # +0.29 ms on a 0.8 ms step, +0.45 ms on a 4.3 ms step, +0.04 ms on a 12 ms
        # step. Small everywhere, and it does NOT grow with the model — but neither
        # measurement supports calling it a scale-independent constant either, so
        # measure on your own setup before it matters to you. Part of it is absorbed by
        # the `clip_grad_norm_(...).item()` the caller runs next anyway. strict=False
        # removes the check and its cost. (Deferring the flag read a step to dodge the
        # CPU stall was tried and measured at 0.07 ms — not worth the deferred-error
        # machinery, so both the read and the raise are immediate.)
        flag = None
        if self.strict:
            flag = torch.tensor([len(incomplete)], dtype=torch.int32,
                                device=self.buckets[0][0].device)
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)

        torch.cuda.current_stream().wait_stream(self.comm_stream)
        if flag is not None and flag.item() > 0:
            raise RuntimeError(
                f"NanoDDP: some rank had bucket(s) that never completed — parameters "
                f"that received no gradient this step. This rank: "
                f"{incomplete or 'none (another rank detected it)'}. Such gradients are "
                f"NOT synchronized, so the replicas silently diverge. Fix the model, or "
                f"pass strict=False to zero-fill and reduce them (which "
                f"mixture-of-experts routing needs).")
        for flat, bucket_index in self._inflight:
            bucket = self.buckets[bucket_index]
            # _unflatten only reads shape and dtype, so the parameters themselves
            # serve as the template — no need to materialize zeros for missing grads.
            for param, reduced in zip(bucket, _unflatten(flat, bucket)):
                if param.grad is None:
                    param.grad = reduced.clone()   # `flat` is about to be released
                else:
                    param.grad.copy_(reduced)
        self._inflight.clear()
        # Re-arm so the object is ready for the next backward — without this, a
        # backward following finalize() without an intervening reset() would count down
        # from zero, fire nothing and reduce nothing, silently. Re-armed COLD
        # (sync=False): a stray backward outside any bracket then does nothing, rather
        # than issuing a step's worth of collectives that the other ranks never match.
        self._arm()
        self.sync = False

    def missing(self):
        """[(bucket index, #parameters that received no gradient)] for every bucket
        still incomplete.

        Keyed on the counter rather than on "was it issued": under canonical ordering a
        bucket can be complete and still waiting its turn, which is not missing.
        """
        return [(i, n) for i, n in enumerate(self._pending) if n != 0]

    # --- the machinery -------------------------------------------------------

    def _hook(self, bucket_index):
        def hook(_param):
            if not self.sync:
                return
            if self._pending[bucket_index] == 0:
                # This bucket already completed inside this bracket, so a SECOND
                # backward ran without a reset. Re-issuing the bucket would break the
                # canonical order; not re-issuing it means finalize() copies back the
                # first backward's values and silently discards this one. Neither is
                # acceptable, so say so.
                raise RuntimeError(
                    f"NanoDDP: bucket {bucket_index} received gradients twice inside "
                    f"one sync_gradients(...) bracket — a second backward() ran before "
                    f"the bracket closed. Give each backward its own bracket; a "
                    f"multi-term objective should sum its terms and call backward once.")
            self._pending[bucket_index] -= 1
            if self._pending[bucket_index] == 0:
                self._release(bucket_index)
        return hook

    def _release(self, bucket_index):
        """A bucket's gradients are all in. Issue it — but only when its turn comes.

        NCCL matches collectives BY ISSUE ORDER: the n-th collective on one rank pairs
        with the n-th on every other. Issuing in completion order is fine while every
        rank's backward has the same shape, and silently wrong the moment it does not.
        With per-expert routing, a block whose experts are all active completes during
        backward on one rank and is deferred to finalize() on another; the two ranks
        then issue the same SET of buckets in a different ORDER, the shapes still
        match, NCCL does not complain, and one block's gradients are averaged against
        another's. Measured on the gate before this existed: 4 of 10 tensors wrong,
        with a relative error of order 1 (the gate uses random data, so the exact
        figure moves run to run — what is stable is that the gradients are not merely
        imprecise, they are the wrong tensors).

        So a bucket is issued only once every earlier bucket has been. Buckets that
        finish early wait; finalize() drains the tail in the same order. Both ranks
        therefore issue 0, 1, ... n-1 whatever their routing did — the order is
        structural rather than emergent. On a dense model this costs nothing, because
        completion order already IS list order (see block_buckets).
        """
        if not self.canonical_order:
            self._fire(bucket_index)                  # pre-fix behaviour; test knob only
            return
        self._ready[bucket_index] = True
        while self._next < len(self.buckets) and self._ready[self._next]:
            self._fire(self._next)
            self._next += 1

    def _fire(self, bucket_index):
        """Start this bucket's all_reduce on the side stream and keep the buffer until
        finalize() copies the result back."""
        bucket = self.buckets[bucket_index]
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in bucket]
        flat = _flatten(grads)                   # transient: reuses freed activations
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=self.group)
        self._inflight.append((flat, bucket_index))
