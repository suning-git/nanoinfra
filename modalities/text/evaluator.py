"""
TextEvaluator — FineWeb validation (CE + bits-per-byte) as a text-modality
fact. Core keeps the Evaluator interface and the LossEvaluator mechanism; what
makes this TEXT is the val stream + the tokenizer's byte table.

THE BUG THIS FILE USED TO CARRY
-------------------------------
It built its batches by calling ``token_data_loader(split="val")`` directly —
the RAW packing path — while training went through ``TextDataSource``, which
assembles each document as ``bos + text_start + … + text_end + eos``. So every
validation number was measured on a sequence format the model had never been
trained on: train batches carried 0.83% control tokens (bos/text_start/text_end),
val batches 0.098% (bos only). Nothing errored and nothing was logged, and the
train→val gap it produced varied by architecture (0.002 nat for a sliding-window
model, 0.098 for a dense one, 0.483 for one whose indexer could reach the
document-structure tokens freely) — i.e. it did NOT cancel out of comparisons.

Now an eval stream is built from the same declaration as a training source
(dataset + split + recipe) and assembled by the same code (``TextDataSource``
through ``MixedDataLoader``). A stream may opt out with ``recipe: null``, which
reproduces the old raw packing — but opting out has to be WRITTEN DOWN, which is
the whole difference from before: the old behaviour was the invisible default.

Nothing shipped here declares ``recipe: null``. Its one current user is a
project-side ``--raw-ruler`` migration bridge that re-measures the old ruler so
historical curves can be reconnected — a migration artifact owned by the project
doing the migrating. (Deliberately not naming that project: a modality that points
at a project goes stale the moment the project is renamed.)
"""

import torch.distributed as dist

from core.evaluation.evaluator import LossEvaluator, Evaluator

from modalities.text.fineweb import token_data_loader
from modalities.text.tokenizer import get_token_bytes


class TextEvaluator(Evaluator):
    """One declared evaluation stream -> {metric: CE, bpb_metric: BPB}.

    Args:
        stream: resolved stream declaration —
            ``{name, files, recipe (dict|None), recipe_name, metric, bpb_metric}``
        eval_config: cadence + budget (``interval_steps`` / ``eval_at`` / ``eval_tokens``)
        device_batch_size, sequence_len: batch geometry (matches training)
        loader_factory: ``callable(files, recipe, recipe_name, B, T) -> fresh iterator``.
            Supplied by the orchestrator so assembly lives in ONE place; only the
            ``recipe: null`` bridge bypasses it.
    """

    def __init__(self, stream, eval_config, device_batch_size, sequence_len,
                 loader_factory=None):
        self.name = stream.get('name', 'text')
        self.files = stream['files']
        self.recipe = stream.get('recipe')          # dict, or None = raw bridge
        self.recipe_name = stream.get('recipe_name',
                                      'raw' if self.recipe is None else '?')
        self.loader_factory = loader_factory
        if self.recipe is not None and loader_factory is None:
            raise ValueError(
                f"eval stream {self.name!r} declares recipe {self.recipe_name!r} but no "
                f"loader_factory was supplied — a recipe stream must be assembled by the "
                f"same path as training."
            )

        self.interval_steps = eval_config.get('interval_steps', 50)
        eval_at = eval_config.get('eval_at')        # optional explicit schedule
        self.eval_at = {int(s) for s in eval_at} if eval_at else None
        self.device_batch_size = device_batch_size
        self.sequence_len = sequence_len

        eval_tokens = eval_config.get('eval_tokens', 10485760)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        eval_steps = eval_tokens // (device_batch_size * sequence_len * world_size)

        self.metric = stream.get('metric', f'val/{self.name}_ce')
        self._eval = LossEvaluator(
            dataloader=None,                        # created per evaluate() call
            eval_steps=max(1, eval_steps),
            mode='logits',
            token_bytes=get_token_bytes(device='cuda'),
            total_metric=self.metric,
            bpb_metric=stream.get('bpb_metric'),
        )

    def describe(self) -> str:
        """One line for the startup log, so a run self-describes its ruler."""
        names = [f.rsplit('/', 1)[-1] for f in self.files]
        return (f"  eval stream [{self.name}] recipe={self.recipe_name} "
                f"metric={self.metric} files={names}")

    def evaluate(self, model, autocast_ctx):
        # Fresh loader each call -> every evaluation reads the same val data from
        # the same starting point, so successive points on a curve differ by the
        # model only, not by which batches happened to come up.
        if self.recipe is None:
            val_loader = token_data_loader(
                B=self.device_batch_size, T=self.sequence_len, files=self.files,
            )
        else:
            val_loader = self.loader_factory(
                files=self.files, recipe=self.recipe, recipe_name=self.recipe_name,
                B=self.device_batch_size, T=self.sequence_len,
            )
        self._eval.dataloader = val_loader
        with autocast_ctx:
            return self._eval.evaluate(model, autocast_ctx)
