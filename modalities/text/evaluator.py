"""
TextEvaluator — FineWeb validation (CE + bits-per-byte) as a text-modality
fact. Moved out of core: core keeps the Evaluator interface and
the LossEvaluator mechanism; what makes this TEXT is the FineWeb val stream +
the tokenizer's byte table, both of which live here.
"""

import torch.distributed as dist

from core.evaluation.evaluator import LossEvaluator, Evaluator

from modalities.text.fineweb import token_data_loader
from modalities.text.tokenizer import get_token_bytes


class TextEvaluator(Evaluator):
    """FineWeb text validation: CE loss + BPB."""

    def __init__(self, eval_config, device_batch_size, sequence_len):
        self.interval_steps = eval_config.get('interval_steps', 50)
        eval_at = eval_config.get('eval_at')            # optional explicit schedule
        self.eval_at = {int(s) for s in eval_at} if eval_at else None
        self.device_batch_size = device_batch_size
        self.sequence_len = sequence_len
        eval_tokens = eval_config.get('eval_tokens', 10485760)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        eval_steps = eval_tokens // (device_batch_size * sequence_len * world_size)

        self._eval = LossEvaluator(
            dataloader=None,  # created per evaluate() call
            eval_steps=eval_steps,
            mode='logits',
            token_bytes=get_token_bytes(device='cuda'),
            total_metric="val/text_ce",
            bpb_metric="val/bpb",
        )

    def evaluate(self, model, autocast_ctx):
        # Create fresh dataloader each eval (resets to start of val data)
        val_loader = token_data_loader(
            B=self.device_batch_size,
            T=self.sequence_len,
            split="val",
        )
        self._eval.dataloader = val_loader
        with autocast_ctx:
            return self._eval.evaluate(model, autocast_ctx)
