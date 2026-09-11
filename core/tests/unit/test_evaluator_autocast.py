"""LossEvaluator enters the autocast context the Trainer hands it (queue #15, public issue #4).

The Evaluator contract is "the Trainer passes its autocast context; the evaluator
enters it around whatever should run under the training precision regime". Every
project evaluator does that; core's own LossEvaluator did not, so a custom System
with fp32 modules was evaluated outside autocast when hooked to the Trainer directly.
A counting context object is the whole test: entered exactly once per evaluate().
"""

import torch

from core.evaluation.evaluator import LossEvaluator
from core.model.heads import LMHead


class _Counting:
    def __init__(self):
        self.entries = 0

    def __enter__(self):
        self.entries += 1

    def __exit__(self, *exc):
        return False


class _TinySystem(torch.nn.Module):
    """trunk + head, the two attributes the eval_loss services look for."""

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = LMHead(dim, vocab)

    def trunk(self, x, token_types=None):
        return self.emb(x)


def _batch(vocab=16):
    torch.manual_seed(0)
    return {"idx": torch.randint(0, vocab, (2, 5)), "targets": torch.randint(0, vocab, (2, 5))}


def test_logits_mode_enters_the_context_once():
    ctx = _Counting()
    ev = LossEvaluator([_batch()], eval_steps=1, mode="logits", total_metric="loss")
    ev.evaluate(_TinySystem(), ctx)
    assert ctx.entries == 1


def test_fused_mode_enters_the_context_once():
    ctx = _Counting()
    ev = LossEvaluator([_batch()], eval_steps=1, mode="fused", total_metric="loss")
    ev.evaluate(_TinySystem(), ctx)
    assert ctx.entries == 1


def test_none_context_is_accepted():
    ev = LossEvaluator([_batch()], eval_steps=1, mode="logits", total_metric="loss")
    assert "loss" in ev.evaluate(_TinySystem(), None)
