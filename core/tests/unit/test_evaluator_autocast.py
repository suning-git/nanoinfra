"""LossEvaluator enters the autocast context the Trainer hands it (public issue #4), once, with gradients off.

The Evaluator contract is "the Trainer passes its autocast context; the evaluator
enters it around whatever should run under the training precision regime". Every
project evaluator does that; core's own LossEvaluator did not, so a custom System
with fp32 modules was evaluated outside autocast when hooked to the Trainer directly.
A counting context object is the whole test: entered exactly once per evaluate().
The Trainer only switches the System to eval(), so gradient recording is the
evaluator's to turn off (evaluate_loss does), and to restore afterwards.
"""

import torch

from core.evaluation.evaluator import LossEvaluator
from core.evaluation.metrics import mean_ce
from core.model.heads import LMHead
from core.model.system import prepare_unpacked_AR_trunk_inputs


class _Counting:
    def __init__(self):
        self.entries = 0

    def __enter__(self):
        self.entries += 1

    def __exit__(self, *exc):
        return False


class _TinySystem(torch.nn.Module):
    """trunk + head + prepare_trunk_inputs, what evaluate_loss looks for; records grad mode inside the trunk."""
    prepare_trunk_inputs = staticmethod(prepare_unpacked_AR_trunk_inputs)

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = LMHead(dim, vocab)
        self.grad_seen = []

    def trunk(self, x, token_types=None):
        self.grad_seen.append(torch.is_grad_enabled())
        return self.emb(x)


def _batch(vocab=16):
    torch.manual_seed(0)
    return {"idx": torch.randint(0, vocab, (2, 5)), "targets": torch.randint(0, vocab, (2, 5))}


def test_enters_the_context_once_with_gradients_off():
    ctx, system = _Counting(), _TinySystem()
    LossEvaluator([_batch()], eval_steps=1, metrics={"loss": mean_ce()}).evaluate(system, ctx)
    assert ctx.entries == 1
    assert system.grad_seen == [False]
    assert torch.is_grad_enabled()                     # restored afterwards


def test_a_real_autocast_context_is_left_as_it_was():
    system = _TinySystem()
    ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    LossEvaluator([_batch()], eval_steps=1, metrics={"loss": mean_ce()}).evaluate(system, ctx)
    assert not torch.is_autocast_enabled("cpu")
