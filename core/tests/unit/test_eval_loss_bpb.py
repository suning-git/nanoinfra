"""bpb from the fused evaluation path equals the logits path (queue #16, public issue #6).

bpb = sum(CE nats over BYTE-BEARING targets) / (ln2 * sum bytes). Zero-byte control
tokens contribute neither nats nor bytes. The fused path used to take the mean loss
over ALL valid targets (control tokens included) and charge it to the byte-bearing
ones — right when every target carries bytes, wrong the moment a control token is
supervised, which the text recipe (`supervise: all`) does at every row boundary.

The batch below has all four ingredients the issue lists: nonuniform per-token loss
(random head weights), zero-byte control targets, IGNORE targets, and a byte table.
The expected value is computed by hand from the logits, independently of both paths.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from core.evaluation.eval_loss import evaluate_loss_fused, evaluate_loss_logits
from core.model.heads import LIGER_AVAILABLE, LigerLMHead, LMHead
from core.tokenization.vocab_layout import VocabLayout

IGNORE = VocabLayout.IGNORE_INDEX
V, H = 16, 8
CONTROL = [12, 13, 14, 15]            # zero-byte ids, like bos / text_start / text_end / eos


class _TinySystem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.head = LMHead(H, V)
        with torch.no_grad():
            self.head.lm_head.weight.normal_(0, 1.0)     # nonuniform per-token losses

    def trunk(self, x, token_types=None):
        return self.emb(x)


def _table():
    torch.manual_seed(2)
    tb = torch.randint(1, 4, (V,), dtype=torch.int64)
    tb[CONTROL] = 0
    return tb


def _batch():
    torch.manual_seed(1)
    idx = torch.randint(0, V, (2, 6))
    targets = torch.randint(0, 12, (2, 6))               # byte-bearing by default
    targets[0, 0] = CONTROL[0]                           # supervised control targets
    targets[0, 5] = CONTROL[3]
    targets[1, 2] = CONTROL[1]
    targets[1, 4] = IGNORE                               # an ignored position
    return {"idx": idx, "targets": targets}


def _by_hand(system, batch, tb):
    logits = system.head(system.trunk(batch["idx"]))
    y = batch["targets"].reshape(-1)
    valid = y >= 0
    per = F.cross_entropy(logits.reshape(-1, V), y, ignore_index=IGNORE, reduction="none")
    nbytes = torch.where(valid, tb[y.clamp(min=0)], torch.zeros_like(y))
    byte_bearing = valid & (nbytes > 0)
    return per[byte_bearing].sum().item() / (math.log(2) * nbytes[byte_bearing].sum().item())


def test_fused_bpb_equals_logits_bpb_and_the_hand_value():
    system, tb, batch = _TinySystem(), _table(), _batch()
    expected = _by_hand(system, batch, tb)
    assert 0 < expected < 20
    with torch.no_grad():
        got_logits = evaluate_loss_logits(system, [batch], 1, token_bytes=tb)["bpb"]
        got_fused = evaluate_loss_fused(system, [batch], 1, token_bytes=tb)["bpb"]
    torch.testing.assert_close(torch.tensor(got_logits), torch.tensor(expected), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.tensor(got_fused), torch.tensor(expected), rtol=1e-5, atol=1e-6)


def test_the_two_token_counterexample_from_the_issue():
    """One high-NLL zero-byte control target + one low-NLL two-byte text target.

    bpb must be the text token's nats over its two bytes only. The old fused path
    averaged the control token's high loss in and charged it to the text token.
    """
    system, batch = _TinySystem(), {"idx": torch.tensor([[3, 7]]), "targets": torch.tensor([[CONTROL[0], 5]])}
    tb = torch.zeros(V, dtype=torch.int64)
    tb[5] = 2
    with torch.no_grad():
        logits = system.head(system.trunk(batch["idx"]))
        per = F.cross_entropy(logits.reshape(-1, V), batch["targets"].reshape(-1), reduction="none")
        expected = per[1].item() / (math.log(2) * 2)
        got_fused = evaluate_loss_fused(system, [batch], 1, token_bytes=tb)["bpb"]
        got_logits = evaluate_loss_logits(system, [batch], 1, token_bytes=tb)["bpb"]
    assert abs(got_logits - expected) < 1e-6
    assert abs(got_fused - expected) < 1e-6


def test_total_loss_still_counts_every_valid_target():
    """The fix touches only bpb: total_loss keeps averaging over ALL valid targets."""
    system, tb, batch = _TinySystem(), _table(), _batch()
    with torch.no_grad():
        a = evaluate_loss_fused(system, [batch], 1, token_bytes=tb)
        b = evaluate_loss_logits(system, [batch], 1, token_bytes=tb)
        c = evaluate_loss_fused(system, [batch], 1)
    assert abs(a["total_loss"] - b["total_loss"]) < 1e-6
    assert abs(a["total_loss"] - c["total_loss"]) < 1e-6


def test_all_byte_bearing_targets_agree_too():
    """No control targets at all: the masked second pass equals the plain loss."""
    system, tb, batch = _TinySystem(), _table(), _batch()
    batch["targets"] = torch.randint(0, 12, (2, 6))      # every target carries bytes
    expected = _by_hand(system, batch, tb)
    with torch.no_grad():
        got_fused = evaluate_loss_fused(system, [batch], 1, token_bytes=tb)["bpb"]
        got_logits = evaluate_loss_logits(system, [batch], 1, token_bytes=tb)["bpb"]
    assert abs(got_fused - expected) < 1e-6 and abs(got_logits - expected) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available() or not LIGER_AVAILABLE, reason="needs CUDA + liger")
def test_liger_arm_gives_the_same_bpb():
    """The masked mean is exact on the liger arm (its fused CE normalizes by the
    non-ignored count, same softcap) — the claim the fix relies on for the memory-safe head."""
    system, tb, batch = _TinySystem().cuda(), _table().cuda(), {k: v.cuda() for k, v in _batch().items()}
    expected = _by_hand(system, batch, tb)
    LigerLMHead.setup(system.head)
    with torch.no_grad():
        got = evaluate_loss_fused(system, [batch], 1, token_bytes=tb)["bpb"]
    assert abs(got - expected) < 1e-4


def test_no_byte_bearing_targets_gives_inf_not_nan():
    system = _TinySystem()
    batch = {"idx": torch.tensor([[1, 2]]), "targets": torch.tensor([[CONTROL[0], CONTROL[1]]])}
    tb = torch.zeros(V, dtype=torch.int64)
    with torch.no_grad():
        out = evaluate_loss_fused(system, [batch], 1, token_bytes=tb)
    assert math.isinf(out["bpb"]) and not math.isnan(out["bpb"])
    assert math.isfinite(out["total_loss"])


def test_a_real_nan_on_byte_bearing_targets_reaches_bpb():
    """GPT-6 acceptance counterexample (queue #16 P2): a NaN embedding used by the batch
    must make bpb NaN like total_loss. A blanket nan_to_num hid it (bpb 0.0)."""
    system, tb, batch = _TinySystem(), _table(), _batch()
    with torch.no_grad():
        system.emb.weight[int(batch["idx"][0, 1])] = float("nan")
        out = evaluate_loss_fused(system, [batch], 1, token_bytes=tb)
    assert math.isnan(out["total_loss"]) and math.isnan(out["bpb"])


def test_an_empty_byte_batch_mixed_with_a_normal_one_accumulates_the_normal_one():
    system, tb, batch = _TinySystem(), _table(), _batch()
    empty = {"idx": torch.tensor([[1, 2]]), "targets": torch.tensor([[CONTROL[0], CONTROL[1]]])}
    with torch.no_grad():
        alone = evaluate_loss_fused(system, [batch], 1, token_bytes=tb)["bpb"]
        mixed = evaluate_loss_fused(system, [empty, batch, empty], 3, token_bytes=tb)["bpb"]
    assert abs(mixed - alone) < 1e-6
