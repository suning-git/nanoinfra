"""
Test evaluate_loss matches Trainer loss calculation.

Tests the loss_eval module using MockModel to isolate the loss computation logic.
Verifies that evaluate_loss produces the same results as Trainer's loss calculation,
especially for selective supervision (e.g., interleaved token sequences).

Usage:
    pytest nanoinfra/tests/test_loss_eval.py -v
    python -m core.tests.test_loss_eval
"""

import torch
from core.evaluation.loss_eval import evaluate_loss


class MockModel(torch.nn.Module):
    """Model that returns a known loss pattern for verification."""

    def __init__(self, loss_pattern=None):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.loss_pattern = loss_pattern  # If None, returns 1.0 everywhere

    def forward(self, x, token_types=None, targets=None, loss_reduction='none'):
        B, T = x.shape
        if self.loss_pattern is not None:
            # Tile pattern to match batch
            loss = self.loss_pattern[:T].unsqueeze(0).expand(B, T).clone()
        else:
            loss = torch.ones(B, T, device=x.device, dtype=torch.float32)

        # Simulate ignore_index=-1 behavior (F.cross_entropy returns 0 for ignored positions)
        if targets is not None:
            loss = torch.where(targets >= 0, loss, torch.zeros_like(loss))

        if loss_reduction == 'none':
            return loss
        return loss.mean()


def create_interleaved_batch(num_frames=4):
    """
    Create a batch mimicking interleaved reconstruction format.

    This simulates post-NextTokenPrediction shift data:
    - Pre-shift tokens: [p1, a1, p2, a2, p3, a3, p4, a4] (8 positions)
    - Post-shift: idx=[:-1], targets=[1:], loss_weights shifted
    - token_types: 2=primary tokens, 1=auxiliary
    - loss_weights: 0 for primary tokens and a1, 1 for a2-a4
    """
    seq_len = num_frames * 2  # 8

    pre_tokens = torch.arange(1000, 1000 + seq_len, dtype=torch.long)
    pre_types = torch.tensor([2, 1] * num_frames, dtype=torch.long)  # j=2, g=1
    pre_weights = torch.zeros(seq_len, dtype=torch.float)
    # auxiliary tokens at positions 1, 3, 5, 7 get weight=1, except a1 (position 1)
    for i in range(num_frames):
        pos_g = i * 2 + 1
        if i > 0:  # a1 not supervised
            pre_weights[pos_g] = 1.0

    # Post-shift
    idx = pre_tokens[:-1].unsqueeze(0)  # [1, 7]
    token_types = pre_types[:-1].unsqueeze(0)  # [1, 7]
    target_types = pre_types[1:].unsqueeze(0)  # [1, 7]
    shifted_weights = pre_weights[1:].unsqueeze(0)  # [1, 7]

    # targets: where shifted_weights=0, set to -1
    raw_targets = pre_tokens[1:].unsqueeze(0)  # [1, 7]
    targets = torch.where(shifted_weights > 0, raw_targets, torch.full_like(raw_targets, -1))

    return {
        'idx': idx,
        'token_types': token_types,
        'targets': targets,
        'target_types': target_types,
        'loss_weights': shifted_weights,
    }


def trainer_loss_calculation(model, batch):
    """Replicate Trainer's loss calculation."""
    with torch.no_grad():
        per_token_loss = model(
            batch['idx'], token_types=batch['token_types'],
            targets=batch['targets'], loss_reduction='none',
        )
        B, T = batch['idx'].shape
        per_token_loss = per_token_loss.reshape(B, T)
        loss = (per_token_loss * batch['loss_weights']).sum() / batch['loss_weights'].sum()
    return loss.item()


def test_eval_matches_trainer_uniform_loss():
    """Test evaluate_loss matches Trainer with uniform per-token loss."""
    model = MockModel(loss_pattern=None)  # Returns 1.0 everywhere
    batch = create_interleaved_batch(num_frames=4)

    # Trainer loss
    trainer_loss = trainer_loss_calculation(model, batch)

    # evaluate_loss (total_loss)
    results = evaluate_loss(model, [batch], steps=1)
    eval_loss = results['total_loss']

    # They should match
    assert abs(trainer_loss - eval_loss) < 1e-6, \
        f"Mismatch! Trainer={trainer_loss:.6f}, eval={eval_loss:.6f}"

    # Also test per-type loss
    results_typed = evaluate_loss(model, [batch], steps=1, type_ids=[1])
    type_1_loss = results_typed['type_losses'][1]

    # With uniform loss=1.0, type 1 loss should also be 1.0
    assert abs(type_1_loss - 1.0) < 1e-6, f"Expected 1.0, got {type_1_loss:.6f}"


def test_eval_matches_trainer_variable_loss():
    """Test evaluate_loss matches Trainer with position-dependent loss."""
    # Loss increases with position: 1.0, 2.0, 3.0, ...
    loss_pattern = torch.arange(1.0, 10.0, dtype=torch.float32)
    model = MockModel(loss_pattern=loss_pattern)
    batch = create_interleaved_batch(num_frames=4)

    # Manually compute expected loss
    # After shift, positions 0-6 have loss_weights [0, 0, 1, 0, 1, 0, 1]
    # Positions with weight=1: 2, 4, 6
    # Loss at these positions: 3.0, 5.0, 7.0
    # Weighted mean = (3 + 5 + 7) / 3 = 5.0
    expected_loss = (3.0 + 5.0 + 7.0) / 3.0

    # Trainer loss
    trainer_loss = trainer_loss_calculation(model, batch)

    # evaluate_loss
    results = evaluate_loss(model, [batch], steps=1)
    eval_loss = results['total_loss']

    # All should match
    assert abs(expected_loss - trainer_loss) < 1e-6, \
        f"Expected={expected_loss:.6f}, Trainer={trainer_loss:.6f}"
    assert abs(trainer_loss - eval_loss) < 1e-6, \
        f"Trainer={trainer_loss:.6f}, eval={eval_loss:.6f}"

    # Check type_losses[1]
    results_typed = evaluate_loss(model, [batch], steps=1, type_ids=[1])
    type_1_loss = results_typed['type_losses'][1]
    assert abs(type_1_loss - expected_loss) < 1e-6, \
        f"Expected type_1_loss={expected_loss:.6f}, got {type_1_loss:.6f}"


def test_unsupervised_type_excluded():
    """Verify unsupervised token type returns inf loss."""
    model = MockModel(loss_pattern=None)  # Returns 1.0 everywhere
    batch = create_interleaved_batch(num_frames=4)

    # Verify primary tokens (type=2) have loss_weights=0
    j_mask = batch['target_types'] == 2
    j_weights = batch['loss_weights'][j_mask]
    assert (j_weights == 0).all(), "primary tokens should have loss_weights=0"

    # If we request type_ids=[2], we should get no valid tokens
    results = evaluate_loss(model, [batch], steps=1, type_ids=[2])
    type_2_loss = results['type_losses'][2]
    # Should be inf (no valid tokens)
    assert type_2_loss == float('inf'), \
        f"type_losses[2] should be inf (no supervised primary tokens), got {type_2_loss}"


def test_type_metrics_int_keys():
    """Test that type_metrics works with int keys (YAML parses int correctly)."""
    model = MockModel(loss_pattern=None)
    batch = create_interleaved_batch(num_frames=4)

    # Integer key (correct)
    results_int = evaluate_loss(model, [batch], steps=1, type_ids=[1])
    assert 1 in results_int['type_losses'], "Integer key 1 should work"

    # Test via SourceLossEvaluator with int keys
    from core.evaluation.evaluator import SourceLossEvaluator

    type_metrics_int = {1: "val/loss_test"}
    eval_int = SourceLossEvaluator([batch], eval_steps=1, type_metrics=type_metrics_int)
    results = eval_int.evaluate(model, None)

    assert "val/loss_test" in results, "Int key should produce metric"
    assert abs(results["val/loss_test"] - 1.0) < 1e-6


def test_multi_batch_accumulation():
    """Test that evaluate_loss correctly accumulates across multiple batches."""
    # Create two batches with different loss patterns
    loss_pattern1 = torch.tensor([1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])  # supervised: 2, 2, 2
    loss_pattern2 = torch.tensor([1.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0])  # supervised: 4, 4, 4

    batch1 = create_interleaved_batch(num_frames=4)
    batch2 = create_interleaved_batch(num_frames=4)

    class StatefulModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.patterns = [loss_pattern1, loss_pattern2]
            self.call_count = 0

        def forward(self, x, token_types=None, targets=None, loss_reduction='none'):
            B, T = x.shape
            pattern = self.patterns[self.call_count % len(self.patterns)]
            self.call_count += 1
            loss = pattern[:T].unsqueeze(0).expand(B, T).clone()
            if targets is not None:
                loss = torch.where(targets >= 0, loss, torch.zeros_like(loss))
            if loss_reduction == 'none':
                return loss
            return loss.mean()

    model = StatefulModel()

    # Manually compute expected
    # Batch 1: supervised positions 2,4,6 have loss 2,2,2 -> sum=6, weight=3
    # Batch 2: supervised positions 2,4,6 have loss 4,4,4 -> sum=12, weight=3
    # Total: sum=18, weight=6 -> mean=3.0
    expected = (6 + 12) / (3 + 3)

    results = evaluate_loss(model, [batch1, batch2], steps=2)

    assert abs(results['total_loss'] - expected) < 1e-6, \
        f"Multi-batch accumulation failed: expected {expected}, got {results['total_loss']}"


def test_total_loss_equals_type_loss_when_all_same_type():
    """Test that total_loss equals type_losses[1] when all supervised tokens are type=1."""
    loss_pattern = torch.arange(1.0, 10.0, dtype=torch.float32)
    model = MockModel(loss_pattern=loss_pattern)
    batch = create_interleaved_batch(num_frames=4)

    # Verify all supervised positions are type=1
    supervised_mask = batch['loss_weights'] > 0
    supervised_types = batch['target_types'][supervised_mask]
    assert (supervised_types == 1).all(), "Test assumption: all supervised should be type=1"

    # Get both total_loss and type_losses
    results = evaluate_loss(model, [batch], steps=1, type_ids=[1])
    total_loss = results['total_loss']
    type_1_loss = results['type_losses'][1]

    assert abs(total_loss - type_1_loss) < 1e-6, \
        f"total_loss ({total_loss}) != type_losses[1] ({type_1_loss})"


def test_debug_batch_contents():
    """Debug: verify batch contents are as expected."""
    batch = create_interleaved_batch(num_frames=4)

    # Check shapes
    assert batch['idx'].shape == (1, 7)
    assert batch['targets'].shape == (1, 7)
    assert batch['token_types'].shape == (1, 7)
    assert batch['target_types'].shape == (1, 7)
    assert batch['loss_weights'].shape == (1, 7)

    # Check loss_weights pattern: [0, 0, 1, 0, 1, 0, 1]
    expected_weights = torch.tensor([[0, 0, 1, 0, 1, 0, 1]], dtype=torch.float)
    assert torch.allclose(batch['loss_weights'], expected_weights), \
        f"Unexpected loss_weights: {batch['loss_weights']}"

    # Check target_types pattern: [1, 2, 1, 2, 1, 2, 1] (g, j, g, j, ...)
    expected_target_types = torch.tensor([[1, 2, 1, 2, 1, 2, 1]], dtype=torch.long)
    assert torch.equal(batch['target_types'], expected_target_types), \
        f"Unexpected target_types: {batch['target_types']}"


if __name__ == "__main__":
    print("Running test_loss_eval tests...\n")

    tests = [
        test_debug_batch_contents,
        test_eval_matches_trainer_uniform_loss,
        test_eval_matches_trainer_variable_loss,
        test_unsupervised_type_excluded,
        test_type_metrics_int_keys,
        test_multi_batch_accumulation,
        test_total_loss_equals_type_loss_when_all_same_type,
    ]

    for test_fn in tests:
        print(f"Running {test_fn.__name__}...", end=" ")
        try:
            test_fn()
            print("PASS")
        except AssertionError as e:
            print(f"FAIL: {e}")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\nAll tests completed.")
