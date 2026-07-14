"""
Quick test for supervision strategies.

Verifies that NextTokenPrediction correctly shifts inputs and targets.
"""

import torch
from core.data.supervision import NextTokenPrediction


def test_next_token_prediction():
    """Test NextTokenPrediction strategy."""
    print("Testing NextTokenPrediction...")

    strategy = NextTokenPrediction()

    # Example sequence: [1, 2, 3, 4, 5, 6]
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])  # [B=1, L=6]
    token_types = torch.tensor([[0, 0, 0, 0, 0, 0]])  # All text

    # Apply strategy
    result = strategy.apply(tokens, token_types)

    # Verify shifting
    assert result['idx'].tolist() == [[1, 2, 3, 4, 5]], \
        f"Expected idx=[1,2,3,4,5], got {result['idx'].tolist()}"

    assert result['targets'].tolist() == [[2, 3, 4, 5, 6]], \
        f"Expected targets=[2,3,4,5,6], got {result['targets'].tolist()}"

    assert result['token_types'].tolist() == [[0, 0, 0, 0, 0]], \
        f"Expected token_types=[0,0,0,0,0], got {result['token_types'].tolist()}"

    print("  ✓ Basic shifting works correctly")

    # Test with attention mask (for padding)
    tokens_padded = torch.tensor([[1, 2, 3, 0, 0, 0]])  # [B=1, L=6] with padding
    token_types_padded = torch.tensor([[0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])  # First 3 are valid

    result = strategy.apply(tokens_padded, token_types_padded, attention_mask)

    # Check that padding positions are masked (-1 in targets)
    assert result['targets'].tolist() == [[2, 3, -1, -1, -1]], \
        f"Expected targets=[2,3,-1,-1,-1], got {result['targets'].tolist()}"

    assert result['attention_mask'].tolist() == [[1, 1, 0, 0, 0]], \
        f"Expected attention_mask=[1,1,0,0,0], got {result['attention_mask'].tolist()}"

    print("  ✓ Padding mask works correctly")

    print("✅ All tests passed!\n")


def test_real_example():
    """Test with a real delimiter example."""
    print("Testing with delimiter format...")

    strategy = NextTokenPrediction()

    # Example: [<bos>, <text_start>, tok1, tok2, tok3, <text_end>, <eos>]
    tokens = torch.tensor([[65518, 65520, 10, 20, 30, 65521, 65519]])  # [B=1, L=7]
    token_types = torch.tensor([[2, 2, 0, 0, 0, 2, 2]])  # control, control, text, text, text, control, control

    result = strategy.apply(tokens, token_types)

    # At position 0: see <bos>, predict <text_start>
    # At position 1: see <bos> <text_start>, predict tok1
    # At position 2: see <bos> <text_start> tok1, predict tok2
    # etc.

    expected_idx = [[65518, 65520, 10, 20, 30, 65521]]
    expected_targets = [[65520, 10, 20, 30, 65521, 65519]]
    expected_types = [[2, 2, 0, 0, 0, 2]]

    assert result['idx'].tolist() == expected_idx, \
        f"Expected idx={expected_idx}, got {result['idx'].tolist()}"

    assert result['targets'].tolist() == expected_targets, \
        f"Expected targets={expected_targets}, got {result['targets'].tolist()}"

    assert result['token_types'].tolist() == expected_types, \
        f"Expected types={expected_types}, got {result['token_types'].tolist()}"

    print("  ✓ Delimiter format works correctly")
    print("  Position 0: see <bos>, predict <text_start>")
    print("  Position 1: see [<bos>, <text_start>], predict tok1")
    print("  Position 2: see [<bos>, <text_start>, tok1], predict tok2")
    print("  ...")
    print("✅ Real example test passed!\n")


if __name__ == "__main__":
    test_next_token_prediction()
    test_real_example()
    print("=" * 60)
    print("All supervision strategy tests passed!")
    print("=" * 60)
