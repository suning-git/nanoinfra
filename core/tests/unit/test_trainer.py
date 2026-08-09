import math

import pytest

from core.training import trainer


@pytest.mark.parametrize(
    ("device_name", "expected_name", "expected_flops"),
    [
        ("NVIDIA GeForce RTX 4060 Ti", "RTX 4060 Ti", 88e12),
        ("NVIDIA A100-SXM4-80GB", "A100", 312e12),
        ("NVIDIA RTX A6000", "RTX A6000", 155e12),
        ("NVIDIA GeForce RTX 4090", "RTX 4090", 330e12),
        ("NVIDIA H100 80GB HBM3", "H100", 989e12),
    ],
)
def test_detect_gpu_type_supports_training_pool(
    monkeypatch, device_name, expected_name, expected_flops
):
    monkeypatch.setattr(
        trainer.torch.cuda, "get_device_name", lambda _: device_name
    )

    gpu_name, promised_flops = trainer.detect_gpu_type()

    assert gpu_name == expected_name
    assert promised_flops == expected_flops


@pytest.mark.parametrize(
    ("device_name", "expected_name"),
    [
        ("NVIDIA GeForce RTX 2080 Ti", "RTX 2080 Ti"),
        ("Quadro RTX 8000", "RTX 8000"),
    ],
)
def test_detect_gpu_type_recognizes_bf16_incompatible_turing_gpus(
    monkeypatch, device_name, expected_name
):
    monkeypatch.setattr(
        trainer.torch.cuda, "get_device_name", lambda _: device_name
    )

    with pytest.warns(RuntimeWarning, match="does not support BF16"):
        gpu_name, promised_flops = trainer.detect_gpu_type()

    assert gpu_name == expected_name
    assert math.isnan(promised_flops)


def test_unknown_gpu_does_not_block_training(monkeypatch):
    monkeypatch.setattr(
        trainer.torch.cuda, "get_device_name", lambda _: "NVIDIA Future GPU"
    )

    with pytest.warns(RuntimeWarning, match="MFU will be unavailable"):
        gpu_name, promised_flops = trainer.detect_gpu_type()

    assert gpu_name == "NVIDIA Future GPU"
    assert math.isnan(promised_flops)
