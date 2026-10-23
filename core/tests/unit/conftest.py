"""Shared fixtures for core's unit tests."""
import pytest


@pytest.fixture
def cpu_cuda_stubs(monkeypatch):
    """The CUDA queries the Trainer makes for timing, MFU and memory, stubbed, so a test can build and run the real
    Trainer on the CPU (its system and batches are on the CPU either way). One place for every such test: a new query
    the Trainer makes is stubbed here."""
    from core.training import trainer as trainer_module
    monkeypatch.setattr(trainer_module.torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(trainer_module.torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(trainer_module.torch.cuda, "get_device_name", lambda *a, **k: "cpu stub")
