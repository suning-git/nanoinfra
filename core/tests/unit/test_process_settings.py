"""Process-wide settings belong to the entry, not to core.

Until 0.3.0 importing core.training.model_setup overwrote the CUDA allocator's configuration
(PYTORCH_CUDA_ALLOC_CONF, the user's own value included), and every build_system — load_system for inference
too — set the float32 matmul precision of the whole process to "high". Pinned here:
  * importing core's assembly changes neither the environment nor the matmul precision; build_system does not
    change the matmul precision either (GPU);
  * core.utils.set_torch_global_defaults, what an entry calls instead: it sets the float32 matmul precision to
    "high" whatever the allocator configuration; it writes the allocator default only when the user set neither
    name (an empty value counts as set), works after `import torch` as long as CUDA is not initialized, refuses
    once it is (the setting would be ignored without a word), and the allocator really does use expandable
    segments afterwards (GPU).
"""

import os
import subprocess
import sys

import pytest
import torch

from core.utils import set_torch_global_defaults

NAMES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _run(code, **env):
    """Run `code` in a fresh interpreter with neither allocator variable set (plus `env`); its stdout, stripped."""
    base = {k: v for k, v in os.environ.items() if k not in NAMES + ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
    out = subprocess.run([sys.executable, "-c", code], check=True, env={**base, **env}, capture_output=True,
                         text=True)
    return out.stdout.strip()


def test_importing_core_assembly_changes_no_process_setting():
    code = ("import os, torch; before = torch.get_float32_matmul_precision(); "
            "import core.training.model_setup, core.training.trainer; "
            "print([os.environ.get(n) for n in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')], "
            "before == torch.get_float32_matmul_precision())")
    assert _run(code, CUDA_VISIBLE_DEVICES="") == "[None, None] True"


@pytest.fixture
def process_settings(monkeypatch):
    """A clean start for set_torch_global_defaults: neither allocator name set (whatever the code under test writes
    is undone after), CUDA counted as not yet initialized (in this pytest process an earlier GPU test may well have
    initialized it) unless a test says otherwise, and the matmul precision at PyTorch's default "highest" (restored
    after). Yields a function that sets the user's allocator configuration."""
    for name in NAMES:
        monkeypatch.setenv(name, "")             # recorded first, so that a write by the code under test is undone
        monkeypatch.delenv(name)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    before = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    def set_user_env(**values):
        for name, value in values.items():
            monkeypatch.setenv(name, value)
    yield set_user_env
    torch.set_float32_matmul_precision(before)


def test_default_written_only_when_the_user_set_none(process_settings):
    set_torch_global_defaults()
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"    # the name torch documents as primary
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert torch.get_float32_matmul_precision() == "high"


@pytest.mark.parametrize("user", [{"PYTORCH_ALLOC_CONF": "max_split_size_mb:128"},
                                  {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
                                  {"PYTORCH_ALLOC_CONF": ""},
                                  {"PYTORCH_CUDA_ALLOC_CONF": ""},
                                  {"PYTORCH_ALLOC_CONF": "expandable_segments:False",
                                   "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"}])
def test_the_users_configuration_is_left_as_it_is(process_settings, user):
    process_settings(**user)
    set_torch_global_defaults()
    assert {n: os.environ.get(n) for n in NAMES} == {n: user.get(n) for n in NAMES}
    assert torch.get_float32_matmul_precision() == "high"               # set all the same


def test_refused_once_cuda_is_initialized(process_settings, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="already initialized"):
        set_torch_global_defaults()
    assert not any(n in os.environ for n in NAMES)


def test_not_refused_once_cuda_is_initialized_if_the_allocator_is_configured(process_settings, monkeypatch):
    # Hydra's --multirun runs its jobs one after another in one process: from the second job on CUDA is initialized
    # and the first call's write is in the environment. Nothing is left to set for the allocator, so nothing is
    # refused, and the matmul precision is set as always.
    process_settings(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    set_torch_global_defaults()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert torch.get_float32_matmul_precision() == "high"


def test_core_utils_does_not_import_torch():
    # importing it, process_identity and print0 never touch torch (so never CUDA); set_torch_global_defaults
    # imports torch only when called
    assert _run("import sys; from core.utils import print0, process_identity, set_torch_global_defaults; "
                "process_identity(); print0('', end=''); print('torch' in sys.modules)") == "False"


@gpu
def test_the_allocator_uses_expandable_segments_after_it():
    # after `import torch`, before CUDA initializes: the way an entry reached through `python -m package.module` gets
    # to call it (the package's __init__ may have imported torch already)
    code = ("import torch; from core.utils import set_torch_global_defaults; set_torch_global_defaults(); "
            "x = torch.empty(1 << 24, device='cuda'); "
            "print({s['is_expandable'] for s in torch.cuda.memory._snapshot()['segments']})")
    assert _run(code) == "{True}"
    assert _run("import torch; x = torch.empty(1 << 24, device='cuda'); "
                "print({s['is_expandable'] for s in torch.cuda.memory._snapshot()['segments']})") == "{False}"


@gpu
def test_build_system_leaves_the_matmul_precision_alone():
    from core.model.gpt import GPT, GPTConfig
    from core.training.model_setup import build_system
    tiny = GPTConfig(sequence_len=16, vocab_size=64, n_layer=1, n_head=2, n_kv_head=2, n_embd=16, n_token_types=2)
    before = torch.get_float32_matmul_precision()
    try:
        for precision in ("highest", "medium"):
            torch.set_float32_matmul_precision(precision)
            build_system(GPT, tiny, parallel="ddp")
            assert torch.get_float32_matmul_precision() == precision
    finally:
        torch.set_float32_matmul_precision(before)
