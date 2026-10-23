"""nano_dsv4's own numerics under core's assembly: the router state and the rotary tables are the model's.

build_system converts parameters only (core/training/precision.py). Before 0.3.0 its bf16
storage cast the whole trunk, buffers included, and the router promoted its bias back to fp32 in forward, too late
for a resume: the bf16 template rounded the checkpoint's fp32 bias on load. Pinned here:
  * the load accumulator is zero after init_weights even though to_empty left its memory uninitialized (it is a
    non-persistent buffer, so no checkpoint restores it either); CPU;
  * under bf16 parameter storage the parameters are bf16 while the router bias, the load accumulator and both
    rotary tables (main, compress) stay fp32; GPU (build_system places on CUDA);
  * a resume under bf16 storage restores a non-zero fp32 router bias exactly; GPU.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arch.nano_dsv4 import NanoDSV4, NanoDSV4Config          # noqa: E402
from arch.parts import NoAuxRouter                           # noqa: E402

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="build_system places the model on CUDA")

# Small, keeping the default's structure: hash-routed layers first, then NoAux-routed ones, CSA and HCA layers.
TINY = NanoDSV4Config(sequence_len=64, vocab_size=256, n_layer=4, n_embd=64, n_head=2, head_dim=32, q_lora_rank=32,
                      o_groups=2, o_lora_rank=32, rope_dim=16, sliding_window=16, csa_rate=4, hca_rate=8,
                      compress_ratios=(0, 4, 8, 4), index_topk=4, index_n_heads=2, index_head_dim=16,
                      n_routed_experts=8, num_experts_per_tok=2, moe_expert_dim=32, num_hash_layers=1)


def _routers(model):
    routers = [m for m in model.modules() if isinstance(m, NoAuxRouter)]
    assert routers, "the config must have NoAux-routed layers"
    return routers


def test_load_accumulator_starts_at_zero_whatever_the_memory_held():
    with torch.device("meta"):
        model = NanoDSV4(TINY)
    model.to_empty(device="cpu")
    for router in _routers(model):
        router._load_accum.fill_(7.0)           # what uninitialized memory may hold
    model.init_weights()
    assert all(torch.equal(r._load_accum, torch.zeros_like(r._load_accum)) for r in _routers(model))
    assert all(torch.equal(r.e_score_correction_bias, torch.zeros_like(r.e_score_correction_bias))
               for r in _routers(model))


def _build(param_dtype):
    from core.training.model_setup import build_system
    return build_system(NanoDSV4, TINY, parallel="ddp", param_dtype=param_dtype)["system"]


@gpu
def test_bf16_storage_converts_parameters_and_leaves_the_models_fp32_buffers():
    system = _build(torch.bfloat16)
    assert {p.dtype for p in system.parameters()} == {torch.bfloat16}
    for router in _routers(system):
        assert router.e_score_correction_bias.dtype == torch.float32
        assert router._load_accum.dtype == torch.float32
    trunk = system.trunk
    for name in ("cos_main", "sin_main", "cos_compress", "sin_compress"):
        assert getattr(trunk, name).dtype == torch.float32, name


@gpu
def test_bf16_resume_restores_a_nonzero_router_bias_exactly(tmp_path):
    import core.model.checkpoint_manager as cm
    from core.training.optim import build_optimizers
    opt_cfg = {"type": "adamw"}
    system = _build(torch.bfloat16)
    biases = []
    for i, router in enumerate(_routers(system)):
        bias = 0.123456 + 1e-5 * torch.arange(router.n_experts, device="cuda", dtype=torch.float32) + i
        router.e_score_correction_bias.copy_(bias)
        biases.append(bias)
    assert any(not torch.equal(b, b.bfloat16().float()) for b in biases), "the values must not be bf16-exact"
    path = tmp_path / "ckpt"
    cm.save_checkpoint_dcp(str(path), system, build_optimizers(system, opt_cfg), meta_data={"step": 1})
    fresh = _build(torch.bfloat16)
    cm.load_checkpoint_dcp(str(path), fresh, build_optimizers(fresh, opt_cfg))
    for router, bias in zip(_routers(fresh), biases):
        assert router.e_score_correction_bias.dtype == torch.float32
        assert torch.equal(router.e_score_correction_bias, bias)
