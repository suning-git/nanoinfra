"""Storage precision across a checkpoint (2026-10): recorded on save, checked on load.

Why this file exists. build_system(param_dtype=...) stores the parameters, and so the gradients and the AdamW
moments, in bf16 or fp32. A resume continues the run's own optimizer state, but DCP converts any tensor whose stored
dtype differs from the live template's, without a word: a bf16 checkpoint resumed by an fp32 run became an fp32 run
in the middle, and the reverse silently rounded the weights and the moments (found while drafting the user guide).

Pinned here, with real DCP saves and loads on CPU in one process (no mocks):
  * meta.json records the parameters' storage ('param_dtype'), parameters only: a buffer is not a storage choice;
  * a resume compares the dtypes DCP itself recorded (not meta.json), key by key, the parameters before anything is
    touched and then the moments; a difference is refused unless checkpoint.allow_param_dtype_change, and only
    bf16 <-> fp32 is converted even then; a buffer in another dtype is converted with a note, not refused (buffers
    are named by named_buffers(); every other tensor is checked as a parameter, so an unknown wrapper fails closed);
  * an allowed conversion is recorded in param_dtype_history, weights and moments apart; later checkpoints carry it,
    unchanged resumes neither drop nor repeat it, and every resume replaces what the model object held;
  * weights-only loads (init_model_from, inference) convert, and say so.
"""

import json

import pytest
import torch

from core.model import checkpoint_manager as cm
from core.model.gpt import GPTConfig
from core.model.system import LMSystem

TINY = GPTConfig(sequence_len=64, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, n_token_types=3)


class _Trunk(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w = torch.nn.Parameter(torch.zeros(4))
        self.v = torch.nn.Parameter(torch.zeros(4))
        self.register_buffer("count", torch.zeros(3, dtype=torch.int64))   # persistent, integer: not a storage choice
        self.register_buffer("scale", torch.ones(3))                       # persistent, float: not a storage choice

    def forward(self, x):
        return x * self.w.float() + self.v.float()


def _system(dtype=torch.float32, v_dtype=None):
    torch.manual_seed(0)
    system = LMSystem(_Trunk(TINY), torch.nn.Linear(4, 4))
    with torch.no_grad():
        for p in system.parameters():
            p.copy_(torch.randn_like(p))
    for name, p in system.named_parameters():
        p.data = p.data.to(v_dtype if (v_dtype is not None and name == "trunk.v") else dtype)
    return system


def _opt(system):
    opt = torch.optim.AdamW(system.parameters(), lr=1e-2, weight_decay=0.1, fused=False, foreach=False)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


def _train(system, opt, steps=3):
    for _ in range(steps):
        system.head(system.trunk(torch.randn(2, 4)).to(system.head.weight.dtype)).float().sum().backward()
        opt.step()
        opt.zero_grad()


def _saved(tmp_path, dtype, v_dtype=None, name="ckpt", step=7):
    system = _system(dtype, v_dtype)
    opt = _opt(system)
    _train(system, opt)
    path = tmp_path / name
    cm.save_checkpoint_dcp(str(path), system, [opt], meta_data={"step": step})
    return path, system, opt


def _meta(path):
    return json.loads((path / "meta.json").read_text())


# ---- what is recorded -------------------------------------------------------------------------------------------

def test_meta_records_the_parameters_storage_not_the_buffers(tmp_path):
    assert _meta(_saved(tmp_path, torch.float32, name="a")[0])["param_dtype"] == "float32"
    assert _meta(_saved(tmp_path, torch.bfloat16, name="b")[0])["param_dtype"] == "bfloat16"
    # mixed parameters: the sorted names; the int64 buffer is in the state dict but never in the summary
    assert _meta(_saved(tmp_path, torch.float32, torch.bfloat16, name="c")[0])["param_dtype"] == ["bfloat16",
                                                                                                    "float32"]
    assert "param_dtype_history" not in _meta(tmp_path / "a")


def test_the_check_reads_dcp_metadata_not_meta_json(tmp_path):
    path, _, _ = _saved(tmp_path, torch.bfloat16)
    stored = cm.stored_tensor_dtypes(str(path), "model")
    assert stored["trunk.w"] == torch.bfloat16 and stored["trunk.count"] == torch.int64
    assert cm.stored_tensor_dtypes(str(path), "optimizer")["state.trunk.w.exp_avg"] == torch.bfloat16
    # meta.json says fp32 (a wrong summary), or says nothing (a checkpoint from before the field): the bf16 on disk
    # still decides
    for meta_dtype in ("float32", None):
        meta = _meta(path)
        meta.pop("param_dtype")
        if meta_dtype:
            meta["param_dtype"] = meta_dtype
        (path / "meta.json").write_text(json.dumps(meta))
        system = _system(torch.float32)
        with pytest.raises(ValueError, match="allow_param_dtype_change"):
            cm.load_checkpoint_dcp(str(path), system, [_opt(system)])


# ---- resume: same storage passes, another is refused, an allowed change converts and is recorded ----------------

def test_same_storage_resumes_without_history(tmp_path):
    path, src, _ = _saved(tmp_path, torch.bfloat16)
    system = _system(torch.bfloat16)
    meta = cm.load_checkpoint_dcp(str(path), system, [_opt(system)])
    assert meta["step"] == 7
    assert all(torch.equal(a, b) for a, b in zip(system.parameters(), src.parameters()))
    assert system.param_dtype_history == []


def test_another_storage_is_refused_before_anything_is_touched(tmp_path):
    path, _, _ = _saved(tmp_path, torch.bfloat16)
    system = _system(torch.float32)
    with torch.no_grad():
        for p in system.parameters():
            p.fill_(7.0)
    opt = _opt(system)
    with pytest.raises(ValueError, match="bfloat16 -> this model float32 \\(exact\\).*allow_param_dtype_change"):
        cm.load_checkpoint_dcp(str(path), system, [opt])
    assert all(torch.equal(p, torch.full_like(p, 7.0)) for p in system.parameters())
    assert not opt.state, "refused before get_state_dict initialized the optimizer's state"


def test_an_allowed_change_converts_and_is_recorded_once(tmp_path):
    path, src, src_opt = _saved(tmp_path, torch.bfloat16)
    system = _system(torch.float32)
    opt = _opt(system)
    cm.load_checkpoint_dcp(str(path), system, [opt], allow_param_dtype_change=True)
    # weights and moments arrive as fp32, equal to the bf16 values (exact widening), and the step count with them
    for (name, p), q in zip(system.named_parameters(), src.parameters()):
        assert p.dtype == torch.float32 and torch.equal(p, q.float()), name
        st, src_st = opt.state[p], src_opt.state[q]
        assert st["exp_avg"].dtype == torch.float32 and torch.equal(st["exp_avg"], src_st["exp_avg"].float())
        assert torch.count_nonzero(st["exp_avg_sq"]) > 0 and float(st["step"]) == 3.0
    (entry,) = system.param_dtype_history
    assert entry == {"step": 7, "checkpoint": str(path.resolve()), "from": "bfloat16", "to": "float32",
                     "weights": "all", "moments": "all"}
    # the next checkpoint carries it; an unchanged resume of that one keeps it as it was, once
    _train(system, opt, steps=1)
    path2 = tmp_path / "next"
    cm.save_checkpoint_dcp(str(path2), system, [opt], meta_data={"step": 8})
    assert _meta(path2)["param_dtype"] == "float32" and _meta(path2)["param_dtype_history"] == [entry]
    again = _system(torch.float32)
    again_opt = _opt(again)
    cm.load_checkpoint_dcp(str(path2), again, [again_opt])
    assert again.param_dtype_history == [entry]
    path3 = tmp_path / "third"
    cm.save_checkpoint_dcp(str(path3), again, [again_opt], meta_data={"step": 9})
    assert _meta(path3)["param_dtype_history"] == [entry]


def test_a_partial_change_lists_the_converted_parameters(tmp_path):
    path, _, _ = _saved(tmp_path, torch.float32, torch.bfloat16)        # trunk.v stored in bf16, the rest fp32
    system = _system(torch.float32)                                     # all fp32 now
    with pytest.raises(ValueError, match="trunk.v"):
        cm.load_checkpoint_dcp(str(path), system, [_opt(system)])
    system = _system(torch.float32)
    cm.load_checkpoint_dcp(str(path), system, [_opt(system)], allow_param_dtype_change=True)
    assert system.param_dtype_history[0]["weights"] == ["trunk.v"]
    assert system.param_dtype_history[0]["moments"] == ["trunk.v"]


def test_moments_stored_otherwise_are_a_change_too(tmp_path):
    system = _system(torch.float32)
    opt = _opt(system)
    _train(system, opt)
    for st in opt.state.values():                       # same fp32 parameters, bf16 moments on disk
        st["exp_avg"] = st["exp_avg"].bfloat16()
        st["exp_avg_sq"] = st["exp_avg_sq"].bfloat16()
    path = tmp_path / "ckpt"
    cm.save_checkpoint_dcp(str(path), system, [opt], meta_data={"step": 7})
    fresh = _system(torch.float32)
    with pytest.raises(ValueError, match="exp_avg"):
        cm.load_checkpoint_dcp(str(path), fresh, [_opt(fresh)])
    fresh = _system(torch.float32)
    fresh_opt = _opt(fresh)
    cm.load_checkpoint_dcp(str(path), fresh, [fresh_opt], allow_param_dtype_change=True)
    assert all(st["exp_avg"].dtype == torch.float32 for st in fresh_opt.state.values())
    (entry,) = fresh.param_dtype_history                # the weights did not change, only the moments
    assert entry["weights"] == [] and entry["moments"] == "all"


def test_only_bf16_fp32_is_converted_even_when_allowed(tmp_path):
    system = _system(torch.float16)
    path = tmp_path / "ckpt"
    cm.save_checkpoint_dcp(str(path), system, [], meta_data={"step": 7})
    live = _system(torch.float32)
    with pytest.raises(ValueError, match="only bfloat16 <-> float32"):
        cm.load_checkpoint_dcp(str(path), live, [_opt(live)], allow_param_dtype_change=True)


def test_a_tensor_missing_from_the_metadata_raises(tmp_path):
    path, _, _ = _saved(tmp_path, torch.float32)
    system = _system(torch.float32)
    system.head.register_parameter("extra", torch.nn.Parameter(torch.zeros(2)))
    with pytest.raises(ValueError, match="not in the checkpoint's DCP metadata"):
        cm.load_checkpoint_dcp(str(path), system, [_opt(system)])


def test_the_switch_must_be_a_bool(tmp_path):
    path, _, _ = _saved(tmp_path, torch.float32)
    system = _system(torch.float32)
    with pytest.raises(TypeError, match="True or False"):
        cm.load_checkpoint_dcp(str(path), system, [_opt(system)], allow_param_dtype_change="yes")
    with pytest.raises(TypeError, match="True or False"):     # passed on as it is, checked by load_checkpoint_dcp
        cm.load_checkpoint_if_needed({"resume_from": str(path), "allow_param_dtype_change": "true"}, system,
                                     [_opt(system)], dataloader=None, rank=0, world_size=1)


def test_the_trainer_checks_the_switch_when_it_is_built(cpu_cuda_stubs):
    # not first at a resume, which may come hours later after a preemption
    from core.training.optim import build_optimizers
    from core.training.trainer import Trainer

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = GPTConfig(sequence_len=8, vocab_size=16, n_layer=1, n_head=1, n_kv_head=1, n_embd=4)
            self.wte, self.lin = torch.nn.Embedding(16, 4), torch.nn.Linear(4, 4)

        def estimate_flops(self):
            return 96

    opt = {"type": "adamw", "lr_max": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "embedding_lr": 0.2,
           "unembedding_lr": 0.004, "max_grad_norm": 1.0, "scheduler": {"type": "linear", "warmup_steps": 0}}
    system = LMSystem(Trunk(), torch.nn.Linear(4, 16))
    config = {"max_steps": 2, "sequence_len": 8, "device_batch_size": 2, "total_batch_size": 16, "optimizer": opt,
              "logging": {"log_every": 1}, "wandb": {"enabled": False},
              "checkpoint": {"enabled": False, "allow_param_dtype_change": "true"}}
    with pytest.raises(TypeError, match="checkpoint.allow_param_dtype_change"):
        Trainer(system=system, optimizers=build_optimizers(system, opt), dataloader=iter([]), config=config,
                rank=0, world_size=1, evaluators=[])


# ---- weights only: converts, and says so ---------------------------------------------------------------------------

def test_weights_only_loads_convert_and_say_so(tmp_path, capsys):
    path, src, _ = _saved(tmp_path, torch.float32)
    system = _system(torch.bfloat16)
    cm.load_model_only(str(path), system)
    assert "float32 -> this model bfloat16 (rounded)" in capsys.readouterr().out
    assert all(torch.equal(p, q.to(torch.bfloat16)) for p, q in zip(system.parameters(), src.parameters()))
    path, src, _ = _saved(tmp_path, torch.bfloat16, name="b")
    system = _system(torch.float32)
    cm.load_model_only(str(path), system)
    assert "bfloat16 -> this model float32 (exact)" in capsys.readouterr().out
    assert all(torch.equal(p, q.float()) for p, q in zip(system.parameters(), src.parameters()))


def test_a_buffer_in_another_dtype_is_converted_with_a_note_not_refused(tmp_path, capsys):
    # A buffer is the model's own (build_system converts parameters only), so its dtype in a checkpoint can differ
    # from the live template's: a model that changes it during training, or an older build that cast buffers too
    # (bf16 storage cast nano_dsv4's fp32 router bias before 0.3.0). The checkpoint must still resume.
    system = _system(torch.bfloat16)
    opt = _opt(system)
    _train(system, opt)
    path = tmp_path / "ckpt"
    cm.save_checkpoint_dcp(str(path), system, [opt], meta_data={"step": 7})     # buffer stored in fp32
    fresh = _system(torch.bfloat16)
    fresh.trunk.scale = fresh.trunk.scale.to(torch.bfloat16)                      # the template's buffer differs
    cm.load_checkpoint_dcp(str(path), fresh, [_opt(fresh)])
    assert "converting buffers on load, checkpoint float32 -> this model bfloat16" in capsys.readouterr().out
    assert fresh.param_dtype_history == [] and fresh.trunk.scale.dtype == torch.bfloat16


def test_every_resume_replaces_the_history_and_weights_only_clears_it(tmp_path):
    converted, _, _ = _saved(tmp_path, torch.bfloat16, name="bf16")
    plain, _, _ = _saved(tmp_path, torch.float32, name="fp32")
    system = _system(torch.float32)
    cm.load_checkpoint_dcp(str(converted), system, [_opt(system)], allow_param_dtype_change=True)
    assert len(system.param_dtype_history) == 1
    cm.load_checkpoint_dcp(str(plain), system, [_opt(system)])           # same object, a checkpoint with no history
    assert system.param_dtype_history == []
    cm.save_checkpoint_dcp(str(tmp_path / "next"), system, [], meta_data={"step": 8})
    assert "param_dtype_history" not in _meta(tmp_path / "next")
    system.param_dtype_history = [{"stale": True}]
    cm.load_model_only(str(plain), system)
    assert system.param_dtype_history == []


def test_the_trainer_entry_converts_when_allowed(tmp_path):
    path, _, _ = _saved(tmp_path, torch.bfloat16)
    system = _system(torch.float32)
    start, _ = cm.load_checkpoint_if_needed({"resume_from": str(path), "allow_param_dtype_change": True}, system,
                                            [_opt(system)], dataloader=None, rank=0, world_size=1)
    assert start == 8 and system.param_dtype_history[0]["to"] == "float32"
    assert all(p.dtype == torch.float32 for p in system.parameters())


# ---- which tensors count as parameters: buffers are named, everything else is checked (fails closed) ---------------

def test_a_buffer_inside_a_checkpoint_wrapped_module_is_still_a_buffer(tmp_path, capsys):
    # activation checkpointing adds '_checkpoint_wrapped_module.' to the module's names but not to the state dict's
    # keys; the stripping matches them, so the buffer gets its note (unstripped, it would be refused as a parameter)
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    path, _, _ = _saved(tmp_path, torch.bfloat16)                                 # buffer stored in fp32
    fresh = _system(torch.bfloat16)
    fresh.trunk.scale = fresh.trunk.scale.to(torch.bfloat16)
    fresh.trunk = checkpoint_wrapper(fresh.trunk)
    assert any("_checkpoint_wrapped_module." in n for n, _ in fresh.named_buffers())
    cm.load_checkpoint_dcp(str(path), fresh, [_opt(fresh)])
    assert "converting buffers on load, checkpoint float32 -> this model bfloat16" in capsys.readouterr().out
    assert fresh.param_dtype_history == []


def test_a_tied_weight_is_checked_under_both_names(tmp_path, capsys):
    # named_parameters() lists a tied weight once; the state dict keys it under both names, and both are parameters
    def tied(dtype):
        system = _system(dtype)
        system.trunk.emb = torch.nn.Parameter(torch.randn(4, 4).to(dtype))
        system.head.weight = system.trunk.emb
        return system
    src = tied(torch.bfloat16)
    src_opt = _opt(src)
    _train(src, src_opt)
    path = tmp_path / "ckpt"
    cm.save_checkpoint_dcp(str(path), src, [src_opt], meta_data={"step": 7})
    live = tied(torch.float32)
    with pytest.raises(ValueError, match="allow_param_dtype_change"):
        cm.load_checkpoint_dcp(str(path), live, [_opt(live)])
    live = tied(torch.float32)
    cm.load_checkpoint_dcp(str(path), live, [_opt(live)], allow_param_dtype_change=True)
    assert "converting buffers" not in capsys.readouterr().out
    assert live.param_dtype_history[0]["weights"] == "all"


def test_a_buffer_shared_by_two_modules_is_a_buffer_under_both_names(tmp_path, capsys):
    # named_buffers() lists a shared buffer once by default; the state dict keys it under both names
    def shared(dtype, buf_dtype):
        system = _system(dtype)
        buf = torch.ones(3, dtype=buf_dtype)
        system.trunk.a, system.trunk.b = torch.nn.Module(), torch.nn.Module()
        system.trunk.a.register_buffer("s", buf)
        system.trunk.b.register_buffer("s", buf)
        return system
    src = shared(torch.bfloat16, torch.float32)
    src_opt = _opt(src)
    _train(src, src_opt)
    path = tmp_path / "ckpt"
    cm.save_checkpoint_dcp(str(path), src, [src_opt], meta_data={"step": 7})
    live = shared(torch.bfloat16, torch.bfloat16)
    cm.load_checkpoint_dcp(str(path), live, [_opt(live)])
    assert "converting buffers on load" in capsys.readouterr().out and live.param_dtype_history == []


def test_a_wrapper_the_check_does_not_know_fails_closed(tmp_path):
    # torch DDP adds 'module.' to the module's names, which the check does not strip: the parameters then match
    # neither list, and must read as parameters (refused), not as buffers (noted)
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        src = _system(torch.bfloat16)                                   # SGD on both sides: no optimizer state
        path = tmp_path / "ckpt"
        cm.save_checkpoint_dcp(str(path), src, [torch.optim.SGD(src.parameters(), lr=0.1)], meta_data={"step": 7})
        live = torch.nn.parallel.DistributedDataParallel(_system(torch.float32))
        assert all(n.startswith("module.") for n, _ in live.named_parameters())
        with pytest.raises(ValueError, match="allow_param_dtype_change"):
            cm.load_checkpoint_dcp(str(path), live, [torch.optim.SGD(live.parameters(), lr=0.1)])
    finally:
        dist.destroy_process_group()


def test_the_refusal_names_the_config_key_and_survives_mixed_storage():
    one = [("trunk.w", torch.bfloat16, torch.float32)]
    with pytest.raises(ValueError, match=r"param_dtype=bf16 in the training config"):
        cm._check_dtype_changes("ckpt", one, False)
    mixed = one + [("head.w", torch.float32, torch.bfloat16)]       # no single dtype to name: still the refusal
    with pytest.raises(ValueError, match=r"\(param_dtype in the training config"):
        cm._check_dtype_changes("ckpt", mixed, False)
