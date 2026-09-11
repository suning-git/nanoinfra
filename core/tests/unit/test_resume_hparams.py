"""Optimizer hyperparameters across a resume (queue #1, CHECKPOINT_RESUME.md §0/§8).

A resume restores every param_group entry from the checkpoint while the optimizers
were built from THIS launch's config. One switch, checkpoint.override_optimizer_hparams:
  false (default)  any adjustable difference refuses to start, listing them;
  true             this launch's adjustable values are adopted; weights, moments,
                   step counts and progress stay the checkpoint's.
Options that shape the state or the update (amsgrad, maximize, fused, ...) must match
either way. Nothing is resumed -> the switch is irrelevant.

Real builder, real DCP save + load_checkpoint_if_needed, CPU, no mocks. The
counter-examples come first: before #1 the same resumes went through silently with
the checkpoint's values.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf.errors import ConfigKeyError

from core.model import checkpoint_manager as cm
from core.model.gpt import GPTConfig
from core.model.system import LMSystem
from core.training.lr_schedulers import get_lr_multiplier
from core.training.optim import build_optimizers
from core.training.optim_resume import ADJUSTABLE, reconcile_hparams, snapshot_hparams
from core.training.trainer import Trainer

REPO = Path(__file__).resolve().parents[3]
TEXT_CONFIGS = REPO / "modalities" / "text" / "configs"
CORE_CONFIGS = REPO / "core" / "configs"

TINY = GPTConfig(sequence_len=64, vocab_size=128, n_layer=2,
                 n_head=2, n_kv_head=2, n_embd=32, n_token_types=3)
BASE = {"type": "adamw", "lr_max": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95],
        "embedding_lr": 0.2, "unembedding_lr": 0.004}
NEW = {**BASE, "lr_max": 1.5e-4, "weight_decay": 0.0, "betas": [0.8, 0.9]}


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TINY
        self.wte = torch.nn.Embedding(8, 4)     # -> embedding group
        self.lin = torch.nn.Linear(4, 4)        # -> matrix group

    def forward(self, idx):
        return self.lin(self.wte(idx))


def _system():
    return LMSystem(_Trunk(), torch.nn.Linear(4, 8))   # head -> unembedding group


def _build(cfg):
    system = _system()
    return system, build_optimizers(system, cfg, 1)


BATCH = torch.tensor([1, 2, 5])


def _step(system, optimizers, batch=BATCH):
    system.head(system.trunk(batch)).sum().backward()
    for opt in optimizers:
        opt.step()
        opt.zero_grad()


def _groups(optimizers):
    return [[{k: v for k, v in g.items() if k != "params"} for g in o.param_groups]
            for o in optimizers]


def _adjustable(optimizers):
    return [[{k: g[k] for k in ADJUSTABLE if k in g} for g in o.param_groups] for o in optimizers]


def _state(system, opt):
    return {name: {k: v.clone() for k, v in opt.state[p].items()} for name, p in system.named_parameters()
            if p in opt.state}


def _params(system):
    return {n: p.detach().clone() for n, p in system.named_parameters()}


@pytest.fixture
def ckpt(tmp_path):
    """Checkpoint at step 7 from optimizers built with BASE after three steps, saved
    with the transient lr already decayed by a schedule (lr = initial_lr * 0.5)."""
    torch.manual_seed(0)
    system, opts = _build(BASE)
    for _ in range(3):
        _step(system, opts)
    for opt in opts:
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * 0.5
    path = tmp_path / "step_000007"
    cm.save_checkpoint_dcp(str(path), system, opts, meta_data={"step": 7})
    return SimpleNamespace(path=path, groups=_groups(opts), params=_params(system),
                           state=[_state(system, o) for o in opts])


def _resume(path, cfg, adopt, checkpoint_config=None):
    """Exactly the Trainer's sequence: build, snapshot, load, reconcile."""
    system, opts = _build(cfg)
    before = snapshot_hparams(opts)
    start, _ = cm.load_checkpoint_if_needed(
        checkpoint_config or {"resume_from": str(path)}, system, opts, None, 0, 1)
    changes = reconcile_hparams(opts, before, adopt=adopt) if start > 0 else []
    return system, opts, changes, start


def _resume_expect_error(path, cfg, adopt, match):
    """Raise expected; asserts nothing was written (every check before any write)."""
    system, opts = _build(cfg)
    before = snapshot_hparams(opts)
    start, _ = cm.load_checkpoint_if_needed({"resume_from": str(path)}, system, opts, None, 0, 1)
    assert start > 0
    loaded = _groups(opts)
    with pytest.raises(ValueError, match=match) as excinfo:
        reconcile_hparams(opts, before, adopt=adopt)
    assert _groups(opts) == loaded, "a refused resume must leave the optimizer as loaded"
    return str(excinfo.value)


# ---- counter-examples --------------------------------------------------------------

def test_false_refuses_each_adjustable_difference_and_writes_nothing(ckpt):
    for cfg, field in (({**BASE, "lr_max": 6e-4}, "initial_lr"),
                       ({**BASE, "weight_decay": 0.0}, "weight_decay"),
                       ({**BASE, "betas": [0.8, 0.9]}, "betas")):
        msg = _resume_expect_error(ckpt.path, cfg, False, rf"param_groups\[\d\]\.{field}")
        assert "override_optimizer_hparams=true" in msg                # the way out is named
    # the matrix group is optimizer[1].param_groups[0]; both values are shown
    msg = _resume_expect_error(ckpt.path, {**BASE, "lr_max": 6e-4}, False, r"optimizer\[1\]\.param_groups\[0\]\.initial_lr")
    assert repr(ckpt.groups[1][0]["initial_lr"]) in msg
    # several differences are listed together, not one at a time
    msg = _resume_expect_error(ckpt.path, NEW, False, r"initial_lr")
    assert "weight_decay" in msg and "betas" in msg


def test_options_that_shape_state_or_update_are_refused_even_with_true(ckpt, tmp_path):
    # a checkpoint whose optimizer was built with maximize=True (a group option, not a hyperparameter)
    system, opts = _build(BASE)
    for opt in opts:
        for g in opt.param_groups:
            g["maximize"] = True
    _step(system, opts)
    path = tmp_path / "maximize"
    cm.save_checkpoint_dcp(str(path), system, opts, meta_data={"step": 3})
    for adopt in (False, True):
        msg = _resume_expect_error(path, BASE, adopt, "cannot be overridden")
        assert "maximize" in msg


def test_snapshot_must_match_the_optimizers(ckpt):
    system, opts = _build(BASE)
    before = snapshot_hparams(opts)
    cm.load_checkpoint_if_needed({"resume_from": str(ckpt.path)}, system, opts, None, 0, 1)
    with pytest.raises(ValueError, match="param groups per optimizer"):
        reconcile_hparams(opts, before[:1], adopt=False)
    with pytest.raises(ValueError, match="param groups per optimizer"):
        reconcile_hparams(opts, [before[0][:1], before[1]], adopt=True)
    lost = [[{k: v for k, v in g.items() if k != "initial_lr"} for g in o] for o in before]
    with pytest.raises(ValueError, match="no initial_lr"):
        reconcile_hparams(opts, lost, adopt=False)


def test_switch_must_be_a_bool(ckpt):
    system, opts = _build(BASE)
    before = snapshot_hparams(opts)
    cm.load_checkpoint_if_needed({"resume_from": str(ckpt.path)}, system, opts, None, 0, 1)
    for bad in (1, 0, "true", "false", None):
        with pytest.raises(TypeError, match="override_optimizer_hparams"):
            reconcile_hparams(opts, before, adopt=bad)


# ---- positives ---------------------------------------------------------------------

def test_false_same_config_resumes_and_the_decayed_lr_is_not_a_difference(ckpt):
    system, opts, changes, start = _resume(ckpt.path, BASE, False)
    assert start == 8 and changes == []
    assert _groups(opts) == ckpt.groups                     # incl. the decayed transient lr
    # the real schedule step recomputes lr from initial_lr, not from the transient lr
    sched = {"type": "linear", "max_steps": 20, "warmup_steps": 0, "warmdown_ratio": 0.5}
    lrm = Trainer._apply_lr_schedule(SimpleNamespace(optimizers=opts, scheduler_config=sched), 8)
    assert lrm == get_lr_multiplier(8, sched) == 1.0
    for o in opts:
        for g in o.param_groups:
            assert g["lr"] == g["initial_lr"] * lrm


def test_true_adopts_every_adjustable_and_keeps_the_checkpoints_state(ckpt):
    system, opts, changes, start = _resume(ckpt.path, NEW, True)
    _, fresh = _build(NEW)
    assert _adjustable(opts) == _adjustable(fresh), "this launch's values, defaults included"
    assert start == 8
    # NEW changes lr_max (matrix only), weight_decay and betas (every group); the
    # embedding / unembedding LRs are equal to the checkpoint's and are not "changes"
    assert {(i, j, k) for i, j, k, _, _ in changes} == {
        (1, 0, "initial_lr"),
        (0, 0, "weight_decay"), (0, 1, "weight_decay"), (1, 0, "weight_decay"),
        (0, 0, "betas"), (0, 1, "betas"), (1, 0, "betas")}
    for n, p in system.named_parameters():
        assert torch.equal(p, ckpt.params[n]), n
    for o, saved in zip(opts, ckpt.state):
        live = _state(system, o)
        assert live.keys() == saved.keys()
        for n in saved:
            for k in ("exp_avg", "exp_avg_sq", "step"):
                assert torch.equal(live[n][k], saved[n][k]), (n, k)
            assert float(live[n]["step"]) == 3.0


def test_true_lists_only_what_changed(ckpt):
    _, _, changes, _ = _resume(ckpt.path, {**BASE, "lr_max": 1.5e-4}, True)
    assert [(i, j, k) for i, j, k, _, _ in changes] == [(1, 0, "initial_lr")]
    _, _, changes, _ = _resume(ckpt.path, BASE, True)
    assert changes == []


def test_non_768_width_is_not_rescaled_and_any_layout_works(ckpt, tmp_path):
    # the builder scaled by (32/768)^-0.5 once; adopting copies its value, no second scaling
    _, opts, _, _ = _resume(ckpt.path, {**BASE, "lr_max": 1.5e-4}, True)
    assert opts[1].param_groups[0]["initial_lr"] == pytest.approx(1.5e-4 * (32 / 768) ** -0.5)
    _, fresh = _build({**BASE, "lr_max": 1.5e-4})
    assert opts[1].param_groups[0]["initial_lr"] == fresh[1].param_groups[0]["initial_lr"]
    # one plain optimizer with one group: nothing assumes core's [2, 1] layout
    system = _system()
    opt = torch.optim.AdamW(system.parameters(), lr=1e-3, weight_decay=0.1, fused=False)
    opt.param_groups[0]["initial_lr"] = 1e-3
    _step(system, [opt])
    path = tmp_path / "single"
    cm.save_checkpoint_dcp(str(path), system, [opt], meta_data={"step": 1})
    for adopt, wd in ((False, 0.1), (True, 0.0)):
        system2 = _system()
        opt2 = torch.optim.AdamW(system2.parameters(), lr=2e-3, weight_decay=wd, fused=False)
        opt2.param_groups[0]["initial_lr"] = 2e-3
        before = snapshot_hparams([opt2])
        start, _ = cm.load_checkpoint_if_needed({"resume_from": str(path)}, system2, [opt2], None, 0, 1)
        assert start == 2
        if adopt:
            changes = reconcile_hparams([opt2], before, adopt=True)
            assert opt2.param_groups[0]["initial_lr"] == 2e-3 and opt2.param_groups[0]["weight_decay"] == 0.0
            assert len(changes) == 2
        else:
            with pytest.raises(ValueError, match=r"optimizer\[0\]\.param_groups\[0\]\.initial_lr"):
                reconcile_hparams([opt2], before, adopt=False)


def test_nothing_resumed_ignores_the_switch(ckpt, tmp_path):
    for checkpoint_config in (
        {"resume_from": None},
        {"init_model_from": str(ckpt.path)},
        {"resume_from": "auto", "save_dir": str(tmp_path / "empty")},
    ):
        for adopt in (False, True):
            system, opts, changes, start = _resume(ckpt.path, NEW, adopt, checkpoint_config)
            assert start == 0 and changes == []
            _, fresh = _build(NEW)
            assert _adjustable(opts) == _adjustable(fresh)
            assert all(o.state == {} for o in opts), "no optimizer history without a resume"
    with pytest.raises(FileNotFoundError):
        _resume(ckpt.path, NEW, True, {"resume_from": str(tmp_path / "missing")})


def test_a_real_update_matches_the_hand_patched_reference(ckpt):
    """After adopt, one optimizer step equals the step a hand-patched optimizer takes
    from the same checkpoint on the same batch — the mechanism, not the log."""
    sched = {"type": "linear", "max_steps": 20, "warmup_steps": 0, "warmdown_ratio": 0.5}
    # A: resume with the new config, override on
    sys_a, opts_a, _, _ = _resume(ckpt.path, NEW, True)
    # B: resume with the checkpoint's config, then patch the adjustable values by hand
    sys_b, opts_b, _, _ = _resume(ckpt.path, BASE, False)
    _, fresh_new = _build(NEW)
    for ob, of in zip(opts_b, fresh_new):
        for gb, gf in zip(ob.param_groups, of.param_groups):
            for k in ADJUSTABLE:
                gb[k] = gf[k]
    # C: plain resume, the checkpoint's hyperparameters (control: the values must matter)
    sys_c, opts_c, _, _ = _resume(ckpt.path, BASE, False)
    for system, opts in ((sys_a, opts_a), (sys_b, opts_b), (sys_c, opts_c)):
        Trainer._apply_lr_schedule(SimpleNamespace(optimizers=opts, scheduler_config=sched), 8)
        _step(system, opts, batch=torch.tensor([3, 4]))
    pa, pb, pc = _params(sys_a), _params(sys_b), _params(sys_c)
    assert all(torch.equal(pa[n], pb[n]) for n in pa)
    assert any(not torch.equal(pa[n], pc[n]) for n in pa)


# ---- config layer ------------------------------------------------------------------

def _compose(*overrides):
    with initialize_config_dir(config_dir=str(TEXT_CONFIGS), version_base=None):
        return compose(config_name="train_text",
                       overrides=[f"hydra.searchpath=[file://{CORE_CONFIGS}]", *overrides])


def test_switch_composes_as_a_bool_defaulting_to_false():
    assert _compose().checkpoint.override_optimizer_hparams is False
    assert _compose("checkpoint.override_optimizer_hparams=true").checkpoint.override_optimizer_hparams is True
    assert _compose("checkpoint.override_optimizer_hparams=false").checkpoint.override_optimizer_hparams is False


def test_old_resume_overrides_entry_is_gone():
    with pytest.raises((ConfigCompositionException, ConfigKeyError)):
        _compose("optimizer.resume_overrides.lr_max=1.5e-4")
    assert "resume_overrides" not in _compose().optimizer


# ---- review follow-ups (2026-09-11) ------------------------------------------------

def _single(system, lr, wd, eps=1e-10, **kw):
    opt = torch.optim.AdamW(system.parameters(), lr=lr, weight_decay=wd, eps=eps, fused=False, **kw)
    opt.param_groups[0]["initial_lr"] = lr
    return opt


def test_eps_is_adjustable_like_the_others(tmp_path):
    system = _system()
    opt = _single(system, 1e-3, 0.1, eps=1e-8)
    _step(system, [opt])
    path = tmp_path / "eps"
    cm.save_checkpoint_dcp(str(path), system, [opt], meta_data={"step": 1})
    for adopt in (False, True):
        system2 = _system()
        opt2 = _single(system2, 1e-3, 0.1, eps=1e-10)
        before = snapshot_hparams([opt2])
        cm.load_checkpoint_if_needed({"resume_from": str(path)}, system2, [opt2], None, 0, 1)
        if adopt:
            changes = reconcile_hparams([opt2], before, adopt=True)
            assert opt2.param_groups[0]["eps"] == 1e-10 and [c[2] for c in changes] == ["eps"]
        else:
            with pytest.raises(ValueError, match=r"param_groups\[0\]\.eps"):
                reconcile_hparams([opt2], before, adopt=False)


def test_true_writes_this_launchs_value_even_inside_the_tolerance(ckpt):
    system, opts = _build(BASE)
    before = snapshot_hparams(opts)
    cm.load_checkpoint_if_needed({"resume_from": str(ckpt.path)}, system, opts, None, 0, 1)
    nudged = opts[1].param_groups[0]["initial_lr"] * (1 + 1e-14)      # same recipe, other arithmetic
    before[1][0]["initial_lr"] = nudged
    assert reconcile_hparams(opts, before, adopt=False) == []          # not a config change
    changes = reconcile_hparams(opts, before, adopt=True)
    assert changes == [] and opts[1].param_groups[0]["initial_lr"] == nudged   # adopted, not reported


def test_a_key_on_one_side_only_is_incompatible(ckpt):
    system, opts = _build(BASE)
    before = snapshot_hparams(opts)
    cm.load_checkpoint_if_needed({"resume_from": str(ckpt.path)}, system, opts, None, 0, 1)
    opts[1].param_groups[0]["from_checkpoint_only"] = True             # as a plain load_state_dict could
    for adopt in (False, True):
        with pytest.raises(ValueError, match="absent in this launch"):
            reconcile_hparams(opts, before, adopt=adopt)
    del opts[1].param_groups[0]["from_checkpoint_only"]
    before[0][0]["this_launch_only"] = 1
    with pytest.raises(ValueError, match="absent in the checkpoint"):
        reconcile_hparams(opts, before, adopt=True)


def test_non_finite_betas_in_this_launch_are_rejected(ckpt):
    system, opts = _build(BASE)
    before = snapshot_hparams(opts)
    cm.load_checkpoint_if_needed({"resume_from": str(ckpt.path)}, system, opts, None, 0, 1)
    before[0][0]["betas"] = (0.9, float("nan"))
    with pytest.raises(ValueError, match="not a usable value"):
        reconcile_hparams(opts, before, adopt=True)


def test_incompatible_options_and_differences_are_reported_together(tmp_path):
    system, opts = _build(BASE)
    for opt in opts:
        for g in opt.param_groups:
            g["maximize"] = True
    _step(system, opts)
    path = tmp_path / "both"
    cm.save_checkpoint_dcp(str(path), system, opts, meta_data={"step": 3})
    msg = _resume_expect_error(path, NEW, False, "cannot be overridden")
    assert "maximize" in msg and "weight_decay" in msg and "override_optimizer_hparams=true" in msg


def test_resume_banner_survives_a_start_step_past_the_horizon(ckpt, capsys):
    """Resuming step 20 into max_steps=10 used to train nothing and exit; the banner
    must not turn that into a ZeroDivisionError or a negative multiplier."""
    system, opts, _, _ = _resume(ckpt.path, BASE, False)
    for warmdown in (0, 0.2):
        me = SimpleNamespace(optimizers=opts, _resume_changes=[], max_steps=4, start_step=8,
                             scheduler_config={"type": "linear", "warmup_steps": 0,
                                               "warmdown_ratio": warmdown, "max_steps": 4})
        Trainer._print_resume_info(me, False)
        out = capsys.readouterr().out
        assert "nothing to train" in out and "LR multiplier" not in out
