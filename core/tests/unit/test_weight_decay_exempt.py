"""Weight decay exemptions declared by modules.

Every parameter is decayed except those its module names in `no_weight_decay_names`: nothing is
inferred from shapes, names or module types. Real builder, CPU; the resume is a real DCP save +
load_checkpoint_if_needed, and the FSDP2 case runs one gloo rank in a spawned process.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from core.model import checkpoint_manager as cm
from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.system import LMSystem
from core.training.optim import build_optimizers, no_weight_decay_params

CFG = {"type": "adamw", "lr_max": 3e-4, "weight_decay": 0.1}


class Gate(nn.Module):
    """GDN-like: A_log and dt_bias parametrize a decay rate, 0 is not their neutral value."""
    no_weight_decay_names = ("A_log", "dt_bias")

    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        self.A_log = nn.Parameter(torch.zeros(width))
        self.dt_bias = nn.Parameter(torch.zeros(width))


class Trunk(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.config = SimpleNamespace(n_embd=width)
        self.wte = nn.Embedding(8, width)                     # embedding role; the rest is the matrix role
        self.gate = Gate(width)
        self.norm = nn.LayerNorm(width)
        self.norm.no_weight_decay_names = ("weight",)         # g * x + b: the gain's neutral value is 1, the bias's 0
        self.out = nn.Linear(width, width)

    def forward(self, idx):
        x = self.wte(idx)
        x = self.gate.proj(x) * self.gate.A_log.exp() + self.gate.dt_bias
        return self.out(self.norm(x))


def _system(trunk=None, head=None):
    return LMSystem(trunk or Trunk(), head or nn.Linear(4, 8))


def _ids(params):
    return [id(p) for p in params]


def _declared(trunk):
    return [trunk.gate.A_log, trunk.gate.dt_bias, trunk.norm.weight]


def test_nothing_declared_keeps_the_three_role_groups():
    """The modern GPT declares nothing: the groups (members, order, settings) are the role groups as before."""
    cfg = GPTConfig(sequence_len=8, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, n_token_types=2)
    trunk, head = GPT(cfg), LMHead(cfg.n_embd, cfg.vocab_size)
    system = LMSystem(trunk, head)
    assert no_weight_decay_params(system) == []
    emb = [p for m in trunk.modules() if isinstance(m, nn.Embedding) for p in m.parameters()]
    matrix = [p for p in trunk.parameters() if id(p) not in set(_ids(emb))]
    opts = build_optimizers(system, CFG)
    assert [[_ids(g["params"]) for g in o.param_groups] for o in opts] == \
        [[_ids(head.parameters()), _ids(emb)], [_ids(matrix)]]
    assert {g["weight_decay"] for o in opts for g in o.param_groups} == {0.1}


def test_declared_parameters_form_a_group_after_their_role_at_weight_decay_0():
    system = _system()
    t = system.trunk
    assert _ids(no_weight_decay_params(system)) == _ids(_declared(t))
    opts = build_optimizers(system, CFG)
    (head_g, emb_g), (rest_g, free_g) = opts[0].param_groups, opts[1].param_groups
    assert _ids(free_g["params"]) == _ids(_declared(t))                       # in the role's own order
    assert _ids(rest_g["params"]) == _ids([t.gate.proj.weight, t.norm.bias, t.out.weight, t.out.bias])
    assert (rest_g["weight_decay"], free_g["weight_decay"]) == (0.1, 0.0)
    assert rest_g["lr"] == free_g["lr"] and rest_g["initial_lr"] == free_g["initial_lr"]
    assert head_g["weight_decay"] == emb_g["weight_decay"] == 0.1             # roles without declarations untouched


def test_an_instance_declaration_replaces_the_class_one():
    system = _system()
    system.trunk.gate.no_weight_decay_names = ("A_log",)
    assert _ids(no_weight_decay_params(system)) == _ids([system.trunk.gate.A_log, system.trunk.norm.weight])
    system.trunk.gate.no_weight_decay_names = ()                              # opting out of the class's
    assert _ids(no_weight_decay_params(system)) == _ids([system.trunk.norm.weight])


class RenamedGate(Gate):
    """Inherits ("A_log", "dt_bias") but calls its rate log_A."""

    def __init__(self, width):
        nn.Module.__init__(self)
        self.proj = nn.Linear(width, width, bias=False)
        self.log_A = nn.Parameter(torch.zeros(width))
        self.dt_bias = nn.Parameter(torch.zeros(width))


@pytest.mark.parametrize("break_it, where, name", [
    (lambda t: setattr(t.gate, "no_weight_decay_names", ("A_lgo",)), "trunk.gate", "A_lgo"),         # a typo
    (lambda t: setattr(t.norm, "no_weight_decay_names", ("weight")), "trunk.norm", "w"),            # no comma: a str
    (lambda t: setattr(t, "no_weight_decay_names", ("norm.weight",)), "trunk", "norm.weight"),      # not its own
    (lambda t: setattr(t, "gate", RenamedGate(4)), "trunk.gate", "A_log"),                          # renamed
])
def test_a_name_that_is_not_one_of_the_modules_own_parameters_raises(break_it, where, name):
    trunk = Trunk()
    break_it(trunk)
    with pytest.raises(ValueError, match=rf"^{where} .*no_weight_decay_names lists {name!r}"):
        build_optimizers(_system(trunk), CFG)


def test_an_alias_can_be_declared_and_a_shared_parameter_is_exempt_once():
    trunk = Trunk()
    trunk.gate.rate = trunk.gate.A_log                        # the same parameter under a second name
    trunk.gate.no_weight_decay_names = ("rate", "dt_bias")
    trunk.out2 = nn.Linear(4, 4)
    trunk.out2.weight = trunk.out.weight                      # shared with out, declared only here
    trunk.out2.no_weight_decay_names = ("weight",)
    system = _system(trunk)
    assert _ids(no_weight_decay_params(system)) == \
        _ids([trunk.gate.A_log, trunk.gate.dt_bias, trunk.norm.weight, trunk.out.weight])
    (rest_g, free_g) = build_optimizers(system, CFG)[1].param_groups
    assert id(trunk.out.weight) in _ids(free_g["params"]) and id(trunk.out.weight) not in _ids(rest_g["params"])


def test_a_wrapper_forwarding_attributes_does_not_take_the_wrapped_modules_declaration():
    """checkpoint_wrapper forwards getattr to the module it wraps; the declaration is read where it was made."""
    trunk = Trunk()
    trunk.gate = checkpoint_wrapper(trunk.gate)
    assert trunk.gate.no_weight_decay_names == ("A_log", "dt_bias")          # what plain getattr would see
    inner = trunk.gate._checkpoint_wrapped_module
    assert _ids(no_weight_decay_params(_system(trunk))) == _ids([inner.A_log, inner.dt_bias, trunk.norm.weight])


def test_the_declarations_survive_the_meta_init_and_deepcopy():
    """Both replace the parameter objects (an attribute on a parameter would be gone); modules and names stay."""
    with torch.device("meta"):
        trunk = Trunk()
    trunk.to_empty(device="cpu")
    system = _system(trunk)
    assert _ids(no_weight_decay_params(system)) == _ids(_declared(trunk))
    twin = copy.deepcopy(system)
    assert _ids(no_weight_decay_params(twin)) == _ids(_declared(twin.trunk))


def _fsdp_worker(rank, init_method, out):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=1)
    try:
        with torch.device("meta"):
            trunk = Trunk()
        trunk.to_empty(device="cpu")
        mesh = init_device_mesh("cpu", (1,))
        fully_shard(trunk.gate, mesh=mesh)
        fully_shard(trunk, mesh=mesh)
        got = no_weight_decay_params(_system(trunk))
        torch.save({"same": [a is b for a, b in zip(got, _declared(trunk))], "n": len(got),
                    "kind": type(got[0].data).__name__, "trunk": type(trunk).__name__}, out)
    finally:
        dist.destroy_process_group()


def test_the_declarations_survive_fsdp2_sharding(tmp_path):
    """fully_shard swaps in new DTensor parameters and the module's class; the declaration finds the live ones."""
    out = tmp_path / "fsdp.pt"
    mp.spawn(_fsdp_worker, args=(f"file://{tmp_path / 'pg'}", str(out)), nprocs=1, join=True)
    got = torch.load(out)
    assert got == {"same": [True, True, True], "n": 3, "kind": "DTensor", "trunk": "FSDPTrunk"}


def _step(system, optimizers):
    system.head(system.trunk(torch.tensor([1, 2, 5]))).sum().backward()
    for opt in optimizers:
        opt.step()
        opt.zero_grad()


def _state(system, optimizers):
    named = dict(system.named_parameters())
    return {(i, n): {k: v.clone() for k, v in o.state[p].items()}
            for i, o in enumerate(optimizers) for n, p in named.items() if p in o.state}


def test_a_role_declared_whole_leaves_no_empty_group_and_resumes(tmp_path):
    """The head's only weight declared: its role is one exempt group, not an empty decayed one beside it (DCP
    refuses empty groups with several optimizers); the checkpoint loads back with its moments."""
    def system():
        head = nn.Linear(4, 8, bias=False)
        head.no_weight_decay_names = ("weight",)
        return _system(head=head)

    torch.manual_seed(0)
    s1 = system()
    o1 = build_optimizers(s1, CFG)
    assert [(len(g["params"]), g["weight_decay"]) for g in o1[0].param_groups] == [(1, 0.0), (1, 0.1)]
    for _ in range(2):
        _step(s1, o1)
    path = tmp_path / "step_000002"
    cm.save_checkpoint_dcp(str(path), s1, o1, meta_data={"step": 2})

    s2 = system()
    o2 = build_optimizers(s2, CFG)
    start, _ = cm.load_checkpoint_if_needed({"resume_from": str(path)}, s2, o2, None, 0, 1)
    assert start > 0
    want, got = _state(s1, o1), _state(s2, o2)
    assert want.keys() == got.keys()
    for key in want:
        for k in want[key]:
            assert torch.equal(want[key][k], got[key][k]), (key, k)
