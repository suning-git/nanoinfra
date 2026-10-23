"""The System's input preparation (chosen at construction; the trunk's `attention` argument; positions checked
where the rotary table is read), the shared mask tools, and installing a rotary table. On the CPU, except the tests
marked `gpu` (skipped without a CUDA device).
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from core.data import PackingBatcher, PaddingBatcher, build_NTP_supervised_sample
from core.data.supervision import IGNORE
from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.masks import FLEX_BLOCK, causal, default_mask_builder, pad_to_flex_block, same_sample
from core.model.system import LMSystem, prepare_packed_AR_trunk_inputs, prepare_unpacked_AR_trunk_inputs
from torch.nn.attention.flex_attention import BlockMask, and_masks, create_block_mask, flex_attention

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
REPO = Path(__file__).resolve().parents[3]

CFG = GPTConfig(sequence_len=32, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, n_token_types=2)
PADS = {"idx": 0, "targets": IGNORE, "token_types": 0, "target_types": 0}


def gpt():
    torch.manual_seed(0)
    model = GPT(CFG)
    model.init_weights()
    with torch.no_grad():                      # init_weights zeroes every output projection: with them at zero no
        for p in model.parameters():           # position reads any other, and every context error is invisible
            if p.ndim == 2:
                p.normal_(0, 0.2)
    return model.eval()                        # params fp32 as constructed; the rotary tables stay bf16


def samples(lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, n in enumerate(lengths):
        toks = torch.randint(1, CFG.vocab_size, (n + 1,), generator=g)
        out.append(build_NTP_supervised_sample(toks, token_types=torch.zeros(n + 1, dtype=torch.long),
                                               meta={"i": i}))
    return out


def test_the_input_preparation_is_chosen_at_construction():
    batch = next(iter(PaddingBatcher(samples([5, 4]), shape=(2, 8), pad_values=PADS)))
    bare = LMSystem(gpt(), LMHead(CFG.n_embd, CFG.vocab_size))
    with pytest.raises(RuntimeError, match="prepare_trunk_inputs_fn"):
        bare.loss(batch)
    with pytest.raises(AttributeError, match="chosen when the System is built"):
        bare.prepare_trunk_inputs = prepare_unpacked_AR_trunk_inputs          # the old way, refused
    system = LMSystem(gpt(), LMHead(CFG.n_embd, CFG.vocab_size),
                      prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)
    assert torch.isfinite(system.loss(batch))


def test_a_subclass_method_and_a_construction_function_are_not_both_taken():
    class Own(LMSystem):
        def prepare_trunk_inputs(self, batch):
            return batch["idx"], {}
    with pytest.raises(TypeError, match="choose one"):
        Own(gpt(), LMHead(CFG.n_embd, CFG.vocab_size), prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)
    batch = next(iter(PaddingBatcher(samples([5, 4]), shape=(2, 8), pad_values=PADS)))
    assert torch.isfinite(Own(gpt(), LMHead(CFG.n_embd, CFG.vocab_size)).loss(batch))


def test_unpacked_takes_one_sample_per_row_and_refuses_packed_rows():
    padded = next(iter(PaddingBatcher(samples([5, 4]), shape=(2, 8), pad_values=PADS)))
    idx, kw = prepare_unpacked_AR_trunk_inputs(padded)
    assert idx is padded["idx"] and set(kw) == {"token_types"}
    packed = next(iter(PackingBatcher(samples([3, 4]), shape=(1, 8), pad_values=PADS)))
    with pytest.raises(ValueError, match="packed"):
        prepare_unpacked_AR_trunk_inputs(packed)
    plain = {"idx": padded["idx"], "targets": padded["targets"]}             # not from collate: its builder vouches
    assert prepare_unpacked_AR_trunk_inputs(plain)[0] is padded["idx"]


def test_packed_reads_doc_id_and_doc_pos_and_the_trunk_checks_positions():
    """A batch without doc_pos is an error, not the trunk's in-row positions (which would number a row's samples on
    as one sequence); a doc_pos the trunk cannot take is refused by the trunk."""
    packed = next(iter(PackingBatcher(samples([3, 4]), shape=(1, 8), pad_values=PADS)))
    for field in ("doc_id", "doc_pos"):
        with pytest.raises(KeyError, match=field):
            prepare_packed_AR_trunk_inputs({k: v for k, v in packed.items() if k != field})
    idx, kw = prepare_packed_AR_trunk_inputs({**packed, "doc_pos": packed["doc_pos"].int()})
    with pytest.raises(ValueError, match="position_ids must be an int64"):
        gpt()(idx, **kw)
    idx, kw = prepare_packed_AR_trunk_inputs(packed)
    assert set(kw) == {"token_types", "attention", "position_ids"} and kw["position_ids"] is packed["doc_pos"]
    assert isinstance(kw["attention"], BlockMask)


def test_positions_are_checked_where_the_table_is_read_not_by_row_length():
    """A packed row may be longer than the rotary table if its positions are not. A negative position (it would read
    the far end of the table) or one past the table is refused by the GPT, where it reads the current table; the
    preparation checks nothing. On the CPU the refusal is an error (on CUDA a device-side
    assert, tested in its own process on the GPU)."""
    model = gpt()
    table = model.rotary_table_len
    s = samples([5, 6, table - 3])                 # three samples, the longest still inside the table
    packed = next(iter(PackingBatcher(s, shape=(1, table + 16), pad_values=PADS)))
    assert packed["idx"].shape[1] > table
    idx, kw = prepare_packed_AR_trunk_inputs(packed)
    with torch.no_grad():
        assert torch.isfinite(model(idx, **kw)).all()
        for wrong in (-1, table):
            pos = kw["position_ids"].clone()
            pos[0, 2] = wrong
            _, kw_wrong = prepare_packed_AR_trunk_inputs({**packed, "doc_pos": pos})   # not checked here
            with pytest.raises(ValueError, match="outside the rotary table"):
                model(idx, **kw_wrong)


def test_position_ids_are_refused_with_a_cache_before_anything_is_written():
    model = gpt()
    idx = torch.randint(1, CFG.vocab_size, (1, 4))

    class Cache:                                  # any cache: the refusal must come first
        def get_pos(self):
            raise AssertionError("touched the cache")
    with pytest.raises(ValueError, match="KV cache"):
        model(idx, kv_cache=Cache(), position_ids=torch.zeros_like(idx))
    with pytest.raises(ValueError, match="int64"):
        model(idx, position_ids=torch.zeros(1, 4, dtype=torch.int32))


def test_packed_rows_give_each_sample_what_it_gives_alone():
    """Two samples in one packed row (same-sample AND causal mask, doc_pos as positions) against the same two
    samples one per row (causal SDPA, default positions): the real positions must agree. Also checks the test can
    fail: without the same-sample rule (causal only) the second sample reads the first and changes."""
    model = gpt()
    s = samples([7, 5])
    packed = next(iter(PackingBatcher(s, shape=(1, 16), pad_values=PADS)))
    padded = next(iter(PaddingBatcher(s, shape=(2, 16), pad_values=PADS)))
    with torch.no_grad():
        idx, kw = prepare_packed_AR_trunk_inputs(packed)
        h_packed = model(idx, **kw)
        h_alone = model(*[prepare_unpacked_AR_trunk_inputs(padded)[0]],
                        **prepare_unpacked_AR_trunk_inputs(padded)[1])
        for i, n in enumerate([7, 5]):
            got = h_packed[0][packed["doc_id"][0] == i]
            assert got.shape[0] == n
            torch.testing.assert_close(got, h_alone[i, :n], atol=2e-5, rtol=2e-5)
        leak = model(idx, token_types=kw["token_types"], position_ids=kw["position_ids"])   # causal only
        assert not torch.allclose(leak[0][packed["doc_id"][0] == 1], h_alone[1, :5], atol=1e-3)


def test_packed_loss_equals_the_per_sample_loss_sum_over_the_same_targets():
    model = gpt()
    head = LMHead(CFG.n_embd, CFG.vocab_size)
    torch.nn.init.normal_(head.lm_head.weight, 0, 0.2)
    system = LMSystem(model, head, prepare_trunk_inputs_fn=prepare_packed_AR_trunk_inputs)
    s = samples([6, 3, 4])
    packed = next(iter(PackingBatcher(s, shape=(1, 16), pad_values=PADS)))
    padded = next(iter(PaddingBatcher(s, shape=(3, 16), pad_values=PADS)))
    unpacked = LMSystem(model, head, prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)
    with torch.no_grad():
        lp = system.loss(packed)
        lu = unpacked.loss(padded)
    # the same 13 supervised targets, mean over them, in both layouts
    torch.testing.assert_close(lp, lu, atol=2e-5, rtol=2e-5)


# ------------------------------------------------- the shared mask tools

def test_pad_to_flex_block_pads_only_up_to_the_block_multiple():
    x = torch.arange(255).view(1, 255)
    p = pad_to_flex_block(x, -1)
    assert p.shape == (1, 2 * FLEX_BLOCK) and p[0, 255] == -1 and torch.equal(p[:, :255], x)
    full = torch.zeros(2, 256, dtype=torch.long)
    assert pad_to_flex_block(full, -1) is full


def test_same_sample_reads_past_the_row_and_keeps_padding_apart():
    doc = torch.tensor([[0, 0, 1, 1, -1]])                 # T = 5: the last tile reaches index 127
    rule = same_sample(doc)
    b = torch.tensor(0)
    seen = [[bool(rule(b, 0, torch.tensor(q), torch.tensor(k))) for k in range(5)] for q in range(5)]
    assert seen == [[True, True, False, False, False], [True, True, False, False, False],
                    [False, False, True, True, False], [False, False, True, True, False], [False] * 5]
    assert not bool(rule(b, 0, torch.tensor(127), torch.tensor(127)))   # padded index: no sample, sees nothing


def test_the_default_mask_builder_is_the_plain_one_on_the_cpu():
    assert default_mask_builder("cpu") is create_block_mask


def test_an_explicit_mask_builder_is_the_one_used():
    packed = next(iter(PackingBatcher(samples([5, 6]), shape=(1, 16), pad_values=PADS)))
    calls = []

    def builder(rule, **shape):
        calls.append(shape)
        return create_block_mask(rule, **shape)
    _, kw = prepare_packed_AR_trunk_inputs(packed, mask_builder=builder)
    assert len(calls) == 1 and calls[0]["Q_LEN"] == calls[0]["KV_LEN"] == 16 and calls[0]["B"] == 1
    assert isinstance(kw["attention"], BlockMask)


# ------------------------------------------------- installing a rotary table

def test_set_rotary_table_checks_first_and_keeps_the_old_tables_on_error():
    model = gpt()
    cos, sin = model.cos, model.sin
    half = CFG.n_embd // CFG.n_head // 2
    good = torch.zeros(1, 77, 1, half, dtype=torch.bfloat16)
    elsewhere = torch.zeros(1, 77, 1, half, dtype=torch.bfloat16, device="meta")
    as_parameter = torch.nn.Parameter(good.clone(), requires_grad=False)
    for bad_cos, bad_sin, msg in ((good.float(), good, "bfloat16"),
                                  (as_parameter, good, "is a Parameter"),
                                  (good, as_parameter, "is a Parameter"),     # sin refused: cos not written either
                                  (good[:, :, :, :-1], good[:, :, :, :-1], "must be"),
                                  (good, good[:, :70], "differ"),
                                  (elsewhere, elsewhere, "is on meta"),
                                  (good[:, :0], good[:, :0], "at least one row")):
        with pytest.raises(ValueError, match=msg):
            model.set_rotary_table(bad_cos, bad_sin)
        assert model.cos is cos and model.sin is sin
    model.set_rotary_table(torch.nn.Buffer(good), torch.nn.Buffer(good.clone()))   # stays non-persistent
    assert {"cos", "sin"} <= model._non_persistent_buffers_set and not {"cos", "sin"} & set(model.state_dict())
    model.set_rotary_table(good, good.clone())
    assert model.rotary_table_len == 77 and model.cos is good
    model.attach_kv_cache(object())                        # any attached static cache: refused while it is there
    with pytest.raises(ValueError, match="static cache"):
        model.set_rotary_table(good, good)
    model.attach_kv_cache(None)
    model.init_weights()                                   # puts the default table back, at its own length
    assert model.rotary_table_len == cos.size(1) and torch.equal(model.cos, cos)
    assert not {"cos", "sin"} & set(model.state_dict())    # still non-persistent buffers, not saved


def test_the_position_check_follows_the_installed_table():
    """The bound is the table installed now, not the one the model was built with: after a longer table, positions
    past the old end are fine; after a shorter one, positions past its end are refused."""
    model = gpt()
    S = model.rotary_table_len
    half = CFG.n_embd // CFG.n_head // 2
    idx = torch.randint(1, CFG.vocab_size, (1, 8))
    longer = torch.randn(1, 2 * S, 1, half).bfloat16()
    model.set_rotary_table(longer, longer.clone())
    with torch.no_grad():
        assert torch.isfinite(model(idx, position_ids=torch.arange(S, S + 8).view(1, 8))).all()
        shorter = torch.randn(1, 4, 1, half).bfloat16()
        model.set_rotary_table(shorter, shorter.clone())
        with pytest.raises(ValueError, match="outside the rotary table"):
            model(idx, position_ids=torch.arange(8).view(1, 8))


# ------------------------------------------------- on a CUDA device

def on_cuda(batch):
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}


@gpu
def test_build_system_and_load_system_pass_the_preparation_through(tmp_path):
    from core.model.checkpoint_manager import save_checkpoint_dcp
    from core.training.model_setup import build_system, load_system
    batch = on_cuda(next(iter(PackingBatcher(samples([5, 4]), shape=(2, 8), pad_values=PADS))))   # one per row
    built = build_system(GPT, CFG, parallel="ddp", prepare_trunk_inputs_fn=prepare_unpacked_AR_trunk_inputs)["system"]
    assert set(built.prepare_trunk_inputs(batch)[1]) == {"token_types"}
    save_checkpoint_dcp(str(tmp_path), built, [], {"step": 0})
    loaded = load_system(str(tmp_path), prepare_trunk_inputs_fn=prepare_packed_AR_trunk_inputs)["system"]
    assert set(loaded.prepare_trunk_inputs(batch)[1]) == {"token_types", "attention", "position_ids"}
    with pytest.raises(RuntimeError, match="prepare_trunk_inputs_fn"):
        load_system(str(tmp_path))["system"].prepare_trunk_inputs(batch)


@gpu
def test_the_compiled_mask_builder_builds_the_same_mask():
    """Rows not a multiple of 128 long, samples crossing a 128 boundary: the compiled create_block_mask gives the same
    blocks (which are skipped, partial, full) as torch's as it is, and flex the same output with either."""
    packed = next(iter(PackingBatcher(samples([5, 140, 3, 90, 200]), shape=(2, 255), pad_values=PADS)))
    rule = and_masks(same_sample(packed["doc_id"].cuda()), causal)
    shape = dict(B=2, H=None, Q_LEN=255, KV_LEN=255, device="cuda")
    build = default_mask_builder("cuda")
    assert build is default_mask_builder("cuda") and build is not create_block_mask     # compiled, once per process
    plain, compiled = create_block_mask(rule, **shape), build(rule, **shape)
    for name in ("kv_num_blocks", "full_kv_num_blocks"):
        assert torch.equal(getattr(plain, name), getattr(compiled, name)), name
    assert torch.equal(plain.to_dense(), compiled.to_dense())
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(2, 2, 255, 16, device="cuda", generator=g) for _ in range(3))
    with torch.no_grad():
        assert torch.equal(flex_attention(q, k, v, block_mask=plain), flex_attention(q, k, v, block_mask=compiled))


@gpu
@pytest.mark.parametrize("wrong", ["-1", "model.rotary_table_len"])
def test_a_position_outside_the_table_stops_the_process_on_cuda(wrong):
    """On CUDA the range check is a device-side assert (no host sync on the hot path): it stops the process, which
    therefore runs as a child here. The same child first runs in-range positions, so the stop is the bad position.
    Both ends: -1 is what the check is for (indexing would silently read the far end of the table); the message
    tells it from CUDA's own index assert."""
    child = textwrap.dedent(f"""
        import torch
        from core.model.gpt import GPT, GPTConfig
        cfg = GPTConfig(sequence_len=32, vocab_size=64, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, n_token_types=2)
        model = GPT(cfg).cuda()
        idx = torch.zeros(1, 8, dtype=torch.long, device="cuda")
        pos = torch.arange(8, device="cuda").view(1, 8)
        with torch.no_grad():
            model(idx, position_ids=pos)
            torch.cuda.synchronize()
            print("in range: ran", flush=True)
            pos[0, 3] = {wrong}
            model(idx, position_ids=pos)
            torch.cuda.synchronize()
        print("outside: not stopped", flush=True)
    """)
    run = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=600, cwd=REPO,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    assert "in range: ran" in run.stdout and "not stopped" not in run.stdout and run.returncode != 0, run.stderr
    assert "Assertion `position_ids outside the rotary table" in run.stderr


@gpu
def test_training_through_compiled_flex_runs_and_reaches_every_layer():
    """The other side of the refusal: with the blocks compiled, a packed batch trains (the trunk lets the flex call
    through inside the compiled frame), and every layer's attention gets a finite, non-zero gradient."""
    from core.training.model_setup import compile_blocks
    torch._dynamo.reset()
    system = LMSystem(gpt().cuda().train(), LMHead(CFG.n_embd, CFG.vocab_size).cuda(),
                      prepare_trunk_inputs_fn=prepare_packed_AR_trunk_inputs)
    compile_blocks(system.trunk)
    packed = on_cuda(next(iter(PackingBatcher(samples([20, 30, 25]), shape=(1, 100), pad_values=PADS))))
    with torch.autocast("cuda", torch.bfloat16):
        loss = system.loss(packed)
    loss.backward()
    for i, block in enumerate(system.trunk.transformer.h):
        g = block.attn.c_q.weight.grad
        assert g is not None and bool(torch.isfinite(g).all()) and float(g.abs().max()) > 0, f"layer {i}"
