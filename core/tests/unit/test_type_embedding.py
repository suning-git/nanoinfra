"""GPT.forward's token-type term: one_hot(types) @ W, the lookup's rows with a deterministic backward.

Why this test exists (2026-10-06/07). The trunk used to add `self.type_emb(token_types)`.
F.embedding's CUDA backward accumulates with atomics, and the type embedding has a handful of rows sharing tens of
thousands of positions, so the order of those adds — and, with fp32 storage, the gradient's low bits — changed from
run to run (on a world-model micro-batch: 30 repeats of the backward, 30 different results). GPT.forward now computes the
term as one_hot(types) @ W. These tests pin what that must keep and what it must add:

  * the forward is the lookup's, bit for bit: under autocast in either storage dtype, and without autocast in
    bf16. Checked through the whole trunk: the same model with the type rows copied into the token embedding and
    looked up there (token_types=None, idx=types) must give the same hidden states as the type term alone (token
    embedding zeroed);
  * the backward is deterministic on CUDA with fp32 storage, at the skew that exposed the lookup;
  * the lookup's range check survives: an out-of-range type id raises (on CPU, where it raises at once; on CUDA it
    is a device-side assert, which would poison the process's CUDA context).
The precision study (gradient error against fp64, timing) was done with the 2026-10 investigation of
the timing-dependent numerics.
"""

import pytest
import torch

from core.model.gpt import GPT, GPTConfig

N_TYPES = 6


def _model(device, dtype, vocab=64):
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=256, vocab_size=vocab, n_layer=1, n_head=2, n_kv_head=2, n_embd=64,
                    n_token_types=N_TYPES)
    with torch.device(device):
        m = GPT(cfg)
        m.init_weights()
    with torch.no_grad():
        m.type_emb.weight.normal_(0.0, 0.5)          # init_weights zeroes it; give the term something to carry
        for p in m.parameters():                     # stored as build_system stores them (buffers untouched)
            p.data = p.data.to(torch.bfloat16).to(dtype)
    return m


def _types(shape, gen, device, hot=0.9):
    t = torch.where(torch.rand(shape, generator=gen) < hot, 1, torch.randint(0, N_TYPES, shape, generator=gen))
    t.view(-1)[:N_TYPES] = torch.arange(N_TYPES)     # every type present
    return t.to(device)


def _as_lookup(m):
    """The same model with the type rows in the token embedding's first rows: idx=types looks them up."""
    with torch.no_grad():
        m.transformer.wte.weight[:N_TYPES] = m.type_emb.weight.to(m.transformer.wte.weight.dtype)
    return m


def _type_term_vs_lookup(device, dtype, autocast):
    gen = torch.Generator().manual_seed(1)
    types = _types((2, 128), gen, device)
    m = _model(device, dtype)
    with torch.no_grad():
        m.transformer.wte.weight.zero_()
    ctx = torch.autocast(device, dtype=torch.bfloat16) if autocast else torch.autocast(device, enabled=False)
    with torch.no_grad(), ctx:
        got = m(torch.zeros_like(types), token_types=types)       # 0 + the type term
        ref = _as_lookup(m)(types)                                # the lookup, through the same trunk
    return got, ref


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_type_term_is_the_lookup_under_autocast_cpu(dtype):
    got, ref = _type_term_vs_lookup("cpu", dtype, autocast=True)
    assert torch.equal(got, ref)


def test_type_term_is_the_lookup_without_autocast_bf16_cpu():
    got, ref = _type_term_vs_lookup("cpu", torch.bfloat16, autocast=False)
    assert torch.equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_type_term_is_the_lookup_under_autocast_cuda(dtype):
    got, ref = _type_term_vs_lookup("cuda", dtype, autocast=True)
    assert torch.equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_type_emb_backward_deterministic_cuda_fp32_storage():
    gen = torch.Generator().manual_seed(2)
    types = _types((8, 2048), gen, "cuda", hot=0.985)            # most positions in one or two rows, as in the world model
    idx = torch.randint(0, 64, (8, 2048), generator=gen).cuda()
    m = _model("cuda", torch.float32)

    def grad():
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(idx, token_types=types).float().square().mean().backward()
        return m.type_emb.weight.grad.clone()

    grads = [grad() for _ in range(10)]
    assert grads[0].dtype == torch.float32 and grads[0].abs().sum() > 0
    assert all(torch.equal(grads[0], g) for g in grads[1:])


def test_out_of_range_type_raises_cpu():
    m = _model("cpu", torch.float32)
    idx = torch.zeros(1, 3, dtype=torch.long)
    # one_hot raises RuntimeError; the old lookup raised IndexError. Either keeps the check.
    with pytest.raises((RuntimeError, IndexError)):
        m(idx, token_types=torch.tensor([[0, 1, N_TYPES]]))
    with pytest.raises((RuntimeError, IndexError)):
        m(idx, token_types=torch.tensor([[0, -1, 2]]))
