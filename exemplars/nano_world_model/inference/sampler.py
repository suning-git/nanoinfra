"""
sampler.py — generate a clip latent frame by latent frame, by asking the model the
question it was trained on.

Training asked one question: given the clean past — every earlier latent frame and the
actions in between — and a block with some cells replaced by [MASK], what are the masked
cells? Generating latent frame k asks exactly that, starting from a fully masked block:

  1. write the td actions that drive frame k into the row, and run the new part of the
     prefix causally into the KV cache;
  2. K rounds: run the block over the cache — its committed cells as codes, the rest
     [MASK], bidirectionally inside the block (attention="bidirectional") — draw a code
     for every masked cell from the video band of the logits, and commit the most
     confident draws, as many as a cosine schedule allows this round (MaskGIT); then
     rewind the cache, so the half-denoised block's keys and values never persist;
  3. run the finished block causally into the cache, as the clean copy of a training
     row holds it.

Step 2's forward sees what the noisy copy of a training row sees — the clean past up to
the block's start, the block itself both ways — at the same rotary positions, so its
logits are the ones training scored (tests/test_sampler.py checks them against the
training forward).

One dynamic KVCache and an eager trunk: a frame of 256 codes costs one forward for the
td actions, K for the block and one to commit it, each of at most 256 positions
(inference/benchmark.py measures it).

    sampler = Sampler(system, layout, rows, mask_id, steps=8, temperature=1.0)
    sampler.start(given_codes, seed=0)            # latent frame 0, local codes [cpf]
    for k in range(rows.n_blocks):
        codes = sampler.step(actions[k])          # the td local action ids driving frame k + 1
"""

import math

import numpy as np
import torch

from core.model.kv_cache import KVCache

from exemplars.nano_world_model import spec


class Sampler:
    """One generation session over one RowLayout: the row being written, its KV cache, and the sampling generator.

    Args:
        system: a System whose trunk carries the 3D rotary table (rope3d.install_rope3d), eager, in eval mode
        vocab: VocabLayout (token types, the video band); rows: RowLayout; mask_id: global id of [MASK]
        steps: K, the denoising rounds per latent frame; temperature: divides the logits before each draw
    """

    def __init__(self, system, vocab, rows, mask_id, steps=8, temperature=1.0):
        self.system, self.vocab, self.rows = system, vocab, rows
        self.mask_id, self.steps, self.temperature = int(mask_id), int(steps), float(temperature)
        self.v_lo, self.v_hi = vocab.ranges[spec.VIDEO_TYPE_ID]
        self.device = next(system.parameters()).device
        self.cache = KVCache.for_model(system.trunk.config, 1, rows.row_len)

    @torch.no_grad()
    def start(self, given_codes, seed=0):
        """A new clip from latent frame 0's local codes [cpf]: the row with every predicted block masked, an empty
        cache, a fresh generator."""
        rows = self.rows
        codes = np.zeros((1, rows.contract["code_len"]), dtype=np.int64)
        codes[0, :rows.cpf] = np.asarray(given_codes, dtype=np.int64)
        actions = np.zeros((1, rows.contract["n_action_tokens"]), dtype=np.int64)
        self.row = rows.assemble(codes, actions)[0].to(self.device)
        for s, e in rows.spans:
            self.row[s:e] = self.mask_id
        self.cache.reset()
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.written, self.k = 0, 1               # cache holds row[:written]; k = the next block

    def _forward(self, lo, hi, attention):
        chunk = self.row[lo:hi].unsqueeze(0)
        return self.system.trunk(chunk, token_types=self.vocab.classify_token_types(chunk), kv_cache=self.cache,
                                 attention=attention)

    @torch.no_grad()
    def block_logits(self):
        """The current block's video-band logits [cpf, n_video] (float32), the block as it stands; the cache is left as
        it was."""
        s, e = self.rows.spans[self.k - 1]
        h = self._forward(s, e, "bidirectional")
        self.cache.rewind(e - s)
        return self.system.head(h)[0].float()[:, self.v_lo:self.v_hi]

    @torch.no_grad()
    def begin(self, actions):
        """Write the td actions driving block k (one local id for all of them, or one per game frame) and run the
        prefix up to the block's start into the cache."""
        rows = self.rows
        assert self.k <= rows.n_blocks, "the clip is complete"
        s, _ = rows.spans[self.k - 1]
        acts = torch.as_tensor(actions, dtype=torch.long, device=self.device).expand(rows.td)
        self.row[s - rows.td:s] = rows.a_off + acts.clamp(0, rows.n_actions - 1)
        self._forward(self.written, s, "causal")
        self.written = s

    @torch.no_grad()
    def denoise(self):
        """K rounds from the masked block to a complete one (module docstring, step 2). Round r leaves
        int(cos(pi/2 * r / K) * cpf) cells masked (none after the last); the draws whose code got the highest
        probability are the ones committed."""
        s, e = self.rows.spans[self.k - 1]
        cpf, K = self.rows.cpf, self.steps
        masked = torch.ones(cpf, dtype=torch.bool, device=self.device)
        left = cpf
        for r in range(1, K + 1):
            keep = int(math.cos(math.pi / 2 * r / K) * cpf) if r < K else 0
            n_commit = left - keep
            if n_commit <= 0:
                continue
            cells = masked.nonzero().squeeze(-1)
            probs = torch.softmax(self.block_logits()[cells] / self.temperature, dim=-1)
            draw = torch.multinomial(probs, 1, generator=self.gen).squeeze(-1)
            conf = probs.gather(-1, draw.unsqueeze(-1)).squeeze(-1)
            top = torch.argsort(conf, descending=True)[:n_commit]
            self.row[s + cells[top]] = self.v_lo + draw[top]
            masked[cells[top]] = False
            left = keep

    @torch.no_grad()
    def commit(self):
        """Run the finished block causally into the cache; move on to the next. Returns its local codes [cpf]."""
        s, e = self.rows.spans[self.k - 1]
        self._forward(s, e, "causal")
        self.written, self.k = e, self.k + 1
        return self.row[s:e] - self.v_lo

    def step(self, actions):
        """One latent frame: begin, denoise, commit."""
        self.begin(actions)
        self.denoise()
        return self.commit()
