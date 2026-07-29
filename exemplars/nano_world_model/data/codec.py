"""codec.py — the frozen Cosmos DV4x8x8 video tokenizer, wrapped to one method.

Only the DATA pipeline needs this file. Training never loads a codec: it reads the
three numbers it needs (vocabulary, spatial and temporal downsampling) from spec.py
as plain constants, so a training job does not put a ~200MB TorchScript pair on the
GPU to learn that 128/8 = 16.

DV4x8x8 means: discrete, temporal /4, spatial /8 (each 8x8 pixel patch of each 4th
frame becomes one of 64000 FSQ codes). It is CAUSAL in time, so T frames become
1 + (T-1)/4 latent frames rather than T/4 — the first frame is encoded alone, which
is what makes the exemplar's "frame 0 is the given observation" layout line up with
a latent frame boundary.

The codec is borrowed, not ours. That is a deliberate choice for an exemplar: it
teaches world-model training without also requiring you to train a tokenizer first.
"""

import numpy as np
import torch


class CosmosDV:
    """Encode pixel clips to discrete code streams.

    Input clips are [T, H, W, 3] uint8 or float in [0,1]; output is one flat int32
    array of `t*h*w` codes per clip, row-major (t, then h, then w) — the same order
    row_layout.py expects when it slices a row into per-latent-frame blocks.
    """

    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16):
        self.device, self.dtype = device, dtype
        self.enc = torch.jit.load(f"{model_dir}/encoder.jit").to(device).eval()

    @staticmethod
    def _pad_frames(T):
        """Cosmos's causal temporal factor wants T = 1 + 4k. Clips that are not are
        padded by repeating the last frame; a 17-frame clip already fits exactly."""
        return 1 + 4 * max(1, (T - 1 + 3) // 4)

    def encode(self, clips):
        """clips: list of [T,H,W,3] arrays, all the same T -> [B, t*h*w] int32.

        Batched in one encoder call. Batch size is the caller's business (encode.py
        picks one that fits); this method just requires them to agree on T.
        """
        xs = []
        for c in clips:
            x = np.asarray(c, dtype=np.float32)
            if x.dtype == np.float32 and x.max() > 1.5:   # uint8-valued -> [0,1]
                x = x / 255.0
            T = x.shape[0]
            Tp = self._pad_frames(T)
            if Tp != T:
                x = np.concatenate([x, np.repeat(x[-1:], Tp - T, axis=0)], axis=0)
            xs.append(x.transpose(3, 0, 1, 2))            # [3,T,H,W]

        xt = torch.from_numpy(np.stack(xs))               # [B,3,T,H,W]
        xt = (xt * 2 - 1).clamp(-1, 1).to(self.dtype).to(self.device)
        with torch.no_grad():
            idx = self.enc(xt)[0]                         # [B,t,h,w] int32
        return idx.reshape(idx.shape[0], -1).to(torch.int32).cpu().numpy()
