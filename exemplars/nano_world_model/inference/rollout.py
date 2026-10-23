"""
rollout.py — what the model learned, as video: generate val clips and decode them next to the truth.

For each of --clips val clips (spread over the split) the model is given latent frame 0
and the actions that were recorded with the clip, and generates every later latent frame
with the sampler (sampler.py). Generated and true codes are decoded by the Cosmos decoder
and written side by side — truth left, generated right — as one animated GIF per clip,
with the share of each generated latent frame's codes that equal the truth printed
alongside. That share is a coarse number (a frame can be right in every way that matters
with few codes equal); the picture is the point.

    python -m exemplars.nano_world_model.inference.rollout --ckpt <checkpoint step dir>
    python -m exemplars.nano_world_model.inference.rollout --ckpt <dir> --clips 8 --steps 16 --temperature 0.8

Needs the codec's decoder (data/download.py fetches it with the encoder) and the val
cache of the clip length the checkpoint was trained on (--frames).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exemplars.nano_world_model import spec

spec.pin_tokenizer()

from core.training.model_setup import load_system                  # noqa: E402

from exemplars.nano_world_model import train_wm                    # noqa: E402
from exemplars.nano_world_model.data.codec import CosmosDV         # noqa: E402
from exemplars.nano_world_model.dataset import VideoRowDataset     # noqa: E402
from exemplars.nano_world_model.inference.sampler import Sampler   # noqa: E402
from exemplars.nano_world_model.rope3d import install_rope3d       # noqa: E402


def save_gif(path, truth, generated, fps):
    """truth, generated [T, H, W, 3] float in [0, 1] -> one GIF, the two side by side."""
    from PIL import Image
    frames = (np.concatenate([truth, generated], axis=2) * 255).round().astype(np.uint8)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", required=True, help="a checkpoint step directory (meta.json + DCP shards)")
    ap.add_argument("--frames", type=int, default=spec.FRAMES, help="the clip length it was trained on")
    ap.add_argument("--res", type=int, default=spec.RES)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8, help="denoising rounds per latent frame")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--out", default=str(spec.PROJECT / "outputs" / "rollouts"))
    args = ap.parse_args()

    layout, resolver = train_wm.assemble_vocab()
    contract = spec.shape_contract(args.frames, args.res)
    rows = train_wm.build_row_layout(layout, resolver, contract)
    trained = json.loads((Path(args.ckpt) / "meta.json").read_text())["model_config"]["sequence_len"]
    if trained != 2 * rows.row_len:
        raise SystemExit(f"the checkpoint was trained on sequences of {trained}, a {args.frames}-frame row makes "
                         f"{2 * rows.row_len}: pass the --frames it was trained with")
    system = load_system(args.ckpt)["system"]
    install_rope3d(system.trunk, rows)              # the table is not in the checkpoint (non-persistent buffers)
    system.eval()
    sampler = Sampler(system, layout, rows, resolver.resolve(spec.MASK_SLOT), steps=args.steps,
                      temperature=args.temperature)

    val = VideoRowDataset(spec.cache_dir(args.frames, args.res), "val")
    codec = CosmosDV(spec.CODEC_DIR)
    side, cpf, td = args.res // spec.CODEC_SPATIAL_DS, contract["codes_per_frame"], contract["td"]
    out = Path(args.out) / Path(args.ckpt).resolve().name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{rows}; {args.clips} val clips, {args.steps} rounds per latent frame, temperature {args.temperature}")

    for n, i in enumerate(val.spread(args.clips)):
        truth = np.asarray(val.codes[i], dtype=np.int64)
        actions = np.asarray(val.actions[i], dtype=np.int64)
        sampler.start(truth[:cpf], seed=args.seed + n)
        generated = [truth[:cpf]]
        with torch.autocast("cuda", torch.bfloat16):
            for k in range(rows.n_blocks):
                generated.append(sampler.step(actions[k * td:(k + 1) * td]).cpu().numpy())
        generated = np.concatenate(generated)
        same = [(generated[k * cpf:(k + 1) * cpf] == truth[k * cpf:(k + 1) * cpf]).mean()
                for k in range(1, contract["n_latent"])]
        pixels = codec.decode(np.stack([truth, generated]), side)
        path = out / f"val{int(i):06d}.gif"
        save_gif(path, pixels[0], pixels[1], args.fps)
        print(f"  val row {int(i)}: codes equal to the truth per latent frame "
              f"{' '.join(f'{s:.2f}' for s in same)} -> {path}")


if __name__ == "__main__":
    main()
