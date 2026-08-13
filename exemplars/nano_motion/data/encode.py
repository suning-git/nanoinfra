"""encode.py — rot139 features to code streams, and captions to pairs.

    prepare.py -> ../train_codec.py -> [encode.py] -> ../train_t2m.py

The AR model never sees rot139. It sees integers, so this step runs every clip
through the trained codec once and stores the result. Doing it here rather than
inside the training loop is not only about speed: it pins WHICH codec produced the
codes. Codes are just integers, and codes from two different codecs are the same
integers on disk — a mislabelled cache trains a model against the wrong decoder and
nothing complains. Every file written here records its codec in a sidecar.

Two kinds of output, keyed by --source:

    lafan1 / amass      <source>_<split>_codes.npz — codes only (unconditional model)
    humanml3d           t2m_humanml3d_<split>.npz — codes + captions (text->motion).
                        Each HumanML3D index row names an AMASS file, a frame range,
                        and a text file of captions written for exactly that range;
                        the crop and the encode happen together HERE, because the
                        alignment has nowhere else to live.

    python -m exemplars.nano_motion.data.encode                       # lafan1
    python -m exemplars.nano_motion.data.encode --source humanml3d    # t2m pairs
    python -m exemplars.nano_motion.data.encode --codec models/motion/mine.pt

Caches land in the regenerable cache root, not beside the dataset — they depend on
a tokenizer, and the rot139 features do not.
"""

import argparse
import csv
import importlib
import json
import time
from pathlib import Path

import numpy as np

from exemplars.nano_motion import spec
from modalities.motion.data import dataset as md
from modalities.motion.data import paths


def load_codec(name, ckpt, device):
    """A shelf tokenizer by name, or a checkpoint you just trained."""
    if ckpt:
        from modalities.motion.tokenizers._convae import MotionCodec
        return MotionCodec(ckpt, device=device), Path(ckpt).name
    mod = importlib.import_module(f"modalities.motion.tokenizers.{name}")
    return mod.load(device=device), name


def encode_clips(codec, clips, progress_every=200):
    """[T,139] clips -> lists of code ids. Clips shorter than the codec's temporal
    downsample produce nothing and are dropped rather than padded — a padded clip is
    a motion that stands still at the end, which is a lie the model would learn."""
    out, dropped = [], 0
    for i, c in enumerate(clips):
        if len(c) < codec.downsample:
            dropped += 1
            out.append(None)
            continue
        out.append(np.asarray(codec.encode(c), dtype=np.int32))
        if i % progress_every == 0:
            print(f"    {i}/{len(clips)}", end="\r", flush=True)
    if dropped:
        print(f"    dropped {dropped} clips shorter than {codec.downsample} frames")
    return out


# --- text->motion pairing: AMASS cropped to HumanML3D's captions ----------------
#
# Ported from the research pipeline that produced the t2m caches this format
# descends from. Three details in it are load-bearing and none are guessable from
# the index file alone:
#
#   1. THE 20fps MAP IS PROPORTIONAL, not seconds-times-rate. HumanML3D annotates a
#      20fps resampling of AMASS, and our loader decimates by an INTEGER stride, so
#      the rate we land on is not exactly 30 (a 100fps source lands on 33.3).
#      Mapping by FRACTION OF THE CLIP is right whatever either rate is:
#          our_idx = round(frame20 / len20_full * our_len),  len20_full = native*20/fps
#   2. THE SPLIT IS HumanML3D's OWN (train.txt / val.txt); its test set is held
#      back here deliberately.
#   3. MIRRORED IDS ("M****") ARE SKIPPED. HumanML3D augments with left-right
#      flipped motion; the captions for those exist but the flipped features do
#      not — pairing them would caption motion that does not exist.

MAX_CAPTIONS = 4        # per clip; HumanML3D gives several, cap to bound augmentation
MIN_CODES = 6           # a clip shorter than this cannot make a training row


def _amass_path(source_path):
    rel = source_path.replace("./pose_data/", "").replace(".npy", ".npz")
    return Path(paths.AMASS_DIR) / rel


def _captions(new_name):
    f = Path(paths.HUMANML3D_DIR) / "texts" / (new_name.replace(".npy", "") + ".txt")
    if not f.exists():
        return []
    caps = []
    for line in f.read_text().splitlines():
        cap = line.split("#")[0].strip()          # text before the first '#'
        if cap:
            caps.append(cap)
        if len(caps) >= MAX_CAPTIONS:
            break
    return caps


def _split_ids():
    """new_name -> split, from HumanML3D's own lists. Mirrored ids dropped here."""
    out = {}
    for split in ("train", "val"):
        f = Path(paths.HUMANML3D_DIR) / f"{split}.txt"
        if not f.exists():
            raise SystemExit(f"{f} missing — HumanML3D is a separate download, "
                             f"see data/README.md")
        for line in f.read_text().splitlines():
            nm = line.strip()
            if nm and not nm.startswith("M"):     # (3) captions exist, motion does not
                out[nm] = split
    return out


def humanml3d_clips(split):
    """Yield (features [T,139], captions, new_name) for one split, in index order."""
    from modalities.motion.data.converters import smpl_body as B
    from modalities.motion.data.converters import smpl_to_rot139 as conv
    from modalities.motion.data.loaders import amass

    split_of = _split_ids()
    J, parents = B.load_body_model("neutral")
    with open(Path(paths.HUMANML3D_DIR) / "index.csv") as f:
        rows = list(csv.DictReader(f))

    missing = skipped = 0
    for r in rows:
        nm = r["new_name"].replace(".npy", "")
        if split_of.get(nm) != split:             # (2) official split, test held back
            continue
        ap = _amass_path(r["source_path"])
        if not ap.exists():
            missing += 1
            continue
        caps = _captions(r["new_name"])
        if not caps:
            skipped += 1
            continue
        d = np.load(ap, allow_pickle=True)
        if "poses" not in d or "trans" not in d:
            skipped += 1
            continue
        fps = float(d["mocap_framerate"]) if "mocap_framerate" in d else amass.TARGET_FPS
        native_len = len(d["poses"])
        poses = amass._resample(np.asarray(d["poses"]), fps)
        trans = amass._resample(np.asarray(d["trans"]), fps)
        feats, _ = conv.smpl_to_features(poses, trans, J, parents)

        our_len = len(feats)
        len20_full = native_len * 20.0 / fps
        if len20_full < 1:
            skipped += 1
            continue
        s20, e20 = int(r["start_frame"]), int(r["end_frame"])
        a = int(round(s20 / len20_full * our_len))        # (1) proportional map
        b = our_len if e20 < 0 else int(round(e20 / len20_full * our_len))
        a, b = max(0, min(a, our_len)), max(0, min(b, our_len))
        if b - a < 22:      # pre-gate before the FK+encode; the real gate is MIN_CODES
            skipped += 1
            continue
        yield feats[a:b].astype(np.float32), caps, nm

    print(f"  ({missing} rows had no local AMASS file — the index also references "
          f"humanact12, which is not AMASS; {skipped} unusable)")


def encode_t2m(codec, tag, out_dir, args):
    """AMASS + HumanML3D -> t2m_humanml3d_<split>.npz (codes + captions).

    Written in the shape T2MDataSource reads: two object arrays, one clip per entry,
    that clip's captions alongside. The data source expands them into
    (caption, motion) pairs — several captions per clip, all describing the same
    motion in different words, which is augmentation rather than a choice.
    """
    for split in args.splits:
        t0 = time.time()
        codes_list, caps_list = [], []
        print(f"\n=== humanml3d/{split}")
        for feats, caps, _nm in humanml3d_clips(split):
            if args.limit and len(codes_list) >= args.limit:
                break
            c = np.asarray(codec.encode(feats), dtype=np.int16)
            if len(c) < MIN_CODES:
                continue
            codes_list.append(c)
            caps_list.append(caps)
            if len(codes_list) % 200 == 0:
                print(f"    {len(codes_list)} clips", end="\r", flush=True)

        path = out_dir / f"t2m_humanml3d_{split}.npz"
        np.savez(path, codes=np.array(codes_list, dtype=object),
                 captions=np.array(caps_list, dtype=object))
        (path.with_suffix(".json")).write_text(json.dumps(
            {"codec": tag, "vocab_size": int(codec.vocab_size),
             "downsample": int(codec.downsample), "rep": codec.rep,
             "source": "humanml3d", "split": split, "clips": len(codes_list),
             "captions": sum(len(c) for c in caps_list)}, indent=2))
        print(f"  {len(codes_list)} clips, {sum(len(c) for c in caps_list)} captions "
              f"-> {path.name} ({time.time() - t0:.0f}s)")

    print("\nnext: python -m exemplars.nano_motion.train_t2m source=humanml3d")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source", default=spec.SOURCE,
                    choices=["lafan1", "amass", "bones_seed", "humanml3d"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--tokenizer", default=spec.TOKENIZER)
    ap.add_argument("--codec", default=None, help="a .pt you trained, instead of a shelf name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="clips per split (smoke test)")
    args = ap.parse_args()

    codec, tag = load_codec(args.tokenizer, args.codec, args.device)
    out_dir = Path(paths.PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"codec {tag}: {codec.vocab_size} codes, downsample {codec.downsample}, "
          f"rep {codec.rep}\n  -> {out_dir}", flush=True)

    if args.source == "humanml3d":
        return encode_t2m(codec, tag, out_dir, args)

    for split in args.splits:
        t0 = time.time()
        clips, _ = md.load_or_build(split, source=args.source, verbose=False)
        if args.limit:
            clips = clips[:args.limit]
        print(f"\n=== {args.source}/{split}: {len(clips)} clips")

        codes = encode_clips(codec, clips)
        kept = [c for c in codes if c is not None]
        total = sum(len(c) for c in kept)

        motion_path = out_dir / f"{args.source}_{split}_codes.npz"
        np.savez(motion_path, codes=np.array(kept, dtype=object))
        (motion_path.with_suffix(".json")).write_text(json.dumps(
            {"codec": tag, "vocab_size": int(codec.vocab_size),
             "downsample": int(codec.downsample), "rep": codec.rep,
             "source": args.source, "split": split, "clips": len(kept), "codes": int(total)},
            indent=2))
        print(f"  {len(kept)} clips, {total} codes -> {motion_path.name} "
              f"({time.time() - t0:.0f}s)")

    print("\nnext: python -m exemplars.nano_motion.train_t2m")


if __name__ == "__main__":
    main()
