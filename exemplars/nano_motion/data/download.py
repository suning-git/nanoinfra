"""download.py — get a motion dataset onto disk.

    [download.py] -> prepare.py -> ../train_codec.py -> encode.py -> ../train_t2m.py

Two datasets, and the difference between them decides what you can train:

  LAFAN1   Ubisoft's motion capture, BVH, freely downloadable. NO CAPTIONS, so it
           trains a tokenizer and an unconditional motion model, not text->motion.
           This is the default because it needs no account.

  AMASS    the large SMPL motion-capture collection, and HumanML3D, the caption set
           written for it. Text->motion needs BOTH. Neither can be fetched by a
           script: AMASS requires registering and accepting a license per subset at
           https://amass.is.tue.mpg.de, and HumanML3D's captions are distributed
           through https://github.com/EricGuo5513/HumanML3D. Download them by hand,
           unpack as described below, and prepare.py takes it from there.

So: LAFAN1 gives you a working pipeline today; AMASS gives you text conditioning
once you have accepted its terms. The exemplar is written to run either way.

    python -m exemplars.nano_motion.data.download            # LAFAN1
    python -m exemplars.nano_motion.data.download --check     # what is already here

Expected layout once everything is in place:

    datasets/lafan1/bvh/*.bvh                   (download.py writes this)
    datasets/lafan1/lafan_pkg/{extract,utils}.py
    datasets/amass/<Subset>/<subject>/*.npz     (you unpack this)
    datasets/humanml3d/{index.csv,texts/,train.txt,val.txt}
"""

import argparse
import io
import zipfile
from pathlib import Path

import urllib.request

from exemplars.nano_motion import spec

LAFAN_ZIP = ("https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/"
             "master/lafan1/lafan1.zip")
LAFAN_PARSER = {
    "extract.py": ("https://raw.githubusercontent.com/ubisoft/"
                   "ubisoft-laforge-animation-dataset/master/lafan1/extract.py"),
    "utils.py": ("https://raw.githubusercontent.com/ubisoft/"
                 "ubisoft-laforge-animation-dataset/master/lafan1/utils.py"),
}


def _get(url):
    print(f"  [get ] {url}", flush=True)
    with urllib.request.urlopen(url) as r:
        return r.read()


def fetch_lafan(root):
    """BVH files + the official parser package. Ubisoft's own license applies; this
    fetches from their repository rather than redistributing anything."""
    bvh_dir = root / "bvh"
    if bvh_dir.exists() and any(bvh_dir.glob("*.bvh")):
        print(f"  [have] {len(list(bvh_dir.glob('*.bvh')))} bvh files")
    else:
        bvh_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(_get(LAFAN_ZIP))) as z:
            for name in z.namelist():
                if name.endswith(".bvh"):
                    (bvh_dir / Path(name).name).write_bytes(z.read(name))
        print(f"  -> {len(list(bvh_dir.glob('*.bvh')))} bvh files")

    # The loader imports these as `lafan_pkg`; they are Ubisoft's parser, fetched
    # rather than vendored so the license question stays with the source.
    pkg = root / "lafan_pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name, url in LAFAN_PARSER.items():
        dst = pkg / name
        if dst.exists():
            print(f"  [have] lafan_pkg/{name}")
            continue
        dst.write_bytes(_get(url))


def check(root):
    """Report what is present, so the missing piece is obvious before a long run."""
    rows = [
        ("LAFAN1 bvh", len(list((root / "lafan1" / "bvh").glob("*.bvh"))), "download.py"),
        ("LAFAN1 parser", (root / "lafan1" / "lafan_pkg" / "extract.py").exists(), "download.py"),
        ("AMASS subsets", len([d for d in (root / "amass").glob("*") if d.is_dir()]),
         "manual, see docstring"),
        ("HumanML3D texts", len(list((root / "humanml3d" / "texts").glob("*.txt"))),
         "manual, see docstring"),
    ]
    for name, got, how in rows:
        mark = "ok " if got else "-- "
        print(f"  {mark} {name:<18} {got!s:>8}   ({how})")
    print("\ntext->motion needs AMASS + HumanML3D; LAFAN1 alone trains a tokenizer "
          "and an unconditional motion model.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true", help="report what is on disk and stop")
    args = ap.parse_args()

    root = spec.DATASETS
    if args.check:
        check(root)
        return

    print(f"LAFAN1 -> {root / 'lafan1'}")
    fetch_lafan(root / "lafan1")
    print("\nnext: python -m exemplars.nano_motion.data.prepare")


if __name__ == "__main__":
    main()
