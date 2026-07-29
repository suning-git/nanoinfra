"""download.py — get a motion dataset onto disk.

    [download.py] -> prepare.py -> ../train_codec.py -> encode.py -> ../train_t2m.py

Two datasets, and the difference between them decides what you can train:

  LAFAN1   Ubisoft's motion capture, BVH, freely downloadable. NO CAPTIONS, so it
           trains a tokenizer and an unconditional motion model, not text->motion.
           This is the default because it needs no account.

  AMASS +  text->motion needs BOTH, and neither can be fetched by a script.
  HumanML3D
           AMASS  — register and accept the licence per subset at
                    https://amass.is.tue.mpg.de, download the "SMPL+H G" archives,
                    unpack to datasets/amass/<Subset>/<subject>/*.npz
           HumanML3D — https://github.com/EricGuo5513/HumanML3D, follow its README to
                    the point where it has produced index.csv, texts/, train.txt and
                    val.txt; copy those four into datasets/humanml3d/

           HumanML3D is captions ONLY: it says which frame range of which AMASS file
           each caption describes. Without AMASS there is nothing to caption; without
           HumanML3D there is nothing to condition on. Which AMASS subsets you take is
           up to you — the index references about a dozen, plus humanact12, which is a
           different dataset and is simply skipped. Every unresolved row is counted and
           reported rather than dropped quietly.

So: LAFAN1 gives you a working pipeline today; AMASS + HumanML3D give you text
conditioning once you have accepted their terms. The exemplar runs either way.

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
    """Report what is present AND what each missing piece blocks.

    A bare inventory makes you work out the consequence yourself; the consequence is
    the part worth printing.
    """
    lafan = len(list((root / "lafan1" / "bvh").glob("*.bvh")))
    parser = (root / "lafan1" / "lafan_pkg" / "extract.py").exists()
    amass = len([d for d in (root / "amass").glob("*") if d.is_dir()])
    texts = len(list((root / "humanml3d" / "texts").glob("*.txt")))
    index = (root / "humanml3d" / "index.csv").exists()
    splits = all((root / "humanml3d" / f"{s}.txt").exists() for s in ("train", "val"))

    for name, got, how in [
        ("LAFAN1 bvh", lafan, "download.py"),
        ("LAFAN1 parser", parser, "download.py"),
        ("AMASS subsets", amass, "amass.is.tue.mpg.de — by hand"),
        ("HumanML3D index.csv", index, "github.com/EricGuo5513/HumanML3D — by hand"),
        ("HumanML3D texts/", texts, "same"),
        ("HumanML3D train/val.txt", splits, "same"),
    ]:
        print(f"  {'ok ' if got else '-- '} {name:<24} {str(got):>8}   ({how})")

    print()
    if lafan and parser:
        print("  ✓ tokenizer + unconditional motion model: ready")
        print("    prepare.py -> train_codec.py -> encode.py -> train_t2m.py")
    else:
        print("  ✗ nothing is ready — run this script without --check for LAFAN1")
    if amass and index and texts and splits:
        print("  ✓ text->motion: ready")
        print("    train_codec.py -> encode.py --source humanml3d -> "
              "train_t2m.py source=humanml3d")
    else:
        need = [n for n, g in [("AMASS", amass), ("index.csv", index),
                               ("texts/", texts), ("train/val.txt", splits)] if not g]
        print(f"  ✗ text->motion blocked: missing {', '.join(need)} "
              f"(both downloads are manual — see this file's docstring)")


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
