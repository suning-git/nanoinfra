"""
The motion modality's data-layout authority.

Importing this module has NO side effects: the data leg's own modules refer to
each other by absolute package path (`from modalities.motion.data import ...`),
like the rest of the repo. The sys.path registration that used to run on import
— the crutch that let research-era FLAT imports (`import smpl_body`,
`import dataset`) resolve — still exists, but only as the explicit function
`register_flat_import_dirs()` below, called by the ARCHIVED projects' own paths
shims. Live code must not call it.

THREE separated data roots (each overridable by env var):

    <repo>/datasets/<name>/   raw datasets + format versions (SHARED inputs)  ($NANOINFRA_DATASETS_DIR)
    <repo>/models/<name>/     pretrained models / promoted artifacts          ($NANOINFRA_MODELS_DIR)
    <base>/motion_caches/     tokenizer-DEPENDENT caches (code streams, t2m pairs)
                              ($NANOINFRA_BASE_DIR)

The split is the point. A rot139 FORMAT VERSION of a dataset is converter output
and depends on no tokenizer, so it lives beside the dataset it came from, at
datasets/<name>/rot139/<split>.npz. A code stream depends on WHICH tokenizer
encoded it — the same integers mean different motion under a different codec —
so it lives in the regenerable cache root and records its tokenizer tag in
tokenizers/REGISTRY.md.
"""

import os
import sys

_PKG = os.path.dirname(os.path.abspath(__file__))                       # modalities/motion/data
_REPO_ROOT = os.path.abspath(os.path.join(_PKG, "..", "..", ".."))     # repo root


def register_flat_import_dirs():
    """Put this package's dirs on sys.path so FLAT imports (`import smpl_body`,
    `import dataset`) resolve — the research-era convention the ARCHIVED motion
    projects (~111 call sites) are frozen on.

    Compat entry ONLY: it is called by those archived projects' own paths shims,
    never by the data leg itself — importing modalities.motion.data must not
    mutate global interpreter state, because the exemplars sit on top of it.
    """
    for _d in (_REPO_ROOT, _PKG,
               os.path.join(_PKG, "converters"), os.path.join(_PKG, "loaders")):
        if _d not in sys.path:
            sys.path.insert(0, _d)


from core.utils import get_base_dir   # noqa: E402

BASE = get_base_dir()                      # artifacts root only (./outputs by default)

# raw datasets + format versions (shared inputs) — own root, NOT under outputs/
DATASETS = os.environ.get("NANOINFRA_DATASETS_DIR") or os.path.join(_REPO_ROOT, "datasets")
LAFAN1_DIR = os.path.join(DATASETS, "lafan1")
AMASS_DIR = os.path.join(DATASETS, "amass")
CMU_BVH_DIR = os.path.join(DATASETS, "cmu_bvh")
BONES_SEED_DIR = os.path.join(DATASETS, "bones_seed")
HUMANML3D_DIR = os.path.join(DATASETS, "humanml3d")   # text captions + index.csv (AMASS↔text)

# pretrained / promoted models (SMPL body model; promoted tokenizer artifacts) — own root
MODELS = os.environ.get("NANOINFRA_MODELS_DIR") or os.path.join(_REPO_ROOT, "models")
AMASS_BODY_MODELS = os.path.join(MODELS, "smplh")

# tokenizer-DEPENDENT caches (code caches / t2m pairings) — the data leg's only
# outputs/ footprint. PROCESSED_DIR is the consumers' contract and keeps its name;
# it points at the regenerable cache root.
PROCESSED_DIR = os.path.join(BASE, "motion_caches")

DEFAULT_SPEC = "rot139"                                # the standard feature spec

# rot139 format-version root (D7.1): datasets/<name>/rot139/<split>.npz. R6c relocates
# the .npz files here + leaves symlinks at the old processed/ path; until then
# format_file() falls back to processed/ so nothing breaks.
def _format_dir(source: str, spec: str) -> str:
    return os.path.join(DATASETS, source, spec)


def ensure_dirs():
    for d in (DATASETS, PROCESSED_DIR):
        os.makedirs(d, exist_ok=True)


def processed_file(source: str, split: str, spec: str = DEFAULT_SPEC) -> str:
    """Path to a dataset FORMAT VERSION: datasets/<source>/<spec>/<split>.npz
    Format versions have exactly one home, beside the dataset."""
    return os.path.join(_format_dir(source, spec), f"{split}.npz")


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.isupper() and isinstance(v, str):
            print(f"{k:18s} {v}")
