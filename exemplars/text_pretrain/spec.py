"""
spec.py — the model + recipe this project trains. THE one knob to turn.

Every stage reads from here, so re-targeting the project is a one-line edit:
change DEPTH (or LR_MAX) and pretrain / scaling / inference all follow. You do
not hunt for a model definition scattered across scripts.

What is NOT here: training itself. That is the blessed text Orchestrator
`modalities.text.train_text` — a maintained building block (read it to see the
full assemble -> train flow). This project only picks the knobs and drives it.
"""

# --- the model + recipe (edit here to re-target the whole pipeline) -----------
DEPTH  = 12          # transformer depth; dim = DEPTH * 64 = 768  ->  ~135M params
LR_MAX = "3e-4"      # max LR — chosen by the LR bracket (see provenance.md);
                     #          kept a string so it doubles as the checkpoint tag
SEED   = 42

# The training Orchestrator (module entry) this project drives, and where its
# checkpoints land (repo-relative — <repo>/models is the models root, so a fork
# works unedited). Chinchilla auto-sizing (max_steps=-1, ratio 20 tok/param)
# derives the token budget from DEPTH — no manual step count.
from pathlib import Path

ORCHESTRATOR = "modalities.text.train_text"
MODELS_ROOT  = str(Path(__file__).resolve().parents[2] / "models" / "exemplars")


def ckpt_dir(depth=DEPTH, lr=LR_MAX):
    """Where the champion for a given (depth, lr) lives on disk."""
    return f"{MODELS_ROOT}/text_pretrain_d{depth}_lr{lr}"


def train_overrides(depth=DEPTH, lr=LR_MAX, **extra):
    """Hydra CLI overrides pinning this project's recipe on top of the
    orchestrator's blessed defaults. Checkpointing / eval cadence are per-stage
    concerns (the champion saves; scaling cells do not) — a stage adds those via
    **extra. Returns a list like ['model.depth=12', 'optimizer.lr_max=3e-4', ...].
    """
    ov = {
        "model.depth": depth,
        "optimizer.lr_max": lr,
        "seed": SEED,
        "max_steps": -1,        # Chinchilla auto-size (token budget from DEPTH)
        "use_compile": "true",
    }
    ov.update(extra)
    return [f"{k}={v}" for k, v in ov.items()]
