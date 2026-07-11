"""spec.py — the ablation this project runs. THE one knob.

A negative architecture ablation: hold everything fixed — the model geometry,
the data, the token budget, the optimizer — and remove ONE thing, the
transformer's residual (identity) connections. Two arms differ only in the trunk:

  * baseline    — the reference GPT               (core/model/gpt.py)
  * no_residual — NoResidualGPT (trunk.py): the residual path removed

Both are driven through the blessed text Orchestrator (modalities.text.train_text)
via its `model.trunk_class` knob — no core edit, no forked training loop. This is
exactly the shape of a Hackathon negative-ablation: change one architectural
thing, train it against your own baseline, read the two curves.
"""
DEPTH = 6                  # smoke scale. Deeper only WIDENS the gap — residuals
                           #   matter more the deeper the stack (try DEPTH=12).
LR_MAX = "3e-4"
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 20_000_000    # tiny on purpose — a smoke test / worked example (minutes
                           #   on one GPU), not a study. The gap opens in the first
                           #   handful of evals regardless.
WARMUP_STEPS = 100
N_EVALS = 30               # log-spaced val evals; the first lands early (~step 5)
EVAL_TOKENS = 131072       # 128K val tokens/eval — cheap; only relative CE matters here

ORCHESTRATOR = "modalities.text.train_text"

# The two arms: (label, trunk import path).  None => the reference GPT.
ARMS = [
    ("baseline",    None),
    ("no_residual", "projects.example_residual_ablation.trunk.NoResidualGPT"),
]


def train_overrides(trunk_class, max_steps, eval_at):
    """Hydra CLI overrides — this project's recipe on the orchestrator's defaults.
    Constant LR (no warmdown) so each curve is genuine loss-vs-step; an explicit
    log-spaced eval schedule; no checkpoints. The ONLY thing that changes between
    the two arms is `model.trunk_class`."""
    ov = {
        "model.depth": DEPTH,
        "optimizer.lr_max": LR_MAX,
        "seed": SEED,
        "sequence_len": SEQ_LEN,
        "device_batch_size": DBS,
        "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,
        "optimizer.scheduler.warmdown_ratio": 0.0,   # constant LR after warmup
        "optimizer.scheduler.final_lr_frac": 1.0,
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 100,
    }
    if trunk_class:
        ov["model.trunk_class"] = trunk_class
    return [f"{k}={v}" for k, v in ov.items()]
