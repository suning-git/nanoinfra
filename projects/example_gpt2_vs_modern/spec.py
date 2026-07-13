"""spec.py — the architecture comparison this project runs. THE one knob.

A generational A/B: same data, same budget, same recipe — two TRUNKS:

  * modern — the current core GPT (RoPE, RMSNorm, ReLU^2, no biases, QK-norm)
  * gpt2   — GPT2Trunk (trunk.py): the classic 2019 GPT-2 / minGPT architecture
             (learned abs-pos, LayerNorm + bias, tanh-GELU, biased linears)

Both share the same untied LMHead and drive through the same orchestrator
(modalities.text.train_text) via `model.trunk_class`, so the curves isolate the
TRUNK architecture. Unlike an ablation (remove one thing), this measures what the
accumulated modern changes buy, together, on the same footing.
"""
DEPTH = 6                  # smoke scale (a couple of minutes on one GPU)
LR_MAX = "3e-4"            # ONE shared recipe — the honest "swap the arch, keep the
                           #   recipe" comparison (the recipe is tuned for the modern
                           #   trunk; per-arm tuning would be a fairer, larger study)
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 20_000_000
WARMUP_STEPS = 100
N_EVALS = 30
EVAL_TOKENS = 131072

ORCHESTRATOR = "modalities.text.train_text"

# The arms: (label, trunk import path).  None => the modern core GPT.
# gpt2_rope is the single-variable ablation: gpt2 with learned-abs-pos swapped
# for RoPE and NOTHING else changed. The gpt2 -> gpt2_rope gap isolates RoPE;
# gpt2_rope -> modern is everything else the modern trunk accumulates.
ARMS = [
    ("modern",    None),
    ("gpt2",      "gpt2.GPT2Trunk"),            # local module (this folder is on PYTHONPATH)
    ("gpt2_rope", "gpt2_rope.GPT2RoPETrunk"),
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
