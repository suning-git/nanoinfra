"""plot.py — the two val-loss curves, modern vs GPT-2-style trunk, from run.py's
results/curves.json. Writes gpt2_vs_modern.png."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "results" / "curves.json").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "modern":    ("#1f77b4", "modern GPT  (RoPE · RMSNorm · ReLU²)"),
        "gpt2":      ("#d62728", "GPT-2 style  (learned-pos · LayerNorm · GELU)"),
        "gpt2_rope": ("#9467bd", "GPT-2 + RoPE  (learned-pos → RoPE, else GPT-2)"),
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for arm in data["arms"]:
        tr = arm["trajectory"]
        color, label = style.get(arm["arm"], ("gray", arm["arm"]))
        ax.plot([p["step"] for p in tr], [p["val"] for p in tr], "-o",
                color=color, lw=1.9, ms=4, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation cross-entropy")
    ax.set_title(f"GPT-2 vs modern architecture  (d{data['depth']})\n"
                 "same data, budget, and recipe — only the trunk differs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = HERE / "gpt2_vs_modern.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
