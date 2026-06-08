"""Render the benchmark figure for the README (AUROC by dataset)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, sublabel, AUROC); top row drawn at top
DATA = [
    ("VitaminC", "sufficiency, contrastive", 0.929),
    ("HotpotQA", "sufficiency, ablation", 0.841),
    ("RAGTruth", "RAG grounding, real outputs", 0.819),
]

labels = [d[0] for d in DATA][::-1]
subs = [d[1] for d in DATA][::-1]
vals = [d[2] for d in DATA][::-1]

fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=160)
fig.patch.set_facecolor("white")

colors = ["#cdd6e0", "#cdd6e0", "#4c78a8"]  # highlight the top (VitaminC) bar
bars = ax.barh(range(len(vals)), vals, height=0.6, color=colors, zorder=3)

ax.axvline(0.5, color="#b0b0b0", ls="--", lw=1, zorder=2)
ax.text(0.5, len(vals) - 0.35, "random", color="#909090", fontsize=8, ha="center")

for i, v in enumerate(vals):
    ax.text(
        v - 0.012,
        i,
        f"{v:.3f}",
        va="center",
        ha="right",
        color="white" if colors[i] == "#4c78a8" else "#2b2b2b",
        fontsize=12,
        fontweight="bold",
        zorder=4,
    )

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
ax.set_xlim(0.5, 1.0)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.tick_params(axis="x", labelsize=8, colors="#888")
ax.set_xlabel("AUROC", fontsize=9, color="#555")
ax.set_title(
    "Grounding / sufficiency detection  (gpt-4.1-mini, single logprob pass)",
    fontsize=11,
    fontweight="bold",
    color="#222",
    pad=10,
)

for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#ccc")
ax.grid(axis="x", color="#eee", zorder=0)
ax.margins(y=0.12)

fig.text(
    0.5,
    -0.02,
    "Contrastive control (VitaminC): same claim, evidence flipped → budget moves correctly in 95% of pairs.",
    ha="center",
    fontsize=7.5,
    color="#999",
)

fig.tight_layout()
fig.savefig("bench/benchmarks.png", bbox_inches="tight", facecolor="white")
print("wrote bench/benchmarks.png")
