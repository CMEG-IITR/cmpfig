import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("model_performance_non_flagged.csv")

# Shorten model names for readability
name_map = {
    "gemini_3.5_flash_outputs":       "Gemini 3.5 Flash",
    "gemini_3.1_flash_lite_outputs":  "Gemini 3.1 Lite",
    "gpt54nano_outputs":              "GPT-4o Nano",
    "gpt54minioutputs":               "GPT-4o Mini",
    "DeepSeek_V4_Flash_outputs":      "DeepSeek V4 Flash",
    "mistral_large_3_outputs":        "Mistral Large 3",
}
df["Model"] = df["Model"].map(name_map)

models = df["Model"].tolist()
x = np.arange(len(models))
bar_w = 0.22

colors = {
    "Good":    "#4CAF50",
    "Average": "#FFC107",
    "Poor":    "#F44336",
}

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle("Model Performance — Non-Flagged Samples", fontsize=16, fontweight="bold", y=1.01)

# ── Subplot 1: Caption (stacked bar) ─────────────────────────────────────────
ax = axes[0, 0]
ax.bar(x, df["Caption Good %"],    label="Good",    color=colors["Good"])
ax.bar(x, df["Caption Average %"], label="Average", color=colors["Average"], bottom=df["Caption Good %"])
ax.bar(x, df["Caption Poor %"],    label="Poor",    color=colors["Poor"],
       bottom=df["Caption Good %"] + df["Caption Average %"])
ax.set_title("Caption Quality", fontsize=13, fontweight="bold")
ax.set_ylabel("Percentage (%)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 105)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# ── Subplot 2: Summary (stacked bar) ─────────────────────────────────────────
ax = axes[0, 1]
ax.bar(x, df["Summary Good %"],    label="Good",    color=colors["Good"])
ax.bar(x, df["Summary Average %"], label="Average", color=colors["Average"], bottom=df["Summary Good %"])
ax.bar(x, df["Summary Poor %"],    label="Poor",    color=colors["Poor"],
       bottom=df["Summary Good %"] + df["Summary Average %"])
ax.set_title("Summary Quality", fontsize=13, fontweight="bold")
ax.set_ylabel("Percentage (%)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 105)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# ── Subplot 3: Hallucination ──────────────────────────────────────────────────
ax = axes[1, 0]
bar_colors = ["#4CAF50" if v <= 10 else "#FFC107" if v <= 20 else "#F44336"
              for v in df["Hallucination %"]]
bars = ax.bar(x, df["Hallucination %"], color=bar_colors, edgecolor="white", linewidth=0.6)
ax.set_title("Hallucination Rate", fontsize=13, fontweight="bold")
ax.set_ylabel("Percentage (%)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, df["Hallucination %"].max() * 1.25)
ax.axhline(10, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="10% threshold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, df["Hallucination %"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# ── Subplot 4: Overall Heatmap (Caption Good, Summary Good, Hallucination) ────
ax = axes[1, 1]
heatmap_cols = ["Caption Good %", "Summary Good %", "Hallucination %"]
col_labels    = ["Caption Good",   "Summary Good",   "Hallucination"]
heat_data = df[heatmap_cols].values  # shape: (n_models, 3)

# Build a score matrix: 0→1 where 1 = best performer per column
score = np.zeros_like(heat_data, dtype=float)
for col_i in range(heat_data.shape[1]):
    col = heat_data[:, col_i]
    lo, hi = col.min(), col.max()
    norm = (col - lo) / (hi - lo + 1e-9)
    score[:, col_i] = norm if col_i < 2 else (1 - norm)  # invert hallucination

# Draw each cell manually with Blues colormap so every cell gets its own shade
blues = plt.cm.Blues
for row_i in range(len(models)):
    for col_i in range(len(heatmap_cols)):
        shade = blues(0.25 + score[row_i, col_i] * 0.7)  # range [0.25, 0.95]
        ax.add_patch(plt.Rectangle((col_i - 0.5, row_i - 0.5), 1, 1, color=shade))

ax.set_xlim(-0.5, len(heatmap_cols) - 0.5)
ax.set_ylim(-0.5, len(models) - 0.5)
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)
ax.set_title("Overall Heatmap", fontsize=13, fontweight="bold")
ax.invert_yaxis()

for row_i in range(len(models)):
    for col_i in range(len(heatmap_cols)):
        txt_color = "white" if score[row_i, col_i] > 0.55 else "black"
        ax.text(col_i, row_i, f"{heat_data[row_i, col_i]:.1f}%",
                ha="center", va="center", fontsize=10, fontweight="bold", color=txt_color)

# Colorbar with bracket note for hallucination
sm = plt.cm.ScalarMappable(cmap=blues, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.8, label="Darker = better")
cbar.ax.text(1.6, 0.18, "* Hallucination:\n  darker = lower %\n  (inverted)",
             transform=cbar.ax.transAxes, fontsize=7.5, color="steelblue",
             va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="aliceblue", edgecolor="steelblue", lw=0.8))

plt.tight_layout()
plt.savefig("model_performance_non_flagged.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: model_performance_non_flagged.png")
