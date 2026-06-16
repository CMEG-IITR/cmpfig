"""
Insight plots from test_results.json (category-level results).

Usage:
    python retrieval/plot_insights.py --output_dir retrieval/outputs_proper
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load(output_dir):
    path = os.path.join(output_dir, "eval", "test_results.json")
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/mnt/d/Subham/Compoun_img_01/retrieval/outputs_proper")
    args = p.parse_args()

    data    = load(args.output_dir)
    overall = data["overall"]
    by_cat  = data.get("by_category", data.get("by_subtype", {}))
    N       = overall["n_samples"]

    min_n   = 200
    cats    = [c for c in by_cat if by_cat[c]["n"] >= min_n]
    ns      = np.array([by_cat[c]["n"]           for c in cats])
    r1_i2t  = np.array([by_cat[c]["i2t_R@1"]     for c in cats])
    r10_i2t = np.array([by_cat[c]["i2t_R@10"]    for c in cats])
    r100_i2t= np.array([by_cat[c]["i2t_R@100"]   for c in cats])
    r1_t2i  = np.array([by_cat[c]["t2i_R@1"]     for c in cats])
    r10_t2i = np.array([by_cat[c]["t2i_R@10"]    for c in cats])
    r100_t2i= np.array([by_cat[c]["t2i_R@100"]   for c in cats])
    mrr_i2t = np.array([by_cat[c]["i2t_MRR"]     for c in cats])
    map_i2t = np.array([by_cat[c]["i2t_mAP@100"] for c in cats])

    sort_idx = np.argsort(r1_i2t)        # ascending: worst → best

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(
        f"Cross-Modal Retrieval — Test Set Analysis  |  "
        f"n={N:,}  |  i2t R@1={overall['i2t_R@1']:.3f}  "
        f"R@10={overall['i2t_R@10']:.3f}  R@100={overall['i2t_R@100']:.3f}  "
        f"MRR={overall['i2t_MRR']:.3f}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Plot 1 (top, full width): grouped bars R@1 / R@10 / R@100 per category ──
    ax1 = fig.add_subplot(3, 2, (1, 2))
    x   = np.arange(len(cats))
    w   = 0.25
    sorted_cats = [cats[i] for i in sort_idx]
    ax1.bar(x - w,   r1_i2t[sort_idx],   w, label="R@1",   color="#e74c3c", alpha=0.88)
    ax1.bar(x,       r10_i2t[sort_idx],  w, label="R@10",  color="#f39c12", alpha=0.88)
    ax1.bar(x + w,   r100_i2t[sort_idx], w, label="R@100", color="#27ae60", alpha=0.88)
    ax1.axhline(overall["i2t_R@1"],   color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.6)
    ax1.axhline(overall["i2t_R@10"],  color="#f39c12", linestyle="--", linewidth=1.2, alpha=0.6)
    ax1.axhline(overall["i2t_R@100"], color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{c}\n(n={by_cat[c]['n']})" for c in sorted_cats],
        fontsize=8, rotation=30, ha="right",
    )
    ax1.set_ylabel("Recall (image → text)")
    ax1.set_title("Per-Category Recall@1 / @10 / @100  (i2t, worst → best)", fontsize=11)
    ax1.set_ylim(0, 1.08)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── Plot 2: pool size vs R@1 scatter ──────────────────────────────────────
    ax2 = fig.add_subplot(3, 2, 3)
    sc = ax2.scatter(
        ns, r1_i2t,
        c=r1_i2t, cmap="RdYlGn", s=80, alpha=0.85,
        edgecolors="grey", linewidths=0.5, vmin=0, vmax=0.7,
    )
    for i, c in enumerate(cats):
        ax2.annotate(c, (ns[i], r1_i2t[i]), fontsize=7, alpha=0.9,
                     xytext=(4, 3), textcoords="offset points")
    ax2.axhline(overall["i2t_R@1"], color="navy", linestyle="--",
                linewidth=1.2, label=f"Overall R@1 = {overall['i2t_R@1']:.3f}")
    ax2.set_xlabel("Test samples per category (pool size)")
    ax2.set_ylabel("i2t R@1")
    ax2.set_title("Pool Size vs Recall@1\n(larger pool = harder retrieval)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax2, label="R@1")

    # ── Plot 3: i2t vs t2i R@1 per category ──────────────────────────────────
    ax3 = fig.add_subplot(3, 2, 4)
    sc2 = ax3.scatter(
        r1_i2t, r1_t2i,
        c=np.log1p(ns), cmap="plasma",
        s=80, alpha=0.85, edgecolors="grey", linewidths=0.5,
    )
    lim = max(r1_i2t.max(), r1_t2i.max()) * 1.08
    ax3.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="i2t = t2i")
    for i, c in enumerate(cats):
        ax3.annotate(c, (r1_i2t[i], r1_t2i[i]), fontsize=7, alpha=0.85,
                     xytext=(3, 2), textcoords="offset points")
    ax3.set_xlabel("i2t R@1  (image query → text)")
    ax3.set_ylabel("t2i R@1  (text query → image)")
    ax3.set_title("i2t vs t2i Recall@1 per Category\n(colour = log sample size)", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    plt.colorbar(sc2, ax=ax3, label="log(n)")

    # ── Plot 4: MRR and mAP@100 per category ─────────────────────────────────
    ax4 = fig.add_subplot(3, 2, 5)
    x2  = np.arange(len(cats))
    ax4.bar(x2 - 0.2, mrr_i2t[sort_idx], 0.4, label="MRR",     color="#3498db", alpha=0.88)
    ax4.bar(x2 + 0.2, map_i2t[sort_idx], 0.4, label="mAP@100", color="#9b59b6", alpha=0.88)
    ax4.axhline(overall["i2t_MRR"],     color="#3498db", linestyle="--", linewidth=1.2, alpha=0.6)
    ax4.axhline(overall["i2t_mAP@100"], color="#9b59b6", linestyle="--", linewidth=1.2, alpha=0.6)
    ax4.set_xticks(x2)
    ax4.set_xticklabels(sorted_cats, fontsize=7.5, rotation=35, ha="right")
    ax4.set_ylabel("Score (image → text)")
    ax4.set_title("MRR and mAP@100 per Category  (worst → best by R@1)", fontsize=10)
    ax4.set_ylim(0, 0.85)
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    # ── Plot 5: overall metrics summary bar ───────────────────────────────────
    ax5 = fig.add_subplot(3, 2, 6)
    metrics   = ["R@1", "R@5", "R@10", "R@50", "R@100", "MRR", "mAP@100"]
    i2t_vals  = [overall[f"i2t_{m}"] for m in metrics]
    t2i_vals  = [overall[f"t2i_{m}"] for m in metrics]
    xm = np.arange(len(metrics))
    ax5.bar(xm - 0.2, i2t_vals, 0.4, label="i2t", color="#2980b9", alpha=0.88)
    ax5.bar(xm + 0.2, t2i_vals, 0.4, label="t2i", color="#e67e22", alpha=0.88)
    for i, (a, b) in enumerate(zip(i2t_vals, t2i_vals)):
        ax5.text(i - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7, color="#2980b9")
        ax5.text(i + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=7, color="#e67e22")
    ax5.set_xticks(xm)
    ax5.set_xticklabels(metrics, fontsize=9)
    ax5.set_ylabel("Score")
    ax5.set_title(f"Overall Retrieval Metrics  (n={N:,} test panels)", fontsize=10)
    ax5.set_ylim(0, 0.85)
    ax5.legend(fontsize=9)
    ax5.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(args.output_dir, "insights.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")

    # ── Text summary ─────────────────────────────────────────────────────────
    print(f"\n── Overall ──")
    print(f"  i2t  R@1={overall['i2t_R@1']:.4f}  R@10={overall['i2t_R@10']:.4f}  "
          f"R@100={overall['i2t_R@100']:.4f}  MRR={overall['i2t_MRR']:.4f}  mAP@100={overall['i2t_mAP@100']:.4f}")
    print(f"  t2i  R@1={overall['t2i_R@1']:.4f}  R@10={overall['t2i_R@10']:.4f}  "
          f"R@100={overall['t2i_R@100']:.4f}  MRR={overall['t2i_MRR']:.4f}  mAP@100={overall['t2i_mAP@100']:.4f}")

    print(f"\n── Hardest categories (by i2t R@1) ──")
    for i in sort_idx[:5]:
        print(f"  {cats[i]:<30}  n={ns[i]:>6}  R@1={r1_i2t[i]:.4f}  R@10={r10_i2t[i]:.4f}  R@100={r100_i2t[i]:.4f}")

    print(f"\n── Easiest categories (by i2t R@1) ──")
    for i in sort_idx[::-1][:5]:
        print(f"  {cats[i]:<30}  n={ns[i]:>6}  R@1={r1_i2t[i]:.4f}  R@10={r10_i2t[i]:.4f}  R@100={r100_i2t[i]:.4f}")

    asym = r1_i2t - r1_t2i
    asym_sort = np.argsort(np.abs(asym))[::-1]
    print(f"\n── Most asymmetric i2t vs t2i ──")
    for i in asym_sort[:5]:
        print(f"  {cats[i]:<30}  i2t={r1_i2t[i]:.4f}  t2i={r1_t2i[i]:.4f}  diff={asym[i]:+.4f}")


if __name__ == "__main__":
    main()
