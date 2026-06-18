#!/usr/bin/env python3
"""
Generate training comparison plots for YOLO12m Baseline vs YOLO12m + Uniqueness Loss.

Reads:
  - ../ModelBench/runs/detect/runs_mydata/yolo12m/results.csv        (baseline)
  - ../ModelBench/runs/detect/runs_mydata/yolo12m_unique/results.csv  (unique)
  - results/yolo12m_baseline_classnms_quality.json
  - results/yolo12m_unique_classnms_quality.json

Outputs (saved to ./plots/):
  01_map50_progression.png          — mAP@50 over epochs
  02_map5095_progression.png        — mAP@50:95 over epochs
  03_train_losses.png               — box / cls / dfl training losses
  04_val_losses.png                 — validation losses
  05_precision_recall.png           — P & R curves over epochs
  06_classnms_overall.png           — TP/FP/FN + P/R/F1 bar chart
  07_classnms_perclass_f1.png       — per-class F1 grouped bars
  08_classnms_perclass_tpfpfn.png   — per-class TP/FP/FN stacked
  09_summary_dashboard.png          — 2×3 summary grid

Usage:
    python plot_training.py
    python plot_training.py --out-dir ./my_plots
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
BENCH_RUNS   = os.path.join(HERE, "../ModelBench/runs/detect/runs_mydata")
RESULTS_DIR  = os.path.join(HERE, "results")

CSV_BASE     = os.path.join(BENCH_RUNS, "yolo12m/results.csv")
CSV_UNIQ     = os.path.join(BENCH_RUNS, "yolo12m_unique/results.csv")
JSON_BASE    = os.path.join(RESULTS_DIR, "yolo12m_baseline_classnms_quality.json")
JSON_UNIQ    = os.path.join(RESULTS_DIR, "yolo12m_unique_classnms_quality.json")

# ── Style ─────────────────────────────────────────────────────────────────────

CLR_BASE  = "#2196F3"   # blue  — baseline
CLR_UNIQ  = "#F44336"   # red   — uniqueness
CLR_BASE2 = "#90CAF9"   # light blue
CLR_UNIQ2 = "#EF9A9A"   # light red

plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "legend.framealpha":0.9,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})

VALID_LABELS = list("ABCDEFGHIJK") + ["single"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p}")


def per_class_f1(record):
    pc = record.get("per_class", {})
    f1s = {}
    for cls, m in pc.items():
        if cls not in VALID_LABELS:
            continue
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1s[cls] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return f1s


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_map50(base, uniq, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = base["epoch"]

    ax.plot(epochs, base["metrics/mAP50(B)"], color=CLR_BASE, lw=2, label="YOLO12m Baseline")
    ax.plot(epochs, uniq["metrics/mAP50(B)"], color=CLR_UNIQ,  lw=2, label="YOLO12m + UniqLoss")
    ax.axvline(5, color="gray", lw=1, ls=":", label="Warmup end (epoch 5)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50")
    ax.set_title("Validation mAP@50 — Baseline vs Uniqueness Loss")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    save(fig, "01_map50_progression.png", out_dir)


def plot_map5095(base, uniq, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = base["epoch"]

    ax.plot(epochs, base["metrics/mAP50-95(B)"], color=CLR_BASE, lw=2, label="YOLO12m Baseline")
    ax.plot(epochs, uniq["metrics/mAP50-95(B)"], color=CLR_UNIQ,  lw=2, label="YOLO12m + UniqLoss")
    ax.axvline(5, color="gray", lw=1, ls=":", label="Warmup end (epoch 5)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50:95")
    ax.set_title("Validation mAP@50:95 — Baseline vs Uniqueness Loss")
    ax.legend()
    ax.set_ylim(0.5, 0.95)
    save(fig, "02_map5095_progression.png", out_dir)


def plot_train_losses(base, uniq, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    epochs = base["epoch"]

    for ax, col, label in zip(
        axes,
        ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
        ["Box Loss", "Class Loss", "DFL Loss"],
    ):
        ax.plot(epochs, base[col], color=CLR_BASE, lw=2, label="Baseline")
        ax.plot(epochs, uniq[col], color=CLR_UNIQ,  lw=2, label="+UniqLoss")
        ax.axvline(5, color="gray", lw=1, ls=":", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Train {label}")
        ax.legend(fontsize=8)

    fig.suptitle("Training Losses — Baseline vs Uniqueness Loss", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "03_train_losses.png", out_dir)


def plot_val_losses(base, uniq, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    epochs = base["epoch"]

    for ax, col, label in zip(
        axes,
        ["val/box_loss", "val/cls_loss", "val/dfl_loss"],
        ["Box Loss", "Class Loss", "DFL Loss"],
    ):
        ax.plot(epochs, base[col], color=CLR_BASE, lw=2, label="Baseline")
        ax.plot(epochs, uniq[col], color=CLR_UNIQ,  lw=2, label="+UniqLoss")
        ax.axvline(5, color="gray", lw=1, ls=":", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Val {label}")
        ax.legend(fontsize=8)

    fig.suptitle("Validation Losses — Baseline vs Uniqueness Loss", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "04_val_losses.png", out_dir)


def plot_prec_recall(base, uniq, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = base["epoch"]

    for ax, col, label in zip(
        axes,
        ["metrics/precision(B)", "metrics/recall(B)"],
        ["Precision", "Recall"],
    ):
        ax.plot(epochs, base[col], color=CLR_BASE, lw=2, label="Baseline")
        ax.plot(epochs, uniq[col], color=CLR_UNIQ,  lw=2, label="+UniqLoss")
        ax.axvline(5, color="gray", lw=1, ls=":", alpha=0.7, label="Warmup end")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Validation {label}")
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=9)

    fig.suptitle("Precision & Recall over Training — Standard NMS", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "05_precision_recall.png", out_dir)


def plot_classnms_overall(jb, ju, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: TP / FP / FN counts
    ax = axes[0]
    cats   = ["TP (correct)", "FP (wrong box)", "FN (missed)"]
    bvals  = [jb["tp"], jb["fp"], jb["fn"]]
    uvals  = [ju["tp"], ju["fp"], ju["fn"]]
    x      = np.arange(len(cats))
    w      = 0.35

    bars_b = ax.bar(x - w/2, bvals, w, color=CLR_BASE, label="Baseline")
    bars_u = ax.bar(x + w/2, uvals, w, color=CLR_UNIQ,  label="+UniqLoss")

    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    for bar in bars_u:
        delta = int(bar.get_height()) - bvals[list(bars_u).index(bar)]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{int(bar.get_height())}\n({'+' if delta>=0 else ''}{delta})",
                ha="center", va="bottom", fontsize=8, color=CLR_UNIQ)

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Count")
    ax.set_title("Detection Counts (Class-Aware NMS)")
    ax.legend()

    # Right: Precision / Recall / F1
    ax = axes[1]
    metrics = ["Precision", "Recall", "F1"]
    bm = [jb["precision"], jb["recall"], jb["f1"]]
    um = [ju["precision"], ju["recall"], ju["f1"]]
    x  = np.arange(len(metrics))

    bars_b = ax.bar(x - w/2, bm, w, color=CLR_BASE, label="Baseline")
    bars_u = ax.bar(x + w/2, um, w, color=CLR_UNIQ,  label="+UniqLoss")

    for bar, val in zip(bars_b, bm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val*100:.2f}%", ha="center", va="bottom", fontsize=9)
    for bar, val, bval in zip(bars_u, um, bm):
        delta = (val - bval) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val*100:.2f}%\n({'+' if delta>=0 else ''}{delta:.1f}pp)",
                ha="center", va="bottom", fontsize=8, color=CLR_UNIQ)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.7, 1.08)
    ax.set_title("Overall Metrics (Class-Aware NMS, 281 test images)")
    ax.legend()

    fig.suptitle("Class-Aware NMS Evaluation — Baseline vs Uniqueness Loss", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "06_classnms_overall.png", out_dir)


def plot_perclass_f1(jb, ju, out_dir):
    f1b = per_class_f1(jb)
    f1u = per_class_f1(ju)
    classes = [c for c in VALID_LABELS if c in f1b or c in f1u]

    bvals = [f1b.get(c, 0) for c in classes]
    uvals = [f1u.get(c, 0) for c in classes]
    delta = [u - b for u, b in zip(uvals, bvals)]

    x = np.arange(len(classes))
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # Top: grouped bars
    ax = axes[0]
    ax.bar(x - w/2, bvals, w, color=CLR_BASE, label="Baseline")
    ax.bar(x + w/2, uvals, w, color=CLR_UNIQ,  label="+UniqLoss")
    for i, (b, u) in enumerate(zip(bvals, uvals)):
        ax.text(i - w/2, b + 0.005, f"{b:.3f}", ha="center", va="bottom", fontsize=7, color=CLR_BASE)
        ax.text(i + w/2, u + 0.005, f"{u:.3f}", ha="center", va="bottom", fontsize=7, color=CLR_UNIQ)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.4, 1.1)
    ax.set_title("Per-Class F1 Score (Class-Aware NMS)")
    ax.legend()

    # Bottom: delta bars
    ax = axes[1]
    colors = [CLR_UNIQ if d >= 0 else CLR_BASE for d in delta]
    bars = ax.bar(x, delta, color=colors, edgecolor="white", lw=0.5)
    for bar, d in zip(bars, delta):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.002 if d >= 0 else -0.006),
                f"{'+' if d>=0 else ''}{d:.3f}",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1 Gain (UniqLoss − Baseline)")
    ax.set_title("Per-Class F1 Improvement")

    fig.suptitle("Per-Class F1 — Baseline vs Uniqueness Loss", fontsize=13)
    fig.tight_layout()
    save(fig, "07_classnms_perclass_f1.png", out_dir)


def plot_perclass_tpfpfn(jb, ju, out_dir):
    classes = VALID_LABELS
    pcb = jb.get("per_class", {})
    pcu = ju.get("per_class", {})

    tpb = [pcb.get(c, {}).get("tp", 0) for c in classes]
    fpb = [pcb.get(c, {}).get("fp", 0) for c in classes]
    fnb = [pcb.get(c, {}).get("fn", 0) for c in classes]
    tpu = [pcu.get(c, {}).get("tp", 0) for c in classes]
    fpu = [pcu.get(c, {}).get("fp", 0) for c in classes]
    fnu = [pcu.get(c, {}).get("fn", 0) for c in classes]

    x = np.arange(len(classes))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False)

    for ax, tp, fp, fn, title in [
        (axes[0], tpb, fpb, fnb, "Baseline"),
        (axes[1], tpu, fpu, fnu, "+UniqLoss"),
    ]:
        ax.bar(x, tp, color="#4CAF50", label="TP")
        ax.bar(x, fp, bottom=tp, color="#F44336", label="FP")
        ax.bar(x, fn, bottom=[t+f for t, f in zip(tp, fp)], color="#FF9800", label="FN")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title(f"TP / FP / FN per Class — {title}")
        ax.legend(fontsize=9)

    fig.suptitle("Per-Class Detection Breakdown (Class-Aware NMS)", fontsize=13)
    fig.tight_layout()
    save(fig, "08_classnms_perclass_tpfpfn.png", out_dir)


def plot_dashboard(base, uniq, jb, ju, out_dir):
    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    epochs = base["epoch"]

    # 1. mAP@50
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, base["metrics/mAP50(B)"], color=CLR_BASE, lw=2, label="Baseline")
    ax.plot(epochs, uniq["metrics/mAP50(B)"], color=CLR_UNIQ,  lw=2, label="+UniqLoss")
    ax.axvline(5, color="gray", lw=1, ls=":")
    ax.set_title("mAP@50"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)

    # 2. Precision
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, base["metrics/precision(B)"], color=CLR_BASE, lw=2, label="Baseline")
    ax.plot(epochs, uniq["metrics/precision(B)"], color=CLR_UNIQ,  lw=2, label="+UniqLoss")
    ax.axvline(5, color="gray", lw=1, ls=":")
    ax.set_title("Precision (std NMS)"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)

    # 3. Recall
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, base["metrics/recall(B)"], color=CLR_BASE, lw=2, label="Baseline")
    ax.plot(epochs, uniq["metrics/recall(B)"], color=CLR_UNIQ,  lw=2, label="+UniqLoss")
    ax.axvline(5, color="gray", lw=1, ls=":")
    ax.set_title("Recall (std NMS)"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)

    # 4. Train cls loss
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, base["train/cls_loss"], color=CLR_BASE, lw=2, label="Baseline")
    ax.plot(epochs, uniq["train/cls_loss"], color=CLR_UNIQ,  lw=2, label="+UniqLoss")
    ax.axvline(5, color="gray", lw=1, ls=":")
    ax.set_title("Train Class Loss"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)

    # 5. P/R/F1 bar (class-singleton suppression)
    ax = fig.add_subplot(gs[1, 1])
    metrics = ["Precision", "Recall", "F1"]
    bm = [jb["precision"], jb["recall"], jb["f1"]]
    um = [ju["precision"], ju["recall"], ju["f1"]]
    x  = np.arange(3)
    ax.bar(x - 0.2, bm, 0.38, color=CLR_BASE, label="Baseline")
    ax.bar(x + 0.2, um, 0.38, color=CLR_UNIQ,  label="+UniqLoss")
    for i, (b, u) in enumerate(zip(bm, um)):
        ax.text(i - 0.2, b + 0.005, f"{b*100:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, u + 0.005, f"{u*100:.1f}%", ha="center", va="bottom", fontsize=7, color=CLR_UNIQ)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0.7, 1.08)
    ax.set_title("Class-Singleton Suppression: P / R / F1"); ax.legend(fontsize=8)

    # 6. Per-class F1 gain
    ax = fig.add_subplot(gs[1, 2])
    f1b = per_class_f1(jb)
    f1u = per_class_f1(ju)
    classes = [c for c in VALID_LABELS if c in f1b or c in f1u]
    delta = [f1u.get(c, 0) - f1b.get(c, 0) for c in classes]
    xi    = np.arange(len(classes))
    colors = [CLR_UNIQ if d >= 0 else CLR_BASE for d in delta]
    ax.bar(xi, delta, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xi); ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylabel("F1 Gain"); ax.set_title("Per-Class F1 Gain (+UniqLoss)")

    fig.suptitle(
        "YOLO12m — Baseline vs Uniqueness Loss  |  Training & Class-Singleton Suppression Summary",
        fontsize=14, fontweight="bold",
    )
    save(fig, "09_summary_dashboard.png", out_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("Plot uniqueness loss training comparison")
    p.add_argument("--out-dir", default=os.path.join(HERE, "plots"))
    p.add_argument(
        "--csv-base", default=CSV_BASE,
        help="Path to baseline results.csv"
    )
    p.add_argument(
        "--csv-uniq", default=CSV_UNIQ,
        help="Path to uniqueness results.csv"
    )
    p.add_argument(
        "--json-base", default=JSON_BASE,
        help="Path to baseline class-singleton suppression JSON"
    )
    p.add_argument(
        "--json-uniq", default=JSON_UNIQ,
        help="Path to uniqueness class-singleton suppression JSON"
    )
    return p.parse_args()


def main():
    args = get_args()

    print("Loading data...")
    for path, label in [
        (args.csv_base,  "Baseline CSV"),
        (args.csv_uniq,  "Unique CSV"),
        (args.json_base, "Baseline JSON"),
        (args.json_uniq, "Unique JSON"),
    ]:
        if not os.path.exists(path):
            print(f"  [ERROR] Not found: {path}")
            return
        print(f"  OK  {label}: {path}")

    base = load_csv(args.csv_base)
    uniq = load_csv(args.csv_uniq)
    jb   = load_json(args.json_base)
    ju   = load_json(args.json_uniq)

    print(f"\nGenerating plots → {args.out_dir}/")
    plot_map50(base, uniq, args.out_dir)
    plot_map5095(base, uniq, args.out_dir)
    plot_train_losses(base, uniq, args.out_dir)
    plot_val_losses(base, uniq, args.out_dir)
    plot_prec_recall(base, uniq, args.out_dir)
    plot_classnms_overall(jb, ju, args.out_dir)
    plot_perclass_f1(jb, ju, args.out_dir)
    plot_perclass_tpfpfn(jb, ju, args.out_dir)
    plot_dashboard(base, uniq, jb, ju, args.out_dir)

    print(f"\nDone — {len(os.listdir(args.out_dir))} plots in {args.out_dir}/")


if __name__ == "__main__":
    main()
