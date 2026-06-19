"""
Plot training loss and validation Recall@100 curves from a completed run.

Usage:
    python retrieval/plot_curves.py
    python retrieval/plot_curves.py --output_dir retrieval/outputs_full
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_train_log(log_path):
    """Extract (step, loss, lr, tau) from train.log."""
    steps, losses, lrs, taus = [], [], [], []
    pattern = re.compile(
        r"step=(\d+)\s+loss=([\d.]+)\s+tau=([\d.]+)\s+lr=([\d.eE+\-]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                taus.append(float(m.group(3)))
                lrs.append(float(m.group(4)))
    return steps, losses, lrs, taus


def load_val_history(eval_dir):
    """Load val Recall@K from all val_step*.json files."""
    files = sorted(
        glob.glob(os.path.join(eval_dir, "val_step*.json")),
        key=lambda x: int(re.search(r"val_step(\d+)", x).group(1)),
    )
    steps, r10, r50, r100, t2i_r100, rsum = [], [], [], [], [], []
    for f in files:
        step = int(re.search(r"val_step(\d+)", f).group(1))
        with open(f) as fh:
            ov = json.load(fh)["overall"]
        steps.append(step)
        r10.append(ov.get("i2t_R@10", 0))
        r50.append(ov.get("i2t_R@50", 0))
        r100.append(ov.get("i2t_R@100", 0))
        t2i_r100.append(ov.get("t2i_R@100", 0))
        rsum.append(sum(ov.get(k, 0) for k in (
            "i2t_R@1", "i2t_R@5", "i2t_R@10",
            "t2i_R@1", "t2i_R@5", "t2i_R@10",
        )))
    return steps, r10, r50, r100, t2i_r100, rsum


def smooth(values, window=10):
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output_dir",
        default="/mnt/d/Subham/Compoun_img_01/retrieval/outputs",
    )
    p.add_argument("--smooth_window", type=int, default=15,
                   help="Moving-average window for train loss (0 = off)")
    args = p.parse_args()

    log_path  = os.path.join(args.output_dir, "logs", "train.log")
    eval_dir  = os.path.join(args.output_dir, "eval")
    save_path = os.path.join(args.output_dir, "training_curves.png")

    # ── Parse ────────────────────────────────────────────────────────────────
    train_steps, train_loss, lrs, taus = parse_train_log(log_path)
    val_steps, v_r10, v_r50, v_r100, v_t2i, v_rsum = load_val_history(eval_dir)

    best_idx  = int(np.argmax(v_rsum))
    best_step = val_steps[best_idx]
    best_val  = v_rsum[best_idx]

    print(f"Train loss entries : {len(train_steps)}")
    print(f"Val checkpoints    : {len(val_steps)}")
    print(f"Best val RSUM      : {best_val:.4f} at step {best_step}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Diagnostics", fontsize=14, fontweight="bold")

    # ── Panel 1: Train Loss ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(train_steps, train_loss, alpha=0.25, color="steelblue", linewidth=0.8, label="raw")
    if args.smooth_window > 1:
        smoothed = smooth(train_loss, args.smooth_window)
        ax.plot(train_steps, smoothed, color="steelblue", linewidth=1.8, label=f"MA-{args.smooth_window}")
    ax.set_xlabel("Step")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Val Recall curves ────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(val_steps, v_r10,  "o-", ms=3, color="coral",     label="i2t R@10")
    ax.plot(val_steps, v_r50,  "s-", ms=3, color="goldenrod",  label="i2t R@50")
    ax.plot(val_steps, v_r100, "^-", ms=3, color="steelblue",  label="i2t R@100")
    ax.plot(val_steps, v_t2i,  "v--",ms=3, color="seagreen",   label="t2i R@100")
    ax.axvline(best_step, color="red", linestyle="--", linewidth=1, label=f"best RSUM ({best_step})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Recall")
    ax.set_title("Validation Recall@K (image→text & text→image)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Panel 3: Loss + R@100 on shared x-axis ────────────────────────────
    ax  = axes[1, 0]
    ax2 = ax.twinx()
    lns1 = ax.plot(train_steps, smooth(train_loss, args.smooth_window) if args.smooth_window > 1 else train_loss,
                   color="steelblue", linewidth=1.5, label="train loss")
    lns2 = ax2.plot(val_steps, v_rsum, "^-", ms=4, color="tomato", linewidth=1.5, label="val RSUM")
    ax.axvline(best_step, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss", color="steelblue")
    ax2.set_ylabel("RSUM (max 6.0)", color="tomato")
    ax.set_title("Loss vs RSUM (dual axis)")
    lns = lns1 + lns2
    ax.legend(lns, [l.get_label() for l in lns], fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Temperature + LR ─────────────────────────────────────────
    ax  = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(train_steps, taus, color="purple", linewidth=1.5, label="tau (temp)")
    ax2.plot(train_steps, lrs, color="darkorange", linewidth=1.2, linestyle="--", label="LR")
    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature τ", color="purple")
    ax2.set_ylabel("Learning Rate", color="darkorange")
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_title("Temperature & Learning Rate Schedule")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()
