"""
Standalone convergence figure for the "Convergence" results section:
training loss vs. epoch (left) and validation RSUM vs. epoch (right),
each on its own single y-axis (no dual-axis panels).

Usage:
    python retrieval/plot_convergence.py
    python retrieval/plot_convergence.py --output_dir retrieval/outputs_no_hardneg_20ep
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

# Nature/paper-style palette
BLUE = "#3B5BA5"      # series-1: training loss
ORANGE = "#E07B39"    # series-2: validation RSUM
BLACK = "#000000"
GRID = "#c9c9c9"
SURFACE = "#ffffff"

STEPS_PER_EPOCH = 1222.5  # 24,450 total steps / 20 epochs (see logs/train.log)


def parse_epoch_loss(log_path):
    """Extract (epoch, avg_loss) from 'Epoch N done — avg_loss=X' lines."""
    epochs, losses = [], []
    pattern = re.compile(r"Epoch (\d+) done — avg_loss=([\d.]+)")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return epochs, losses


def load_val_rsum(eval_dir):
    """Load validation RSUM (R@1+R@5+R@10, both directions) at each eval checkpoint."""
    files = sorted(
        glob.glob(os.path.join(eval_dir, "val_step*.json")),
        key=lambda x: int(re.search(r"val_step(\d+)", x).group(1)),
    )
    steps, rsum = [], []
    for f in files:
        step = int(re.search(r"val_step(\d+)", f).group(1))
        with open(f) as fh:
            ov = json.load(fh)["overall"]
        steps.append(step)
        rsum.append(sum(ov.get(k, 0) for k in (
            "i2t_R@1", "i2t_R@5", "i2t_R@10",
            "t2i_R@1", "t2i_R@5", "t2i_R@10",
        )))
    return steps, rsum


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BLACK)
        ax.spines[spine].set_linewidth(1.5)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=BLACK, labelsize=11, width=1.2)


def bold_ticks(ax):
    """Call after all scale/limit/locator changes are final."""
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_color(BLACK)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output_dir",
        default="/mnt/d/Subham/Compoun_img_01/retrieval/outputs_no_hardneg_20ep",
    )
    args = p.parse_args()

    log_path = os.path.join(args.output_dir, "logs", "train.log")
    eval_dir = os.path.join(args.output_dir, "eval")
    pdf_path = os.path.join(args.output_dir, "convergence.pdf")
    png_path = os.path.join(args.output_dir, "convergence.png")

    epochs, losses = parse_epoch_loss(log_path)
    val_steps, val_rsum = load_val_rsum(eval_dir)
    val_epochs = [s / STEPS_PER_EPOCH for s in val_steps]

    # Wide, print-scale two-panel figure; all text rendered in bold black
    # for maximum legibility when shrunk to column width in the paper.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"wspace": 0.28})
    fig.patch.set_facecolor(SURFACE)

    # ── Panel A: training loss vs epoch ─────────────────────────────
    style_axis(ax1)
    ax1.plot(epochs, losses, color=BLUE, linewidth=2.5, solid_capstyle="round",
              solid_joinstyle="round", zorder=3)
    ax1.scatter(epochs, losses, s=90, color=BLUE, edgecolors=BLACK,
                linewidths=1.2, zorder=4)
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch", color=BLACK, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Training loss (InfoNCE, log scale)", color=BLACK, fontsize=14, fontweight="bold")
    ax1.set_title("(a) Training loss", color=BLACK, fontsize=16, fontweight="bold", loc="center", pad=10)
    ax1.set_xlim(-0.5, 19.5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.annotate(f"{losses[0]:.2f}", (epochs[0], losses[0]),
                 textcoords="offset points", xytext=(8, 10), fontsize=12, fontweight="bold", color=BLACK)
    ax1.annotate(f"{losses[-1]:.3f}", (epochs[-1], losses[-1]),
                 textcoords="offset points", xytext=(-42, 10), fontsize=12, fontweight="bold", color=BLACK)
    bold_ticks(ax1)

    # ── Panel B: validation RSUM vs epoch ───────────────────────────
    style_axis(ax2)
    ax2.plot(val_epochs, val_rsum, color=ORANGE, linewidth=2.5, solid_capstyle="round",
              solid_joinstyle="round", zorder=3)
    ax2.scatter(val_epochs, val_rsum, s=90, color=ORANGE, edgecolors=BLACK,
                linewidths=1.2, zorder=4)
    ax2.set_xlabel("Epoch", color=BLACK, fontsize=14, fontweight="bold")
    ax2.set_ylabel("Validation RSUM (R@1+R@5+R@10, max 6.0)",
                    color=BLACK, fontsize=13, fontweight="bold")
    ax2.set_title("(b) Validation RSUM", color=BLACK, fontsize=16, fontweight="bold", loc="center", pad=10)
    ax2.set_xlim(-0.5, 20.5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.annotate(f"{val_rsum[0]:.2f}", (val_epochs[0], val_rsum[0]),
                 textcoords="offset points", xytext=(10, 8), fontsize=12, fontweight="bold", color=BLACK)
    ax2.annotate(f"{val_rsum[-1]:.2f}", (val_epochs[-1], val_rsum[-1]),
                 textcoords="offset points", xytext=(-36, -20), fontsize=12, fontweight="bold", color=BLACK)
    bold_ticks(ax2)

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=SURFACE)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved -> {pdf_path}")
    print(f"Saved -> {png_path}")


if __name__ == "__main__":
    main()
