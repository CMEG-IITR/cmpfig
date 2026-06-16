#!/usr/bin/env python3
"""
Synthetic compound figure generator for MatDetect.

Pipeline:
  1. Copy all original training images + labels into output-dir (including singles)
  2. Extract ground-truth crops from mydata_all/train/ using YOLO labels
  3. Augment crops for classes below --augment-threshold (flip, brightness, contrast)
  4. Compose synthetic compound figures in randomized grid layouts
  5. Save images + YOLO .txt annotations to --output-dir

The result is a complete, self-contained dataset in output-dir that can replace
mydata_all/train directly in your training config.

Usage:
    python gen_synthetic.py
    python gen_synthetic.py --n-images 1000 --output-dir mydata_all/train_syn
    python gen_synthetic.py --n-images 2000 --augment-threshold 150
    python gen_synthetic.py --skip-copy   # synthetic only, no originals
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K',
    'L','M','N','O','P','Q','R','S','T','single','common'
]
LETTER_CLASSES = list(range(20))  # 0=A … 19=T

# Grid layout options per panel count: list of (rows, cols)
GRID_LAYOUTS = {
    2:  [(1,2),(2,1)],
    3:  [(1,3),(3,1)],
    4:  [(2,2),(1,4),(4,1)],
    5:  [(2,3),(3,2)],
    6:  [(2,3),(3,2)],
    7:  [(3,3),(2,4),(4,2)],
    8:  [(2,4),(4,2),(3,3)],
    9:  [(3,3),(4,3)],
    10: [(2,5),(5,2),(3,4),(4,3)],
    11: [(3,4),(4,3)],
    12: [(3,4),(4,3),(3,5)],
}


def get_args():
    p = argparse.ArgumentParser(description="Synthetic compound figure generator")
    p.add_argument('--train-img-dir', default='mydata_all/train/images',
                   help='Training images folder')
    p.add_argument('--train-lbl-dir', default='mydata_all/train/labels',
                   help='Training YOLO labels folder')
    p.add_argument('--output-dir', default='mydata_all/train_syn',
                   help='Output folder for synthetic images and labels')
    p.add_argument('--n-images', type=int, default=500,
                   help='Number of synthetic figures to generate')
    p.add_argument('--min-panels', type=int, default=2,
                   help='Minimum panels per figure')
    p.add_argument('--max-panels', type=int, default=12,
                   help='Maximum panels per figure')
    p.add_argument('--augment-threshold', type=int, default=150,
                   help='Classes with fewer crops than this get flipped/augmented')
    p.add_argument('--rare-class-ratio', type=float, default=0.3,
                   help='Fraction of images that must include G or beyond (forces longer sequences)')
    p.add_argument('--skip-copy', action='store_true',
                   help='Do not copy original training images; generate synthetic only')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 – Copy originals (including single-panel images)
# ---------------------------------------------------------------------------

def copy_originals(img_dir: Path, lbl_dir: Path,
                   out_img: Path, out_lbl: Path) -> int:
    """
    Copy every image + label from the original train split into out_img/out_lbl.
    Files are prefixed with 'orig_' so they never collide with 'syn_' names.
    Single-panel images (class 20) are included automatically since we copy
    everything without filtering.

    Returns the number of images copied.
    """
    copied = 0
    label_files = sorted(lbl_dir.glob('*.txt'))

    for lbl_src in tqdm(label_files, desc='Copying originals', leave=False):
        stem = lbl_src.stem

        img_src = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            cand = img_dir / (stem + ext)
            if cand.exists():
                img_src = cand
                break
        if img_src is None:
            continue

        shutil.copy2(img_src,  out_img / f'orig_{stem}{img_src.suffix}')
        shutil.copy2(lbl_src,  out_lbl / f'orig_{stem}.txt')
        copied += 1

    return copied


# ---------------------------------------------------------------------------
# Step 2 – Extract GT crops
# ---------------------------------------------------------------------------

def extract_gt_crops(img_dir: Path, lbl_dir: Path) -> dict:
    """Return dict {class_id: [PIL.Image, ...]} from ground-truth YOLO labels."""
    crops = defaultdict(list)
    label_files = sorted(lbl_dir.glob('*.txt'))

    for lbl_file in tqdm(label_files, desc='Extracting GT crops', leave=False):
        stem = lbl_file.stem
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            cand = img_dir / (stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crops[cls].append(img.crop((x1, y1, x2, y2)))

    return crops


# ---------------------------------------------------------------------------
# Step 2 – Augmentation for rare classes
# ---------------------------------------------------------------------------

def augment_pool(crops: dict, threshold: int) -> dict:
    """Expand crop pools below threshold with flips and photometric variants."""
    for cls in LETTER_CLASSES:
        if cls not in crops or len(crops[cls]) >= threshold:
            continue
        originals = list(crops[cls])
        expanded = list(originals)
        for img in originals:
            expanded.append(ImageOps.mirror(img))                                   # H-flip
            expanded.append(ImageOps.flip(img))                                     # V-flip
            expanded.append(ImageOps.mirror(ImageOps.flip(img)))                    # both
            expanded.append(ImageEnhance.Brightness(img).enhance(0.80))
            expanded.append(ImageEnhance.Brightness(img).enhance(1.20))
            expanded.append(ImageEnhance.Contrast(img).enhance(0.80))
            expanded.append(ImageEnhance.Contrast(img).enhance(1.20))
            expanded.append(img.rotate( 5, expand=False, fillcolor=(255,255,255)))
            expanded.append(img.rotate(-5, expand=False, fillcolor=(255,255,255)))
        crops[cls] = expanded
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        print(f'    {name}: {len(originals)} → {len(expanded)} crops after augmentation')
    return crops


# ---------------------------------------------------------------------------
# Step 3 – Composition
# ---------------------------------------------------------------------------

def resize_with_pad(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize keeping aspect ratio, pad remainder with white."""
    img = img.copy()
    img.thumbnail((target_w, target_h), Image.LANCZOS)
    canvas = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    px = (target_w - img.width) // 2
    py = (target_h - img.height) // 2
    canvas.paste(img, (px, py))
    return canvas


def get_layout(n: int):
    """Return a random (rows, cols) for n panels."""
    opts = GRID_LAYOUTS.get(n)
    if opts:
        return random.choice(opts)
    # Fallback for large n
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def compose_figure(panel_crops: list, gap: int, cell_w: int, cell_h: int):
    """
    Place panel_crops in a grid.

    Args:
        panel_crops: [(class_id, PIL.Image), ...]
        gap: pixel gap between and around cells
        cell_w, cell_h: target cell dimensions

    Returns:
        (PIL.Image, [(class_id, x1, y1, x2, y2), ...]) pixel-space bboxes
    """
    n = len(panel_crops)
    rows, cols = get_layout(n)

    canvas_w = cols * cell_w + (cols + 1) * gap
    canvas_h = rows * cell_h + (rows + 1) * gap
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))

    bboxes = []
    for idx, (cls_id, crop) in enumerate(panel_crops):
        row, col = divmod(idx, cols)
        x0 = gap + col * (cell_w + gap)
        y0 = gap + row * (cell_h + gap)
        canvas.paste(resize_with_pad(crop, cell_w, cell_h), (x0, y0))
        bboxes.append((cls_id, x0, y0, x0 + cell_w, y0 + cell_h))

    return canvas, bboxes


# ---------------------------------------------------------------------------
# Step 4 – Sampling
# ---------------------------------------------------------------------------

def max_consecutive_from_A(crops: dict) -> int:
    """How many consecutive classes starting from A=0 have at least one crop."""
    n = 0
    for cls in LETTER_CLASSES:
        if crops.get(cls):
            n = cls + 1
        else:
            break
    return n


def sample_sequence(crops: dict, min_panels: int, max_panels: int,
                    force_length: int = None):
    """
    Return [(class_id, PIL.Image), ...] for a consecutive A-onwards sequence.
    force_length pins the number of panels (used for targeting rare classes).
    """
    max_avail = max_consecutive_from_A(crops)
    if max_avail < min_panels:
        return None

    if force_length is not None:
        n = min(force_length, max_avail)
    else:
        n = random.randint(min_panels, min(max_panels, max_avail))

    if n < min_panels:
        return None

    return [(cls, random.choice(crops[cls])) for cls in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_img = Path(args.output_dir) / 'images'
    out_lbl = Path(args.output_dir) / 'labels'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # ── Step 1 ──────────────────────────────────────────────────────────────
    n_copied = 0
    if not args.skip_copy:
        print('\n[1/4] Copying original training images + labels (incl. singles)...')
        n_copied = copy_originals(
            Path(args.train_img_dir), Path(args.train_lbl_dir),
            out_img, out_lbl,
        )
        print(f'  Copied {n_copied} original images.')
    else:
        print('\n[1/4] Skipping original copy (--skip-copy set).')

    # ── Step 2 ──────────────────────────────────────────────────────────────
    print('\n[2/4] Extracting ground-truth crops from training set...')
    crops = extract_gt_crops(Path(args.train_img_dir), Path(args.train_lbl_dir))
    print('  Crop pool (before augmentation):')
    for cls in sorted(crops):
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        print(f'    {name}: {len(crops[cls])}')

    # ── Step 3 ──────────────────────────────────────────────────────────────
    print(f'\n[3/4] Augmenting classes with < {args.augment_threshold} crops...')
    crops = augment_pool(crops, args.augment_threshold)

    # How deep can we go?
    max_depth = max_consecutive_from_A(crops)
    print(f'  Max consecutive sequence (A→{CLASS_NAMES[max_depth-1]}): {max_depth} panels')

    # ── Step 4 ──────────────────────────────────────────────────────────────
    print(f'\n[4/4] Generating {args.n_images} synthetic figures...')

    # Decide how many figures should be "long" (to cover rare classes G-K)
    n_rare = int(args.n_images * args.rare_class_ratio)
    n_normal = args.n_images - n_rare

    # For rare targets: lengths 7-12 (covers G..K), spread evenly
    rare_lengths = []
    if max_depth >= 7:
        # evenly distribute forced lengths between 7 and max_depth
        for i in range(n_rare):
            rare_lengths.append(random.randint(7, min(12, max_depth)))

    # Build generation queue: (is_rare, force_length)
    queue = [(False, None)] * n_normal + [(True, l) for l in rare_lengths]
    random.shuffle(queue)

    count = 0
    skipped = 0
    class_tally = defaultdict(int)

    for is_rare, force_len in tqdm(queue, desc='Composing figures'):
        panels = sample_sequence(
            crops,
            args.min_panels,
            args.max_panels,
            force_length=force_len,
        )
        if panels is None:
            skipped += 1
            continue

        cell_w = random.randint(120, 320)
        cell_h = random.randint(100, 280)
        gap    = random.randint(2, 12)

        try:
            fig, bboxes = compose_figure(panels, gap, cell_w, cell_h)
        except Exception:
            skipped += 1
            continue

        stem = f'syn_{count:05d}'
        fig.save(out_img / f'{stem}.jpg', quality=95)

        FW, FH = fig.size
        with open(out_lbl / f'{stem}.txt', 'w') as f:
            for cls_id, x1, y1, x2, y2 in bboxes:
                cx = ((x1 + x2) / 2) / FW
                cy = ((y1 + y2) / 2) / FH
                bw = (x2 - x1) / FW
                bh = (y2 - y1) / FH
                f.write(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
                class_tally[cls_id] += 1

        count += 1

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f'\nDone.')
    print(f'  Original images copied : {n_copied}')
    print(f'  Synthetic figures made : {count}  ({skipped} skipped)')
    print(f'  Total in dataset       : {n_copied + count}')
    print(f'  Output: {args.output_dir}/')
    print('\nAnnotation count per class in synthetic set:')
    for cls in sorted(class_tally):
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        print(f'  {name}: {class_tally[cls]}')


if __name__ == '__main__':
    main()
