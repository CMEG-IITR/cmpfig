#!/usr/bin/env python3
"""
Convert your annotated data (COCO JSON) to the YOLO format used by MatDetect.

Input layout:
    mydata/
    ├── annotations_coco.json
    └── selected_images/   ← .jpg files

Output layout:
    mydata/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Category mapping:
    COCO id 1-20  (A–T)      → YOLO class 0–19
    COCO id 21    (single)   → YOLO class 0  (A) — whole-image single panel
    COCO id 22    (common)   → SKIPPED
    COCO id 23    (unlabel)  → SKIPPED

COCO bbox: [x_topleft, y_topleft, width, height]  (absolute pixels)
YOLO bbox: [xc, yc, w, h]                         (normalised 0–1, centre)

Usage:
    python convert_mydata.py \
        --json        mydata/annotations_coco.json \
        --images-dir  mydata/selected_images \
        --output-dir  mydata \
        --val-ratio   0.1 \
        --test-ratio  0.1 \
        --seed        42
"""

import os
import json
import shutil
import random
import argparse
from collections import defaultdict, Counter


# ── category mapping ──────────────────────────────────────────────────────────

# COCO category_id (1-based) → YOLO class id (0-based)
# Classes 0-19: A–T  (panel labels)
# Class 20:     single  (whole-figure single panel)
# Class 21:     common  (shared/common element)
# 23 (unlabel) is deliberately omitted → skipped
COCO_TO_YOLO = {i: i - 1 for i in range(1, 21)}   # 1→0, 2→1, ..., 20→19
COCO_TO_YOLO[21] = 20                               # single → 20
COCO_TO_YOLO[22] = 21                               # common → 21

YOLO_ID_TO_LABEL = {i: chr(ord("A") + i) for i in range(20)}
YOLO_ID_TO_LABEL[20] = "single"
YOLO_ID_TO_LABEL[21] = "common"


# ── conversion helpers ────────────────────────────────────────────────────────

def coco_to_yolo_bbox(x, y, w, h, img_w, img_h):
    """COCO [x_tl, y_tl, w, h] → YOLO [xc, yc, w, h] normalised."""
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    # clamp to [0, 1]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    wn = max(0.001, min(1.0, wn))
    hn = max(0.001, min(1.0, hn))
    return xc, yc, wn, hn


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("COCO → YOLO converter for MatDetect")
    p.add_argument("--json",        default="mydata/annotations_coco.json")
    p.add_argument("--images-dir",  default="mydata/selected_images")
    p.add_argument("--output-dir",  default="mydata")
    p.add_argument("--val-ratio",   type=float, default=0.1)
    p.add_argument("--test-ratio",  type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)

    # ── load JSON ─────────────────────────────────────────────────────────────
    print(f"Loading: {args.json}")
    with open(args.json) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}

    # group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # ── process each image ────────────────────────────────────────────────────
    valid_images = []
    yolo_labels  = {}             # stem → list of "cls xc yc w h" strings
    skip_cats    = Counter()
    kept_cats    = Counter()
    bad_box      = 0

    for img_info in coco["images"]:
        img_id   = img_info["id"]
        fname    = img_info["file_name"]
        stem     = os.path.splitext(fname)[0]
        img_w    = img_info["width"]
        img_h    = img_info["height"]

        lines = []
        for ann in anns_by_image[img_id]:
            cat_id = ann["category_id"]

            if cat_id not in COCO_TO_YOLO:
                skip_cats[cat_map.get(cat_id, str(cat_id))] += 1
                continue

            yolo_cls = COCO_TO_YOLO[cat_id]
            x, y, w, h = ann["bbox"]

            if w <= 0 or h <= 0:
                bad_box += 1
                continue

            xc, yc, wn, hn = coco_to_yolo_bbox(x, y, w, h, img_w, img_h)
            lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            kept_cats[cat_map.get(cat_id, str(cat_id))] += 1

        if not lines:
            continue  # skip images with no usable annotations

        # check source image exists
        src = os.path.join(args.images_dir, fname)
        if not os.path.exists(src):
            print(f"  [warn] image not found: {src}")
            continue

        valid_images.append((stem, fname, src))
        yolo_labels[stem] = lines

    print(f"\nImages with usable annotations : {len(valid_images)}")
    print(f"Bad boxes (zero/neg size)       : {bad_box}")
    print(f"\nKept categories:")
    for name, cnt in sorted(kept_cats.items()):
        print(f"  {name:10s}: {cnt:5d}")
    print(f"\nSkipped categories:")
    for name, cnt in sorted(skip_cats.items()):
        print(f"  {name:10s}: {cnt:5d}")

    # ── train/val/test split (80/10/10) ──────────────────────────────────────
    random.shuffle(valid_images)
    n_val   = max(1, int(len(valid_images) * args.val_ratio))
    n_test  = max(1, int(len(valid_images) * args.test_ratio))
    n_train = len(valid_images) - n_val - n_test
    train_images = valid_images[:n_train]
    val_images   = valid_images[n_train:n_train + n_val]
    test_images  = valid_images[n_train + n_val:]
    print(f"\nSplit → train: {len(train_images)}   val: {len(val_images)}   test: {len(test_images)}")

    # ── write output ──────────────────────────────────────────────────────────
    splits = {"train": train_images, "val": val_images, "test": test_images}

    for split, images in splits.items():
        img_out = os.path.join(args.output_dir, split, "images")
        lbl_out = os.path.join(args.output_dir, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for stem, fname, src in images:
            shutil.copy2(src, os.path.join(img_out, fname))
            with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
                f.write("\n".join(yolo_labels[stem]) + "\n")

        print(f"  Wrote {len(images):4d} images → {split}/")

    # ── class distribution in train ───────────────────────────────────────────
    print("\nClass distribution (train set):")
    cls_counts = Counter()
    for stem, _, _ in train_images:
        for line in yolo_labels[stem]:
            cls_counts[int(line.split()[0])] += 1
    for cls_id in sorted(cls_counts):
        label = YOLO_ID_TO_LABEL.get(cls_id, str(cls_id))
        print(f"  {label}: {cls_counts[cls_id]}")

    print(f"\nDone. Output saved to: {os.path.abspath(args.output_dir)}/  (train / val / test)")
    print(f"\nTo train on your data only:")
    print(f"  python train.py --train-dir {args.output_dir}/train --val-dir {args.output_dir}/val \\")
    print(f"      --output ./checkpoints_mydata --epochs 60 --batch-size 8 --patience 12")
    print(f"\nTo merge with existing data (recommended — more samples = better model):")
    print(f"  python merge_datasets.py")


if __name__ == "__main__":
    main()
