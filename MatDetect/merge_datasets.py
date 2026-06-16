#!/usr/bin/env python3
"""
Merge existing data (data/train + data/val) with your annotated data
(mydata/train + mydata/val) into a combined dataset (data_merged/).

Your 753 real materials-science images are kept in their own val split so
you can measure in-domain performance separately.

Output:
    data_merged/
    ├── train/          ← existing train + mydata train
    │   ├── images/
    │   └── labels/
    └── val/            ← mydata val only  (clean in-domain val)
        ├── images/
        └── labels/

Usage:
    python merge_datasets.py
    # then train:
    python train.py --train-dir ./data_merged/train --val-dir ./data_merged/val \
        --output ./checkpoints_merged --epochs 60 --batch-size 8 --patience 12
"""

import os
import shutil
import glob
from collections import Counter


def copy_split(src_dir: str, dst_dir: str, prefix: str = ""):
    """Copy all images+labels from src_dir to dst_dir, renaming with prefix to avoid collisions."""
    img_src = os.path.join(src_dir, "images")
    lbl_src = os.path.join(src_dir, "labels")
    img_dst = os.path.join(dst_dir, "images")
    lbl_dst = os.path.join(dst_dir, "labels")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    n_copied = 0
    for img_path in glob.glob(os.path.join(img_src, "*")):
        ext  = os.path.splitext(img_path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        new_stem = prefix + stem

        lbl_path = os.path.join(lbl_src, stem + ".txt")
        if not os.path.exists(lbl_path):
            continue

        shutil.copy2(img_path, os.path.join(img_dst, new_stem + ext))
        shutil.copy2(lbl_path, os.path.join(lbl_dst, new_stem + ".txt"))
        n_copied += 1

    return n_copied


def count_classes(split_dir: str):
    counts = Counter()
    for lbl in glob.glob(os.path.join(split_dir, "labels", "*.txt")):
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    return counts


def main():
    base       = os.path.dirname(os.path.abspath(__file__))
    existing   = os.path.join(base, "data")
    mydata     = os.path.join(base, "mydata")
    output     = os.path.join(base, "data_merged")

    # sanity checks
    for path in [existing, mydata]:
        if not os.path.isdir(path):
            print(f"[error] directory not found: {path}")
            print("  Run convert_mydata.py first to generate mydata/train and mydata/val")
            return

    mydata_train = os.path.join(mydata, "train")
    mydata_val   = os.path.join(mydata, "val")
    if not os.path.isdir(mydata_train):
        print("[error] mydata/train not found — run convert_mydata.py first")
        return

    print("Merging datasets...\n")

    # ── merged train = existing train + mydata train ─────────────────────────
    merged_train = os.path.join(output, "train")
    n1 = copy_split(os.path.join(existing, "train"), merged_train, prefix="ex_")
    n2 = copy_split(mydata_train,                    merged_train, prefix="my_")
    print(f"Train: {n1} existing  +  {n2} mydata  =  {n1+n2} total")

    # ── merged val = mydata val only (in-domain) ─────────────────────────────
    merged_val = os.path.join(output, "val")
    n3 = copy_split(mydata_val, merged_val, prefix="my_")
    print(f"Val  : {n3} mydata images (in-domain only)")

    # ── stats ─────────────────────────────────────────────────────────────────
    cls_train = count_classes(merged_train)
    cls_val   = count_classes(merged_val)
    LABEL     = {i: chr(ord("A") + i) for i in range(26)}

    print("\nClass distribution in merged train:")
    for cls_id in sorted(cls_train):
        print(f"  {LABEL.get(cls_id, str(cls_id))}: {cls_train[cls_id]}")

    print("\nClass distribution in merged val (in-domain):")
    for cls_id in sorted(cls_val):
        print(f"  {LABEL.get(cls_id, str(cls_id))}: {cls_val[cls_id]}")

    print(f"\nDone. Merged dataset at: {output}")
    print("\nNext step — train on merged data:")
    print(f"  python train.py \\")
    print(f"      --train-dir {output}/train \\")
    print(f"      --val-dir   {output}/val \\")
    print(f"      --output    ./checkpoints_merged \\")
    print(f"      --epochs    60 --batch-size 8 --patience 12")


if __name__ == "__main__":
    main()
