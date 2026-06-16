#!/usr/bin/env python3
"""
Build an organized dataset folder from the downloaded materials Arrow dataset.

Output layout:
    ./omni_materials/
        images/          ← copied/linked images
        labels/          ← YOLO-format .txt  (one per image)
        metadata.csv     ← image_name, caption, raw_caption, recaption_model

YOLO label format (one subfigure per line):
    class_id  xc  yc  w  h   (all normalized 0-1)

Class map: panel letter from subfigures_info "legend" field
    a/A → 0,  b/B → 1, ..., t/T → 19
Only subfigures with a clean single-letter legend (a-t) are kept.

Usage:
    python build_dataset.py
    python build_dataset.py --out-dir ./my_dataset
    python build_dataset.py --source-data ./materials_data --images-src ./materials_images
"""

import os
import re
import csv
import json
import shutil
import argparse
from pathlib import Path


# a→0, b→1, ..., t→19  (same as MatDetect training classes)
LETTER2ID = {chr(ord('a') + i): i for i in range(20)}
ID2LETTER  = {v: k for k, v in LETTER2ID.items()}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-data",  default="./materials_data",
                   help="Arrow dataset folder from download_materials.py")
    p.add_argument("--images-src",   default="./materials_images",
                   help="Folder where images were saved by download_materials.py")
    p.add_argument("--out-dir",      default="./omni_materials")
    return p.parse_args()


def bbox_to_yolo(bbox):
    """Convert [x1,y1,x2,y2] normalized → xc,yc,w,h normalized."""
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return xc, yc, w, h


def is_valid_bbox(bbox):
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1


def main():
    args = get_args()

    images_out = os.path.join(args.out_dir, "images")
    labels_out = os.path.join(args.out_dir, "labels")
    csv_out    = os.path.join(args.out_dir, "metadata.csv")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    from datasets import load_from_disk
    ds = load_from_disk(args.source_data)
    print(f"Loaded {len(ds):,} rows from {args.source_data}")

    n_ok      = 0
    n_no_img  = 0
    n_no_lbl  = 0
    class_counts = {}

    with open(csv_out, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=[
            "image_name", "caption", "raw_caption", "recaption_model"
        ])
        writer.writeheader()

        for idx, row in enumerate(ds):
            # ── locate source image ────────────────────────────────────────────
            src_img = row.get("image_file", "")
            if not src_img or not os.path.exists(src_img):
                # fallback: look in images-src by index
                candidate = os.path.join(args.images_src, f"{idx+1:06d}.png")
                if os.path.exists(candidate):
                    src_img = candidate
                else:
                    n_no_img += 1
                    continue

            img_ext   = Path(src_img).suffix or ".png"
            img_name  = f"{idx+1:06d}{img_ext}"
            dst_img   = os.path.join(images_out, img_name)
            lbl_name  = f"{idx+1:06d}.txt"
            dst_lbl   = os.path.join(labels_out, lbl_name)

            # ── parse subfigures_info → YOLO labels ───────────────────────────
            subfigs_raw = row.get("subfigures_info", "[]") or "[]"
            try:
                subfigs = json.loads(subfigs_raw) if isinstance(subfigs_raw, str) else subfigs_raw
            except (json.JSONDecodeError, TypeError):
                subfigs = []

            yolo_lines = []
            for sf in subfigs:
                bbox = sf.get("bbox", [])
                if not is_valid_bbox(bbox):
                    continue
                legend = str(sf.get("legend", "") or "").strip().lower()
                if not re.fullmatch(r"[a-t]", legend):
                    continue                          # skip messy / non-letter labels
                cls_id = LETTER2ID[legend]
                class_counts[legend] = class_counts.get(legend, 0) + 1
                xc, yc, w, h = bbox_to_yolo(bbox)
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if not yolo_lines:
                n_no_lbl += 1
                continue

            # ── copy image ────────────────────────────────────────────────────
            shutil.copy2(src_img, dst_img)

            # ── write label ───────────────────────────────────────────────────
            with open(dst_lbl, "w") as lf:
                lf.write("\n".join(yolo_lines) + "\n")

            # ── write CSV row ─────────────────────────────────────────────────
            writer.writerow({
                "image_name":     img_name,
                "caption":        row.get("caption",        "") or "",
                "raw_caption":    row.get("raw_caption",    "") or "",
                "recaption_model":row.get("recaption_model","") or "",
            })

            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  processed {n_ok} ...", flush=True)

    print(f"\nDone.")
    print(f"  Saved   : {n_ok} images+labels")
    print(f"  No image: {n_no_img} skipped")
    print(f"  No bbox : {n_no_lbl} skipped (subfigures had no valid bbox)")
    print(f"\n  Class counts in labels:")
    for letter in sorted(class_counts.keys()):
        cid = LETTER2ID[letter]
        print(f"    [{cid}] {letter}: {class_counts[letter]}")
    print(f"\n  images/  → {images_out}")
    print(f"  labels/  → {labels_out}")
    print(f"  CSV      → {csv_out}")

    # write classes.txt  (same order as MatDetect: a=0 … t=19)
    classes_path = os.path.join(args.out_dir, "classes.txt")
    with open(classes_path, "w") as f:
        for i in range(20):
            f.write(chr(ord("a") + i) + "\n")
    print(f"  classes.txt → {classes_path}")


if __name__ == "__main__":
    main()
