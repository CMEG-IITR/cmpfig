#!/usr/bin/env python3
"""
Merge all COCO JSON annotation files in my_annotated_data/ into one.

Discovers every *_annotations_coco.json and its matching *_selected_images/
folder automatically, so no arguments are needed.

Output:
    my_annotated_data/merged/
    ├── merged_coco.json     ← single merged COCO annotation file
    └── images/              ← all images, prefixed to avoid name collisions

Usage:
    python merge_coco_annotations.py

After this, convert to YOLO format with:
    python convert_mydata.py \
        --json       my_annotated_data/merged/merged_coco.json \
        --images-dir my_annotated_data/merged/images \
        --output-dir mydata_all \
        --val-ratio  0.2
"""

import os
import json
import shutil
import glob

BASE      = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, "my_annotated_data")
OUTPUT    = os.path.join(INPUT_DIR, "merged")
IMG_OUT   = os.path.join(OUTPUT, "images")
JSON_OUT  = os.path.join(OUTPUT, "merged_coco.json")


def dataset_prefix(json_path: str) -> str:
    """alloy_annotations_coco.json → 'alloy'"""
    name = os.path.basename(json_path)
    return name.replace("_annotations_coco.json", "")


def find_image_dir(prefix: str) -> str:
    candidate = os.path.join(INPUT_DIR, f"{prefix}_selected_images")
    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"Image folder not found: {candidate}")
    return candidate


def main():
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_annotations_coco.json")))
    if not json_files:
        print(f"[error] No *_annotations_coco.json files found in {INPUT_DIR}")
        return

    os.makedirs(IMG_OUT, exist_ok=True)

    merged_images      = []
    merged_annotations = []
    categories         = None   # taken from the first file, same across all

    img_id  = 1   # global unique image id counter
    ann_id  = 1   # global unique annotation id counter
    missing = 0
    copied  = 0

    for json_path in json_files:
        prefix   = dataset_prefix(json_path)
        img_dir  = find_image_dir(prefix)

        with open(json_path) as f:
            data = json.load(f)

        if categories is None:
            categories = data["categories"]

        print(f"\n[{prefix}]  images={len(data['images'])}  "
              f"annotations={len(data['annotations'])}")

        # map old image id → new image id
        old_to_new_img = {}

        for img in data["images"]:
            old_id   = img["id"]
            new_name = f"{prefix}_{img['file_name']}"

            src = os.path.join(img_dir, img["file_name"])
            dst = os.path.join(IMG_OUT, new_name)

            if not os.path.exists(src):
                print(f"  [warn] missing image: {src}")
                missing += 1
                continue

            shutil.copy2(src, dst)
            copied += 1

            old_to_new_img[old_id] = img_id
            merged_images.append({
                "id":        img_id,
                "file_name": new_name,
                "width":     img["width"],
                "height":    img["height"],
            })
            img_id += 1

        for ann in data["annotations"]:
            if ann["image_id"] not in old_to_new_img:
                continue   # image was skipped (missing file)

            merged_annotations.append({
                "id":          ann_id,
                "image_id":    old_to_new_img[ann["image_id"]],
                "category_id": ann["category_id"],
                "bbox":        ann["bbox"],
                "area":        ann.get("area", 0),
                "iscrowd":     ann.get("iscrowd", 0),
            })
            ann_id += 1

    merged = {
        "images":      merged_images,
        "annotations": merged_annotations,
        "categories":  categories,
    }

    with open(JSON_OUT, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Merged {len(json_files)} datasets")
    print(f"  Total images     : {len(merged_images)}  (skipped {missing} missing)")
    print(f"  Total annotations: {len(merged_annotations)}")
    print(f"  Output JSON      : {JSON_OUT}")
    print(f"  Output images    : {IMG_OUT}/  ({copied} files)")
    print(f"\nNext — convert to YOLO format:")
    print(f"  python convert_mydata.py \\")
    print(f"      --json       {JSON_OUT} \\")
    print(f"      --images-dir {IMG_OUT} \\")
    print(f"      --output-dir mydata_all \\")
    print(f"      --val-ratio  0.2")


if __name__ == "__main__":
    main()
