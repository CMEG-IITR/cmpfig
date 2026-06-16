#!/usr/bin/env python3
"""
Crop detected panels from images using the JSON files produced by infer.py.

For each image, reads its JSON, crops every detected bounding box, and saves
the crop as <stem>_<label>.jpg in the output folder.

Example output:
    img2498_A.jpg
    img2498_B.jpg
    img2499_A.jpg

Usage:
    python crop_panels.py --image-dir ./testing --json-dir ./inference_results --output-dir ./crops
"""

import os
import json
import argparse

from PIL import Image
from tqdm import tqdm


def get_args():
    p = argparse.ArgumentParser("Crop detected panels from images")
    p.add_argument("--image-dir",  required=True,
                   help="Folder containing the original images")
    p.add_argument("--json-dir",   required=True,
                   help="Folder containing the JSON files from infer.py")
    p.add_argument("--output-dir", required=True,
                   help="Folder where cropped panels will be saved")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of images to process (default: all)")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted(
        f for f in os.listdir(args.json_dir) if f.endswith(".json")
    )
    if not json_files:
        raise RuntimeError(f"No JSON files found in: {args.json_dir}")

    if args.limit:
        json_files = json_files[:args.limit]

    print(f"JSON files : {len(json_files)}")
    print(f"Output dir : {args.output_dir}\n")

    n_crops = 0

    for jname in tqdm(json_files, desc="Cropping"):
        with open(os.path.join(args.json_dir, jname)) as f:
            data = json.load(f)

        stem = os.path.splitext(data["file"])[0]

        # find the image file
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
            candidate = os.path.join(args.image_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"[skip] image not found for {jname}")
            continue

        if not data["detections"]:
            continue

        pil = Image.open(img_path).convert("RGB")

        # track label counts to handle duplicate labels (A, A, B → A_1, A_2, B)
        label_count = {}

        for det in data["detections"]:
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
            label = det["label_name"]

            label_count[label] = label_count.get(label, 0) + 1
            suffix = f"_{label_count[label]}" if label_count[label] > 1 else ""

            crop = pil.crop((x1, y1, x2, y2))
            out_name = f"{stem}_{label}{suffix}.jpg"
            crop.save(os.path.join(args.output_dir, out_name))
            n_crops += 1

    print(f"\nSaved {n_crops} crops → {args.output_dir}/")


if __name__ == "__main__":
    main()
