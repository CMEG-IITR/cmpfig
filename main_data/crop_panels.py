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
    python crop_panels.py --image-dir ./testing --json-dir ./inference_results --output-dir ./crops --workers 64
"""

import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


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
    p.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1),
                   help="Number of parallel worker processes (default: min(64, cpu count))")
    return p.parse_args()


def process_one(jpath, image_dir, output_dir):
    """Process a single JSON file. Returns (n_crops, message_or_None)."""
    with open(jpath) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return 0, None

    jname = os.path.basename(jpath)
    stem = os.path.splitext(data["file"])[0]

    img_path = None
    for ext in IMG_EXTS:
        candidate = os.path.join(image_dir, stem + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        return 0, f"[skip] image not found for {jname}"

    if not data["detections"]:
        return 0, None

    # for each label keep only the highest-confidence detection
    best = {}
    for det in data["detections"]:
        lbl = det["label_name"]
        if lbl not in best or det["score"] > best[lbl]["score"]:
            best[lbl] = det

    pil = Image.open(img_path).convert("RGB")

    n_crops = 0
    for det in best.values():
        label = det["label_name"]
        out_name = f"{stem}_{label}.jpg"
        if label == "single":
            pil.save(os.path.join(output_dir, out_name))
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
            crop = pil.crop((x1, y1, x2, y2))
            crop.save(os.path.join(output_dir, out_name))
        n_crops += 1

    return n_crops, None


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
    print(f"Workers    : {args.workers}")
    print(f"Output dir : {args.output_dir}\n")

    n_crops = 0
    jpaths = [os.path.join(args.json_dir, j) for j in json_files]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(process_one, jp, args.image_dir, args.output_dir)
            for jp in jpaths
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Cropping"):
            count, msg = fut.result()
            n_crops += count
            if msg:
                tqdm.write(msg)

    print(f"\nSaved {n_crops} crops → {args.output_dir}/")


if __name__ == "__main__":
    main()
