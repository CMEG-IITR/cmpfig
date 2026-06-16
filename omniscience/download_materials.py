#!/usr/bin/env python3
"""
Download only Materials-science rows from UniParser/OmniScience.

Filters rows where:
  - `subject`     contains "material"  (case-insensitive), OR
  - `raw_subject` contains "material"  (case-insensitive)

Output:
  ./materials_data/         ← HuggingFace Dataset (Arrow format, fast reload)
  ./materials_data.jsonl    ← one JSON record per line (no images)
  ./materials_images/       ← PNG/JPG per row, named by index

Usage:
    python download_materials.py
    python download_materials.py --save-images      # also dump images
    python download_materials.py --max-rows 500     # cap for testing
"""

import os
import json
import argparse
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",   default="./materials_data")
    p.add_argument("--jsonl-out",    default="./materials_data.jsonl")
    p.add_argument("--images-dir",   default="./materials_images")
    p.add_argument("--save-images",  action="store_true",
                   help="Save each row's image to --images-dir")
    p.add_argument("--max-rows",     type=int, default=None,
                   help="Stop after this many matching rows (for testing)")
    return p.parse_args()


def is_materials(row: dict) -> bool:
    subject     = str(row.get("subject",     "") or "").lower()
    raw_subject = str(row.get("raw_subject", "") or "").lower()
    return "material" in subject or "material" in raw_subject


def main():
    args = get_args()

    from datasets import load_dataset

    print("Streaming UniParser/OmniScience (train split) ...")
    ds = load_dataset(
        "UniParser/OmniScience",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    if args.save_images:
        os.makedirs(args.images_dir, exist_ok=True)

    matched = []
    n_seen  = 0
    n_match = 0

    with open(args.jsonl_out, "w") as jf:
        for row in ds:
            n_seen += 1

            if not is_materials(row):
                if n_seen % 5000 == 0:
                    print(f"  scanned {n_seen:,}  matched {n_match:,} ...", flush=True)
                continue

            n_match += 1

            # pull image out before serialising
            img = row.pop("image", None)

            if args.save_images and img is not None:
                img_path = os.path.join(args.images_dir, f"{n_match:06d}.png")
                img.save(img_path)
                row["image_file"] = img_path

            matched.append(row)
            jf.write(json.dumps(row, default=str) + "\n")

            if n_seen % 1000 == 0:
                print(f"  scanned {n_seen:,}  matched {n_match:,} ...", flush=True)

            if args.max_rows and n_match >= args.max_rows:
                print(f"  --max-rows {args.max_rows} reached, stopping early.")
                break

    print(f"\nDone. Scanned {n_seen:,} rows → {n_match:,} materials rows.")
    print(f"JSONL  → {args.jsonl_out}")

    # save as HuggingFace Dataset (Arrow) for fast future loading
    if matched:
        from datasets import Dataset
        hf_ds = Dataset.from_list(matched)
        hf_ds.save_to_disk(args.output_dir)
        print(f"Arrow  → {args.output_dir}/")
        print(f"\nReload later with:")
        print(f"  from datasets import load_from_disk")
        print(f"  ds = load_from_disk('{args.output_dir}')")


if __name__ == "__main__":
    main()
