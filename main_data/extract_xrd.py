#!/usr/bin/env python3
"""Extract XRD-only entries from dataset_index.json, copy their images into a
single folder, and write a new JSON index with updated image_path values."""

import argparse
import json
import os
import shutil


def is_xrd(entry: dict) -> bool:
    subtype = str(entry.get("visualization_subtype", "")).lower()
    category = str(entry.get("visualization_category", "")).lower()
    return "xrd" in subtype or "xrd" in category


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/mnt/d/Subham/Compoun_img_01/main_data/dataset_index.json",
        help="Path to the source dataset_index.json",
    )
    parser.add_argument(
        "--output",
        default="/mnt/d/Subham/Compoun_img_01/main_data/xrd_dataset_index.json",
        help="Path to write the filtered XRD-only JSON",
    )
    parser.add_argument(
        "--base-dir",
        default="/mnt/d/Subham/Compoun_img_01/main_data",
        help="Base directory that image_path values in the input JSON are relative to",
    )
    parser.add_argument(
        "--images-out",
        default="/mnt/d/Subham/Compoun_img_01/main_data/xrd_images",
        help="Folder to copy all matched XRD images into",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    xrd_entries = [entry for entry in data if is_xrd(entry)]

    os.makedirs(args.images_out, exist_ok=True)

    seen_filenames = {}
    missing = 0
    copied = 0
    for entry in xrd_entries:
        src = os.path.join(args.base_dir, entry["image_path"])
        filename = entry["image_filename"]

        if filename in seen_filenames and seen_filenames[filename] != src:
            raise ValueError(f"Filename collision for {filename}: {seen_filenames[filename]} vs {src}")
        seen_filenames[filename] = src

        dst = os.path.join(args.images_out, filename)
        if not os.path.isfile(src):
            missing += 1
            print(f"MISSING SOURCE: {src}")
            continue

        if not os.path.isfile(dst):
            shutil.copy2(src, dst)
        copied += 1

        entry["image_path"] = os.path.join(os.path.basename(args.images_out), filename)

    with open(args.output, "w") as f:
        json.dump(xrd_entries, f, indent=2)

    print(f"Total entries scanned: {len(data)}")
    print(f"XRD entries found: {len(xrd_entries)}")
    print(f"Images copied: {copied}")
    print(f"Images missing: {missing}")
    print(f"Images folder: {args.images_out}")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
