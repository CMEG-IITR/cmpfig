#!/usr/bin/env python3
"""
Phase 1 — Build master JSON index from all CSVs + crop folders.

Produces dataset_index.json — metadata only, no images.
Each entry points to a crop file on disk.

Usage:
    python build_index.py
    python build_index.py --out dataset_index.json
"""

import os
import json
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ── domain config ─────────────────────────────────────────────────────────────

DOMAINS = {
    "ni_alloy": {
        "crops_dir": "./ni_prod_crops",
        "csv":       "./ni_alloy_linked_dataset.csv",
    },
    "ceramics": {
        "crops_dir": "./ceramics_prod_crops",
        "csv":       "./ceramics_linked_dataset.csv",
    },
    "composite": {
        "crops_dir": "./composite_prod_crops",
        "csv":       "composite_linked_dataset.csv",
    },
    "alloy": {
        "crops_dir":"./alloy_prod_crops",
        "csv": "alloy_linked_dataset.csv"
    }
}

CSV_COLS = [
    "visualization_category",
    "visualization_subtype",
    "subcaption",
    "summary",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv_index(csv_path):
    """Returns {image_filename: {col: val}} for fast lookup."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    index = {}
    for _, row in df.iterrows():
        if row.get("matched", "").strip().lower() != "true":
            continue
        if not row.get("subcaption", "").strip() or not row.get("summary", "").strip():
            continue
        index[row["image_filename"]] = {
            "image_id":               row.get("image_id",               ""),
            "panel_suffix":           row.get("panel_suffix",           ""),
            "visualization_category": row.get("visualization_category", ""),
            "visualization_subtype":  row.get("visualization_subtype",  ""),
            "subcaption":             row.get("subcaption",             ""),
            "summary":                row.get("summary",                ""),
        }
    return index


def derive_from_filename(filename):
    """
    'composite_img10000_A.jpg' → image_id='composite_img10000', panel_suffix='A'
    Falls back gracefully if pattern doesn't match.
    """
    stem = Path(filename).stem          # 'composite_img10000_A'
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, ""


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("Build master JSON index for dataset upload")
    p.add_argument("--out", default="./dataset_index.json",
                   help="Output JSON path (default: ./dataset_index.json)")
    return p.parse_args()


def main():
    args = get_args()
    records = []

    for domain, cfg in DOMAINS.items():
        crops_dir = Path(cfg["crops_dir"])
        if not crops_dir.exists():
            print(f"[skip] {domain} — crops dir not found: {crops_dir}")
            continue

        print(f"\n── {domain} ──────────────────────────")

        # load CSV index if available
        csv_index = {}
        if cfg["csv"] and os.path.exists(cfg["csv"]):
            print(f"  Loading CSV: {cfg['csv']}")
            csv_index = load_csv_index(cfg["csv"])
            print(f"  CSV rows   : {len(csv_index)}")

        # scan crop files
        crop_files = sorted(crops_dir.iterdir())
        print(f"  Crop files : {len(crop_files)}")

        matched   = 0
        unmatched = 0

        for f in tqdm(crop_files, desc=f"  {domain}"):
            if not f.is_file():
                continue

            fname = f.name

            if fname in csv_index:
                meta = csv_index[fname]
                records.append({
                    "image_path":              str(f),
                    "image_filename":          fname,
                    "image_id":                meta["image_id"],
                    "panel_suffix":            meta["panel_suffix"],
                    "visualization_category":  meta["visualization_category"],
                    "visualization_subtype":   meta["visualization_subtype"],
                    "subcaption":              meta["subcaption"],
                    "summary":                 meta["summary"],
                    "domain":                  domain,
                })
                matched += 1
            else:
                unmatched += 1

        print(f"  Matched    : {matched}")
        print(f"  Skipped    : {unmatched}  (no CSV match)")

    # write JSON
    print(f"\nTotal records : {len(records)}")
    print(f"Writing → {args.out}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Done  — {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
