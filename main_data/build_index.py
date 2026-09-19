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
    },
    "steel": {
        "crops_dir": "./steel_prod_crops",
        "csv":       "steel_prod_dataset.csv",
    },
    "additive": {
        "crops_dir": "./additive_elsevier_crops",
        "csv":       "additive_elsevier_dataset.csv",
    },
}

CSV_COLS = [
    "visualization_category",
    "visualization_subtype",
    "subcaption",
    "summary",
]

# Cross-domain duplicate figures were scraped independently into more than one
# production pipeline (e.g. a composite-materials paper also picked up by the
# ceramics scraper). `merged_duplicates.csv` records, per duplicate figure,
# which production file it was dropped from and which one it was kept in.
# `dropped_from_file` maps to the `domain` whose crops must be excluded here.
DUPLICATES_CSV = "./merged_duplicates.csv"

DROP_FILE_TO_DOMAIN = {
    "composite_prod.csv":        "composite",
    "ni_alloy_production.csv":   "ni_alloy",
    "ceramics_production.csv":   "ceramics",
    "alloy_elsevier_figures_cc_by_with_references.csv": "alloy",
}


# ── helpers ───────────────────────────────────────────────────────────────────


def load_duplicate_image_ids(csv_path):
    """Returns {domain: {image_id, ...}} of figures to exclude as cross-domain dupes."""
    removal = {domain: set() for domain in DROP_FILE_TO_DOMAIN.values()}
    if not csv_path or not os.path.exists(csv_path):
        return removal

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    for dropped_from, group in df.groupby("dropped_from_file"):
        domain = DROP_FILE_TO_DOMAIN.get(dropped_from)
        if domain is None:
            continue
        names = group["downloaded_image_name"].str.strip()
        removal[domain].update(n for n in names if n)
    return removal

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
    p.add_argument("--log", default="./build_index.log",
                   help="Log file path (default: ./build_index.log)")
    p.add_argument("--duplicates-csv", default=DUPLICATES_CSV,
                   help="CSV of cross-domain duplicate figures to exclude "
                        "(default: ./merged_duplicates.csv; pass '' to disable)")
    return p.parse_args()


def main():
    args = get_args()
    records = []
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    dup_removal = load_duplicate_image_ids(args.duplicates_csv)
    for domain, ids in dup_removal.items():
        if ids:
            log(f"[dedup] {domain}: {len(ids)} cross-domain duplicate figures will be excluded")

    total_crop_files = 0
    total_raw_matches = 0
    total_deduped = 0
    total_matched = 0
    total_skipped = 0

    for domain, cfg in DOMAINS.items():
        crops_dir = Path(cfg["crops_dir"])
        if not crops_dir.exists():
            log(f"[skip] {domain} — crops dir not found: {crops_dir}")
            continue

        log(f"\n── {domain} ──────────────────────────")

        # load CSV index if available
        csv_index = {}
        if cfg["csv"] and os.path.exists(cfg["csv"]):
            log(f"  Loading CSV: {cfg['csv']}")
            csv_index = load_csv_index(cfg["csv"])
            log(f"  CSV rows   : {len(csv_index)}")

        # scan crop files
        crop_files = sorted(crops_dir.iterdir())
        log(f"  Crop files : {len(crop_files)}")

        removal_ids = dup_removal.get(domain, set())

        matched   = 0
        unmatched = 0
        deduped   = 0

        for f in tqdm(crop_files, desc=f"  {domain}"):
            if not f.is_file():
                continue

            fname = f.name

            if fname in csv_index:
                meta = csv_index[fname]
                if meta["image_id"] in removal_ids:
                    deduped += 1
                    continue
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

        log(f"  Raw matches: {matched + deduped}")
        log(f"  Deduped out: -{deduped}  (cross-domain duplicate figures)")
        log(f"  Matched    : {matched}")
        log(f"  Skipped    : {unmatched}  (no CSV match)")

        total_crop_files  += len(crop_files)
        total_raw_matches += matched + deduped
        total_deduped     += deduped
        total_matched     += matched
        total_skipped     += unmatched

    # totals — reconciles: crop files = raw matches + skipped; matched = raw matches - deduped
    log("\n── totals ──────────────────────────────")
    log(f"  Crop files : {total_crop_files}")
    log(f"  Raw matches: {total_raw_matches}")
    log(f"  Deduped out: -{total_deduped}  (cross-domain duplicate figures)")
    log(f"  Matched    : {total_matched}")
    log(f"  Skipped    : {total_skipped}  (no CSV match)")

    # write JSON
    log(f"\nTotal records : {len(records)}")
    log(f"Writing → {args.out}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(args.out) / 1e6
    log(f"Done  — {size_mb:.1f} MB")

    with open(args.log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
