"""
prepare_data.py
---------------
Loads all (image, subcaption) pairs from your annotation JSONs,
matches them to panel image files, and produces stratified
train / val / test splits saved as HuggingFace datasets.

Expected folder layout (two supported conventions):
  Convention A — flat:
    data_root/
      img100.json
      img100_A.png   (or img100_a.png)
      img162.json
      img162_a.png
      ...

  Convention B — separated:
    data_root/
      annotations/img100.json ...
      images/img100_A.png ...

Usage:
  python prepare_data.py \
      --data_root /path/to/your/data \
      --out_dir   /path/to/output \
      --val_frac  0.10 \
      --test_frac 0.10
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from sklearn.model_selection import train_test_split


# ── helpers ──────────────────────────────────────────────────────────────────

def find_image(data_root: Path, stem: str, panel: str) -> Path | None:
    """
    Try several naming conventions to locate a panel image file.
    stem  = 'img162'
    panel = 'a'  or  'A'
    """
    candidates = [
        f"{stem}_{panel}.png",
        f"{stem}_{panel}.jpg",
        f"{stem}_{panel.upper()}.png",
        f"{stem}_{panel.lower()}.png",
        f"{stem}_panel_{panel}.png",
        f"{stem}_panel_{panel.upper()}.png",
    ]
    search_dirs = [
        data_root,
        data_root / "images",
    ]
    for d in search_dirs:
        for c in candidates:
            p = d / c
            if p.exists():
                return p
    return None


def load_annotations(data_root: Path):
    """
    Walk data_root (and data_root/annotations/) for *.json files,
    parse every panel entry, and match to its image file.
    Returns a flat list of dicts.
    """
    search_dirs = [data_root, data_root / "annotations"]
    json_files  = []
    for d in search_dirs:
        if d.exists():
            json_files.extend(d.glob("*.json"))

    records = []
    missing = 0

    for jf in sorted(json_files):
        stem = jf.stem            # e.g. 'img162'
        try:
            data = json.loads(jf.read_text())
        except json.JSONDecodeError:
            print(f"[WARN] Cannot parse {jf} — skipping")
            continue

        panels = data.get("panels", [])
        for entry in panels:
            panel_id  = str(entry.get("panel", ""))
            subcat    = entry.get("visualization_category", "Unknown")
            subtype   = entry.get("visualization_subtype",  "Unknown")
            subcap    = entry.get("subcaption", "").strip()
            summary   = entry.get("summary",    "").strip()

            if not subcap:          # skip if annotation is empty
                continue

            img_path = find_image(data_root, stem, panel_id)
            if img_path is None:
                missing += 1
                continue

            records.append({
                "image_path":            str(img_path),
                "figure_id":             stem,
                "panel_id":              panel_id,
                "visualization_category": subcat,
                "visualization_subtype":  subtype,
                "subcaption":            subcap,
                "summary":               summary,
                # composite key used for stratification
                "label":                 f"{subcat}__{subtype}",
            })

    print(f"Loaded {len(records)} records  |  {missing} image files not found")
    return records


# ── split logic ───────────────────────────────────────────────────────────────

def stratified_split(records, val_frac=0.10, test_frac=0.10, seed=42):
    """
    Stratify by visualization_subtype so every subtype is represented
    in all three splits proportionally.
    """
    labels = [r["label"] for r in records]

    # first cut off test
    train_val, test = train_test_split(
        records, test_size=test_frac,
        stratify=labels, random_state=seed
    )
    # then cut off val from the remainder
    val_relative = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val, test_size=val_relative,
        stratify=[r["label"] for r in train_val],
        random_state=seed
    )
    return train, val, test


# ── HuggingFace dataset ───────────────────────────────────────────────────────

def records_to_hf(records: list[dict]) -> Dataset:
    """Convert list of records to a HuggingFace Dataset with PIL images."""
    images = []
    for r in records:
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"[WARN] Cannot open {r['image_path']}: {e}")
            images.append(Image.new("RGB", (224, 224)))   # placeholder

    return Dataset.from_dict({
        "image":                   images,
        "figure_id":               [r["figure_id"]               for r in records],
        "panel_id":                [r["panel_id"]                for r in records],
        "visualization_category":  [r["visualization_category"]  for r in records],
        "visualization_subtype":   [r["visualization_subtype"]   for r in records],
        "subcaption":              [r["subcaption"]              for r in records],
        "summary":                 [r["summary"]                 for r in records],
    }).cast_column("image", HFImage())


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--val_frac",   type=float, default=0.10)
    parser.add_argument("--test_frac",  type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load
    records = load_annotations(data_root)
    if not records:
        raise RuntimeError("No records found — check your data_root layout")

    # 2. print class distribution
    dist = defaultdict(int)
    for r in records:
        dist[r["visualization_subtype"]] += 1
    print("\nVisualization subtype distribution:")
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {k:<35} {v:>7,}")

    # 3. split
    train, val, test = stratified_split(
        records, args.val_frac, args.test_frac, args.seed
    )
    print(f"\nSplit sizes — train: {len(train)}  val: {len(val)}  test: {len(test)}")

    # 4. save
    dd = DatasetDict({
        "train": records_to_hf(train),
        "val":   records_to_hf(val),
        "test":  records_to_hf(test),
    })
    dd.save_to_disk(str(out_dir / "matfig_captioning"))
    print(f"\nDataset saved to {out_dir / 'matfig_captioning'}")

    # 5. save a plain JSON manifest for inspection
    for split_name, split_records in [("train", train), ("val", val), ("test", test)]:
        manifest = [
            {k: v for k, v in r.items() if k != "label"}
            for r in split_records
        ]
        manifest_path = out_dir / f"{split_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Manifests saved.")


if __name__ == "__main__":
    main()
