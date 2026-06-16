#!/usr/bin/env python3
"""
Phase 2 — Upload dataset_index.json to HuggingFace as Parquet shards.

Reads image paths from the index, streams them one at a time via a generator
(memory safe), and pushes to HuggingFace using the datasets library.
HuggingFace handles sharding, Parquet conversion, and image encoding.

Columns:
  image                   HFImage (shown in dataset viewer)
  image_id                str
  panel_suffix            str
  visualization_category  str
  visualization_subtype   str
  subcaption              str
  summary                 str

Usage:
  python upload_dataset.py
  python upload_dataset.py --shard-size 1GB
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi


FEATURES = Features({
    "image":                  HFImage(),
    "image_id":               Value("string"),
    "panel_suffix":           Value("string"),
    "visualization_category": Value("string"),
    "visualization_subtype":  Value("string"),
    "subcaption":             Value("string"),
    "summary":                Value("string"),
})


def get_args():
    p = argparse.ArgumentParser("Upload dataset to HuggingFace as Parquet shards")
    p.add_argument("--index",      default="./dataset_index.json")
    p.add_argument("--repo",       default="subham2507/MatSciFig",
                   help="HuggingFace repo id (default: subham2507/MatSciFig)")
    p.add_argument("--shard-size", default="500MB",
                   help="Max shard size (default: 500MB)")
    return p.parse_args()


def main():
    args = get_args()

    # load index
    print(f"Loading index: {args.index}")
    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)
    print(f"Total records: {len(index)}\n")

    # generator — lazily loads one image at a time, memory safe
    def generate():
        for rec in index:
            img_path = Path(rec["image_path"])
            if not img_path.exists():
                print(f"  [skip] {img_path.name} not found")
                continue
            yield {
                "image":                  str(img_path),
                "image_id":               rec["image_id"],
                "panel_suffix":           rec["panel_suffix"],
                "visualization_category": rec["visualization_category"],
                "visualization_subtype":  rec["visualization_subtype"],
                "subcaption":             rec["subcaption"],
                "summary":                rec["summary"],
            }

    print("Building dataset from generator...")
    ds = Dataset.from_generator(generate, features=FEATURES)

    print(f"Pushing to HuggingFace: {args.repo}")
    ds.push_to_hub(
        args.repo,
        split="train",
        max_shard_size=args.shard_size,
        commit_message="Add MatSciFig dataset",
    )

    # upload README
    readme = Path("./README.md")
    if readme.exists():
        HfApi().upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        print("README uploaded")

    print(f"\nAll done → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
