#!/usr/bin/env python3
"""
Load all results JSON files and print a ranked benchmark table.

Usage:
    python compare.py
    python compare.py --results-dir ./results --sort map50
"""

import os
import json
import argparse


COLS = ["map50", "map75", "map50_95", "precision", "recall", "f1", "apf", "apc", "apr"]
COL_LABELS = {
    "map50":     "mAP@50",
    "map75":     "mAP@75",
    "map50_95":  "mAP@50:95",
    "precision": "Precision",
    "recall":    "Recall",
    "f1":        "F1",
    "apf":       "APf(freq)",
    "apc":       "APc(com)",
    "apr":       "APr(rare)",
}


def get_args():
    p = argparse.ArgumentParser("ModelBench — compare results")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--sort",        default="map50",
                   choices=COLS, help="Column to sort by (desc)")
    return p.parse_args()


def load_results(results_dir: str) -> list[dict]:
    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        rows.append(data)
    return rows


def fmt(v) -> str:
    if v is None or v < 0:
        return "   -  "
    return f"{v:.4f}"


def print_table(rows: list[dict], sort_by: str):
    rows = sorted(rows, key=lambda r: r.get(sort_by, -1), reverse=True)

    name_w = max(len(r.get("name", "")) for r in rows) + 2
    name_w = max(name_w, 10)

    header = f"{'Model':<{name_w}}"
    for col in COLS:
        header += f"  {COL_LABELS[col]:>10}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in rows:
        name = r.get("name", r.get("model", "?"))
        line = f"{name:<{name_w}}"
        for col in COLS:
            line += f"  {fmt(r.get(col, -1)):>10}"
        print(line)

    print(sep)
    print(f"\n  Sorted by {COL_LABELS[sort_by]} (descending)")
    print(f"  '-' means metric was not computed for that model.\n")


def main():
    args = get_args()

    if not os.path.isdir(args.results_dir):
        print(f"[error] Results directory not found: {args.results_dir}")
        print("        Run benchmark.py first.")
        return

    rows = load_results(args.results_dir)
    if not rows:
        print(f"[error] No results found in {args.results_dir}")
        print("        Run benchmark.py first.")
        return

    print(f"Loaded {len(rows)} result(s) from {args.results_dir}/")
    print_table(rows, args.sort)


if __name__ == "__main__":
    main()
