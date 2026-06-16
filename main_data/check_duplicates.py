#!/usr/bin/env python3
"""
Scan a folder of inference JSON files and report images where the same
panel label (A, B, C … single, common) is detected more than once.

Usage:
    python check_duplicates.py --json-dir ./alloy_json_results
    python check_duplicates.py --json-dir ./results --min-score 0.5
"""

import os
import json
import argparse
from collections import defaultdict


def get_args():
    p = argparse.ArgumentParser("Check for duplicate panel labels in inference JSONs")
    p.add_argument("--json-dir",   required=True, help="Folder containing .json inference files")
    p.add_argument("--min-score",  type=float, default=0.0,
                   help="Only consider detections above this confidence (default: 0.0 = all)")
    p.add_argument("--verbose",    action="store_true",
                   help="Print every duplicate box, not just the summary")
    return p.parse_args()


def check_file(json_path, min_score):
    with open(json_path) as f:
        rec = json.load(f)

    detections = rec.get("detections", [])
    if min_score > 0:
        detections = [d for d in detections if d.get("score", 0) >= min_score]

    label_to_dets = defaultdict(list)
    for d in detections:
        label_to_dets[d["label_name"]].append(d)

    duplicates = {lbl: dets for lbl, dets in label_to_dets.items() if len(dets) > 1}
    return duplicates


def main():
    args = get_args()

    json_files = sorted(
        os.path.join(args.json_dir, f)
        for f in os.listdir(args.json_dir)
        if f.endswith(".json") and not f.startswith("_")
    )
    if not json_files:
        print(f"No .json files found in: {args.json_dir}")
        return

    n_total    = len(json_files)
    n_clean    = 0
    n_dupes    = 0
    label_freq = defaultdict(int)   # how many files each label was duplicated in

    dupe_files = []

    for path in json_files:
        try:
            dupes = check_file(path, args.min_score)
        except Exception as e:
            print(f"[error] {os.path.basename(path)}: {e}")
            continue

        if not dupes:
            n_clean += 1
            continue

        n_dupes += 1
        fname = os.path.basename(path)
        dupe_files.append((fname, dupes))

        for lbl in dupes:
            label_freq[lbl] += 1

        if args.verbose:
            print(f"\n{fname}")
            for lbl, dets in sorted(dupes.items()):
                scores = [f"{d['score']:.3f}" for d in dets]
                bboxes = [d["bbox"] for d in dets]
                print(f"  {lbl} × {len(dets)}   scores: {scores}")
                for bbox in bboxes:
                    print(f"    bbox: {[round(v,1) for v in bbox]}")

    # ── summary ───────────────────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  DUPLICATE LABEL REPORT")
    print(f"  Folder   : {args.json_dir}")
    if args.min_score > 0:
        print(f"  Min score: {args.min_score}")
    print(sep)
    print(f"  Total files checked : {n_total}")
    print(f"  Clean (no dupes)    : {n_clean}  ({100*n_clean/n_total:.1f}%)")
    print(f"  Files with dupes    : {n_dupes}  ({100*n_dupes/n_total:.1f}%)")

    if label_freq:
        print(f"\n  Duplicate frequency by label (# files affected):")
        for lbl, cnt in sorted(label_freq.items(), key=lambda x: -x[1]):
            print(f"    {lbl:>8}  :  {cnt} file(s)")

    if not args.verbose and dupe_files:
        print(f"\n  Files with duplicates (use --verbose for box details):")
        for fname, dupes in dupe_files[:30]:
            labels = ", ".join(
                f"{lbl}×{len(dets)}" for lbl, dets in sorted(dupes.items())
            )
            print(f"    {fname}  →  {labels}")
        if len(dupe_files) > 30:
            print(f"    ... and {len(dupe_files) - 30} more")

    print(sep)


if __name__ == "__main__":
    main()
