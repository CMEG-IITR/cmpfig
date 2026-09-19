#!/usr/bin/env python3
"""
Evaluate the final trained checkpoints on MatDetect/data/val and write
results/<name>.json in the same format as the existing ModelBench/results/*.

Checkpoints used:
  - YOLO family : ./runs/detect/runs/<name>/weights/best.pt
  - DAB-DETR    : path read from ../MatDetect/checkpoints/best_path.txt

No retraining — pure evaluation. Reuses eval_yolo()/eval_hf()/frequency-split
helpers from eval_extended.py.

Usage:
    ./rfdetr_env/bin/python eval_final.py
    ./rfdetr_env/bin/python eval_final.py --models yolo12m,dabdetr
    ./rfdetr_env/bin/python eval_final.py --val-dir ../MatDetect/data/val --device 0
"""

import os
import json
import argparse

from eval_extended import (
    eval_yolo, eval_hf,
    count_class_instances, frequency_splits, agg,
)

YOLO_NAMES = ["yolov8m", "yolov9c", "yolov10m", "yolo11m", "yolo12m"]


def get_args():
    p = argparse.ArgumentParser("Evaluate final YOLO + DAB-DETR checkpoints on MatDetect/data/val")
    p.add_argument("--val-dir",       default="../MatDetect/data/val")
    p.add_argument("--runs-dir",      default="./runs/detect/runs")
    p.add_argument("--best-path-txt", default="../MatDetect/checkpoints/best_path.txt")
    p.add_argument("--results-dir",   default="./results")
    p.add_argument("--device",        default="0")
    p.add_argument("--conf",          type=float, default=0.50, help="Confidence threshold (YOLO + P/R/F1 for HF)")
    p.add_argument("--iou-match",     type=float, default=0.40, help="IoU threshold for GT matching (HF models)")
    p.add_argument("--models",        default=None,
                   help="Comma-separated subset, e.g. yolo12m,dabdetr. Default: all.")
    return p.parse_args()


def build_registry(args):
    registry = {}
    for name in YOLO_NAMES:
        ckpt = os.path.join(args.runs_dir, name, "weights", "best.pt")
        registry[name] = {"family": "yolo", "checkpoint": ckpt}

    with open(args.best_path_txt) as f:
        dab_ckpt_rel = f.read().strip()
    # best_path.txt is relative to MatDetect/, resolve relative to ModelBench/
    matdetect_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.best_path_txt)))
    dab_ckpt = os.path.normpath(os.path.join(matdetect_dir, dab_ckpt_rel.lstrip("./")))
    registry["dabdetr"] = {"family": "hf", "checkpoint": dab_ckpt}

    return registry


def main():
    args = get_args()
    device_str = f"cuda:{args.device}"
    os.makedirs(args.results_dir, exist_ok=True)

    labels_dir = os.path.join(args.val_dir, "labels")
    counts = count_class_instances(labels_dir)
    frequent, common, rare = frequency_splits(counts)
    print(f"Frequent classes ({len(frequent)}): {frequent}")
    print(f"Common   classes ({len(common)}):   {common}")
    print(f"Rare     classes ({len(rare)}):     {rare}\n")

    registry = build_registry(args)
    allowed = set(args.models.split(",")) if args.models else None

    for name, entry in registry.items():
        if allowed and name not in allowed:
            continue

        ckpt = entry["checkpoint"]
        if not os.path.exists(ckpt):
            print(f"{name}: SKIPPED — checkpoint not found: {ckpt}")
            continue

        print(f"Processing: {name}  ({ckpt}) ...", end=" ", flush=True)
        try:
            if entry["family"] == "yolo":
                per_cls, full = eval_yolo(ckpt, args.val_dir, args.device, conf_thresh=args.conf)
            else:
                per_cls, full = eval_hf(ckpt, args.val_dir, device_str,
                                         conf_thresh=args.conf, iou_match=args.iou_match)

            record = {"name": name, "checkpoint": ckpt}
            record.update(full)
            record["apr"] = agg(per_cls, rare)
            record["apc"] = agg(per_cls, common)
            record["apf"] = agg(per_cls, frequent)

            out = os.path.join(args.results_dir, f"{name}.json")
            with open(out, "w") as f:
                json.dump(record, f, indent=2)

            print(f"map50={record['map50']:.4f}  map50-95={record['map50_95']:.4f}  "
                  f"APr={record['apr']:.4f}  APc={record['apc']:.4f}  APf={record['apf']:.4f}")

        except Exception as e:
            print(f"FAILED — {e}")

    print(f"\nDone. Results written to {args.results_dir}/")
    print("Run 'python compare.py' to see the updated table.")


if __name__ == "__main__":
    main()
