#!/usr/bin/env python3
"""
Train any Ultralytics model (YOLOv8/v9/v10/v11, RT-DETR) and save metrics.

Usage:
    python trainers/train_yolo.py --model yolo11n.pt --name yolo11n --epochs 50
    python trainers/train_yolo.py --model rtdetr-l.pt --name rtdetrl --epochs 50
"""

import os
import json
import shutil
import argparse


def get_args():
    p = argparse.ArgumentParser("Train ultralytics model")
    p.add_argument("--model",       required=True, help="e.g. yolo11n.pt, rtdetr-l.pt")
    p.add_argument("--name",        required=True, help="Run name, used for results/<name>.json")
    p.add_argument("--data",        default="../ModelBench/data.yaml")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--imgsz",       type=int,   default=1024)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--device",      default="0", help="GPU id or 'cpu'")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--runs-dir",    default="./runs")
    return p.parse_args()


def main():
    args = get_args()

    from ultralytics import YOLO

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model  : {args.model}")
    print(f"  Name   : {args.name}")
    print(f"  Epochs : {args.epochs}")
    print(f"{'='*60}\n")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.runs_dir,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )

    box  = metrics.box
    prec = float(box.mp)
    rec  = float(box.mr)
    f1   = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
    record = {
        "model":      args.model,
        "name":       args.name,
        "epochs":     args.epochs,
        "map50":      round(float(box.map50), 4),
        "map75":      round(float(box.map75), 4),
        "map50_95":   round(float(box.map),   4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1":         f1,
        "per_class":  {
            str(i): round(float(v), 4)
            for i, v in enumerate(box.ap_class_index.tolist())
        } if hasattr(box, "ap_class_index") else {},
    }

    out = os.path.join(args.results_dir, f"{args.name}.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResults saved → {out}")

    # copy ultralytics results.csv (per-epoch loss/metric log) to results dir
    run_csv = os.path.join(args.runs_dir, args.name, "results.csv")
    if os.path.exists(run_csv):
        dest = os.path.join(args.results_dir, f"{args.name}_losses.csv")
        shutil.copy2(run_csv, dest)
        print(f"Loss log   → {dest}")


if __name__ == "__main__":
    main()
