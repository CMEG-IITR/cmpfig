#!/usr/bin/env python3
"""
Evaluate class-singleton suppression quality against ground-truth labels on the test split.

For each image: after applying class-singleton suppression (keep top-1 box per class,
regardless of spatial location), check how many kept boxes are correct (IoU >= 0.5 with GT).

Reports: TP, FP, FN, Precision, Recall per model.

Usage:
    python eval_classnms_quality.py --weights path/to/best.pt --name yolo12m_baseline
    python eval_classnms_quality.py --weights path/to/best.pt --name yolo12m_unique
"""

import argparse
import json
import os
from pathlib import Path

import yaml
import torch
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def box_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def class_singleton_suppression(scores, labels):
    """Return indices keeping only the highest-scoring box per class (class-singleton suppression)."""
    best = {}
    for i, (s, c) in enumerate(zip(scores, labels)):
        c = int(c)
        if c not in best or s > best[c][0]:
            best[c] = (s, i)
    return [idx for _, idx in sorted(best.values(), key=lambda x: x[1])]


def load_gt(label_path: Path, img_w: int, img_h: int):
    """Load YOLO-format GT labels → list of (class_id, x1, y1, x2, y2)."""
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append((cid, x1, y1, x2, y2))
    return boxes


def get_args():
    p = argparse.ArgumentParser("Evaluate class-singleton suppression quality against GT on test split")
    p.add_argument("--weights",     required=True)
    p.add_argument("--name",        required=True)
    p.add_argument("--test-data",   default=None)
    p.add_argument("--conf",        type=float, default=0.55)
    p.add_argument("--iou",         type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--iou-match",   type=float, default=0.5,  help="GT matching IoU threshold")
    p.add_argument("--imgsz",       type=int,   default=1024)
    p.add_argument("--device",      default="0")
    p.add_argument("--results-dir", default="./results")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if args.test_data is None:
        args.test_data = str(Path(__file__).parent / "data_test.yaml")

    # Resolve test images and labels directories
    with open(args.test_data) as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg["path"])
    img_dir   = (base / cfg["test"]).resolve()
    label_dir = Path(str(img_dir).replace("/images", "/labels"))

    print(f"\n{'='*60}")
    print(f"  Class-singleton suppression quality: {args.name}")
    print(f"  Images : {img_dir}")
    print(f"  Labels : {label_dir}")
    print(f"{'='*60}\n")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    image_files = sorted(p for p in img_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    tp_total = fp_total = fn_total = 0
    per_class_tp = {}; per_class_fp = {}; per_class_fn = {}

    for img_path in tqdm(image_files, desc="Evaluating"):
        result = model.predict(str(img_path), conf=args.conf, iou=args.iou,
                               imgsz=args.imgsz, device=args.device,
                               verbose=False)[0]

        h, w = result.orig_shape
        gt_boxes = load_gt(label_dir / (img_path.stem + ".txt"), w, h)

        if result.boxes is None or len(result.boxes) == 0:
            fn_total += len(gt_boxes)
            for (cid, *_) in gt_boxes:
                per_class_fn[cid] = per_class_fn.get(cid, 0) + 1
            continue

        scores = result.boxes.conf.cpu().tolist()
        labels = result.boxes.cls.cpu().tolist()
        boxes  = result.boxes.xyxy.cpu().tolist()

        # Apply class-singleton suppression (top-1 box per class per image)
        keep = class_singleton_suppression(scores, labels)
        kept_boxes  = [boxes[i]  for i in keep]
        kept_labels = [int(labels[i]) for i in keep]

        # Match kept predictions to GT (greedy IoU matching)
        matched_gt = set()
        for pred_box, pred_cls in zip(kept_boxes, kept_labels):
            matched = False
            for gi, (gt_cls, *gt_box) in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                if gt_cls == pred_cls and box_iou(pred_box, gt_box) >= args.iou_match:
                    matched_gt.add(gi)
                    matched = True
                    break
            if matched:
                tp_total += 1
                per_class_tp[pred_cls] = per_class_tp.get(pred_cls, 0) + 1
            else:
                fp_total += 1
                per_class_fp[pred_cls] = per_class_fp.get(pred_cls, 0) + 1

        fn = len(gt_boxes) - len(matched_gt)
        fn_total += fn
        for gi, (cid, *_) in enumerate(gt_boxes):
            if gi not in matched_gt:
                per_class_fn[cid] = per_class_fn.get(cid, 0) + 1

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0

    print(f"\n{'='*55}")
    print(f"  Model      : {args.name}")
    print(f"  Images     : {len(image_files)}")
    print(f"  TP         : {tp_total}")
    print(f"  FP         : {fp_total}  (singleton kept wrong box)")
    print(f"  FN         : {fn_total}  (missed GT boxes)")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"{'='*55}")

    names = model.names
    all_cls = sorted(set(list(per_class_tp) + list(per_class_fp) + list(per_class_fn)))
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<8}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'Prec':>7}  {'Rec':>7}")
    print(f"  {'-'*46}")
    for cid in all_cls:
        name = names.get(cid, str(cid)) if names else str(cid)
        tp = per_class_tp.get(cid, 0)
        fp = per_class_fp.get(cid, 0)
        fn = per_class_fn.get(cid, 0)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {name:<8}  {tp:>6}  {fp:>6}  {fn:>6}  {p:>7.3f}  {r:>7.3f}")

    all_cls = sorted(set(list(per_class_tp) + list(per_class_fp) + list(per_class_fn)))
    per_class = {}
    for cid in all_cls:
        name = names.get(cid, str(cid)) if names else str(cid)
        tp = per_class_tp.get(cid, 0)
        fp = per_class_fp.get(cid, 0)
        fn = per_class_fn.get(cid, 0)
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn}

    record = {
        "name": args.name, "weights": args.weights,
        "conf": args.conf,
        "iou_nms": args.iou,
        "iou_match": args.iou_match,
        "images": len(image_files),
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "per_class": per_class,
    }
    out = os.path.join(args.results_dir, f"{args.name}_classnms_quality.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
