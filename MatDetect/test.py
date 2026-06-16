#!/usr/bin/env python3
"""
Evaluate MatDetect (DAB-DETR) on a labelled folder.
Outputs per-class table: AP@50 | AP@75 | AP@50:95 | Precision | Recall | F1

Usage:
    python test.py \
        --checkpoint ./checkpoints/epoch046 \
        --data-dir   ./data/val \
        --threshold  0.3
"""

import os
import argparse
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

try:
    from torchmetrics.detection import MeanAveragePrecision
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    print("[warn] torchmetrics not installed — pip install torchmetrics\n")


NUM_CLASSES = 22
ID2LABEL    = {i: chr(ord("A") + i) for i in range(20)}
ID2LABEL[20] = "single"
ID2LABEL[21] = "common"


# ── helpers ────────────────────────────────────────────────────────────────────

def yolo_to_xyxy(boxes_cxcywh: torch.Tensor, W: int, H: int) -> torch.Tensor:
    cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


def read_labels(path: str):
    boxes, classes = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            
            xc, yc, bw, bh = map(float, parts[1:5])
            boxes.append([xc, yc, bw, bh])
            classes.append(cls)
    return boxes, classes


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    ix1 = torch.max(ax1.unsqueeze(1), bx1.unsqueeze(0))
    iy1 = torch.max(ay1.unsqueeze(1), by1.unsqueeze(0))
    ix2 = torch.min(ax2.unsqueeze(1), bx2.unsqueeze(0))
    iy2 = torch.min(ay2.unsqueeze(1), by2.unsqueeze(0))
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter).clamp(1e-6)


def update_pr_counts(pred_boxes, pred_scores, pred_labels,
                     gt_boxes, gt_labels,
                     tp_map, fp_map, fn_map,
                     iou_thresh: float = 0.5):
    """Update TP/FP/FN counts per class for precision/recall at confidence threshold."""
    matched_gt = set()

    order = pred_scores.argsort(descending=True)
    for i in order.tolist():
        cls = pred_labels[i].item()
        box = pred_boxes[i].unsqueeze(0)

        gt_mask = (gt_labels == cls).nonzero(as_tuple=True)[0]
        if gt_mask.numel() == 0:
            fp_map[cls] += 1
            continue

        iou = box_iou(box, gt_boxes[gt_mask])[0]
        best_iou, best_j = iou.max(0)
        best_j_global = gt_mask[best_j].item()

        if best_iou >= iou_thresh and best_j_global not in matched_gt:
            tp_map[cls] += 1
            matched_gt.add(best_j_global)
        else:
            fp_map[cls] += 1

    for i, cls in enumerate(gt_labels.tolist()):
        if i not in matched_gt:
            fn_map[cls] += 1


# ── print table ────────────────────────────────────────────────────────────────

def print_table(rows, ap50_map, ap75_map, ap5095_map, tp_map, fp_map, fn_map):
    header = (
        f"{'Class':>6} | {'AP@50':>7} | {'AP@75':>7} | {'AP@50:95':>9} | "
        f"{'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'TP':>5} | {'FP':>5} | {'FN':>5}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    sum_ap50 = sum_ap75 = sum_ap5095 = 0.0
    sum_prec = sum_rec = sum_f1 = 0.0
    n_cls = 0
    total_tp = total_fp = total_fn = 0

    for cls_id in sorted(rows):
        label = ID2LABEL.get(cls_id, str(cls_id))
        ap50   = ap50_map.get(cls_id,   -1)
        ap75   = ap75_map.get(cls_id,   -1)
        ap5095 = ap5095_map.get(cls_id, -1)

        tp = tp_map[cls_id]
        fp = fp_map[cls_id]
        fn = fn_map[cls_id]

        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        def fmt(v): return f"{v:.4f}" if v >= 0 else "  -   "

        print(
            f"{label:>6} | {fmt(ap50):>7} | {fmt(ap75):>7} | {fmt(ap5095):>9} | "
            f"{prec:>7.4f} | {rec:>7.4f} | {f1:>7.4f} | {tp:>5} | {fp:>5} | {fn:>5}"
        )

        if ap50 >= 0:
            sum_ap50   += ap50
            sum_ap75   += max(ap75, 0)
            sum_ap5095 += max(ap5095, 0)
            sum_prec   += prec
            sum_rec    += rec
            sum_f1     += f1
            n_cls += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

    print(sep)
    if n_cls > 0:
        macro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        macro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        macro_f1   = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)
                      if (macro_prec + macro_rec) > 0 else 0)
        print(
            f"{'Mean':>6} | {sum_ap50/n_cls:>7.4f} | {sum_ap75/n_cls:>7.4f} | "
            f"{sum_ap5095/n_cls:>9.4f} | {macro_prec:>7.4f} | {macro_rec:>7.4f} | "
            f"{macro_f1:>7.4f} | {total_tp:>5} | {total_fp:>5} | {total_fn:>5}"
        )
    print(sep)
    print(f"\n  Note: AP values come from torchmetrics (COCO-style).")
    print(f"  Precision/Recall/F1 computed at --conf threshold after NMS at --iou threshold.")


# ── main ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("MatDetect — evaluation")
    p.add_argument("--checkpoint",  default="./checkpoints/epoch046")
    p.add_argument("--data-dir",    default="./data/val",
                   help="folder with images/ and labels/ sub-directories")
    p.add_argument("--conf",        type=float, default=0.3,
                   help="Confidence score threshold (default: 0.3)")
    p.add_argument("--iou",         type=float, default=0.5,
                   help="NMS IoU threshold (default: 0.5)")
    p.add_argument("--max-samples", type=int,   default=None)
    p.add_argument("--cuda-device", default="cuda:0")
    return p.parse_args()


def main():
    args = get_args()

    if not os.path.isdir(args.checkpoint):
        best_txt = os.path.join(os.path.dirname(args.checkpoint), "best_path.txt")
        if os.path.exists(best_txt):
            with open(best_txt) as f:
                args.checkpoint = f.read().strip()
            print(f"Using best checkpoint: {args.checkpoint}")

    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Conf       : {args.conf}")
    print(f"IoU (NMS)  : {args.iou}")

    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    model     = AutoModelForObjectDetection.from_pretrained(args.checkpoint).to(device)
    model.eval()

    images_dir = os.path.join(args.data_dir, "images")
    labels_dir = os.path.join(args.data_dir, "labels")

    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if args.max_samples:
        stems = stems[:args.max_samples]
    print(f"Images     : {len(stems)}\n")

    # separate metric objects for each IoU level
    if HAS_TORCHMETRICS:
        m50   = MeanAveragePrecision(iou_thresholds=[0.50], class_metrics=True)
        m75   = MeanAveragePrecision(iou_thresholds=[0.75], class_metrics=True)
        m5095 = MeanAveragePrecision(
            iou_thresholds=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95],
            class_metrics=True,
        )

    tp_map  = defaultdict(int)
    fp_map  = defaultdict(int)
    fn_map  = defaultdict(int)
    n_images, n_skipped = 0, 0

    with torch.no_grad():
        for stem in tqdm(stems, desc="Evaluating"):
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(images_dir, stem + ext)
                if os.path.exists(p):
                    img_path = p; break
            lbl_path = os.path.join(labels_dir, stem + ".txt")

            if img_path is None or not os.path.exists(lbl_path):
                n_skipped += 1; continue

            pil  = Image.open(img_path).convert("RGB")
            W, H = pil.size

            gt_raw, gt_cls = read_labels(lbl_path)
            if not gt_raw:
                n_skipped += 1; continue

            gt_boxes_t  = torch.tensor(gt_raw,  dtype=torch.float32)
            gt_labels_t = torch.tensor(gt_cls,  dtype=torch.long)
            gt_xyxy     = yolo_to_xyxy(gt_boxes_t, W, H)

            enc    = processor(images=pil, return_tensors="pt")
            outs   = model(pixel_values=enc["pixel_values"].to(device))
            res    = processor.post_process_object_detection(
                        outs, threshold=args.conf, target_sizes=[(H, W)]
                     )[0]

            pred_boxes  = res["boxes"].cpu()
            pred_scores = res["scores"].cpu()
            pred_labels = res["labels"].cpu()

            # NMS — transformer detectors don't apply it by default
            if pred_boxes.numel() > 0:
                from torchvision.ops import batched_nms
                keep        = batched_nms(pred_boxes, pred_scores, pred_labels, args.iou)
                pred_boxes  = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

            preds_dict  = {"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}
            target_dict = {"boxes": gt_xyxy,   "labels": gt_labels_t}

            if HAS_TORCHMETRICS:
                m50.update([preds_dict],   [target_dict])
                m75.update([preds_dict],   [target_dict])
                m5095.update([preds_dict], [target_dict])

            update_pr_counts(pred_boxes, pred_scores, pred_labels,
                             gt_xyxy, gt_labels_t,
                             tp_map, fp_map, fn_map)
            n_images += 1

    print(f"Evaluated {n_images} images  ({n_skipped} skipped)\n")

    # ── build per-class AP dicts ───────────────────────────────────────────────
    ap50_map = ap75_map = ap5095_map = {}
    all_classes = set()

    if HAS_TORCHMETRICS:
        r50   = m50.compute()
        r75   = m75.compute()
        r5095 = m5095.compute()

        classes_50   = r50.get("classes",   torch.tensor([])).tolist()
        classes_75   = r75.get("classes",   torch.tensor([])).tolist()
        classes_5095 = r5095.get("classes", torch.tensor([])).tolist()

        ap50_map   = {int(c): float(v) for c, v in zip(classes_50,   r50["map_per_class"].tolist())   if float(v) >= 0}
        ap75_map   = {int(c): float(v) for c, v in zip(classes_75,   r75["map_per_class"].tolist())   if float(v) >= 0}
        ap5095_map = {int(c): float(v) for c, v in zip(classes_5095, r5095["map_per_class"].tolist()) if float(v) >= 0}

        all_classes = set(ap50_map) | set(ap75_map) | set(ap5095_map)

        print(f"  Overall  mAP@0.50      : {r50['map'].item():.4f}")
        print(f"  Overall  mAP@0.75      : {r75['map'].item():.4f}")
        print(f"  Overall  mAP@0.50:0.95 : {r5095['map'].item():.4f}")
    else:
        print("[warn] torchmetrics not available — AP columns will be empty.")

    all_classes |= set(tp_map) | set(fp_map) | set(fn_map)
    print_table(all_classes, ap50_map, ap75_map, ap5095_map, tp_map, fp_map, fn_map)


if __name__ == "__main__":
    main()
