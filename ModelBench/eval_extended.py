#!/usr/bin/env python3
"""
Compute APr / APc / APf (rare / common / frequent) for every trained model
and patch the results JSONs in-place. No retraining.

Frequency splits (based on val set instance counts):
  Frequent (APf) : classes with > 100 instances  → classes 0–8
  Common   (APc) : classes with 10–100 instances → classes 9–13
  Rare     (APr) : classes with 1–10  instances  → classes 14–20

Usage:
    python eval_extended.py
    python eval_extended.py --val-dir ../MatDetect/data/val --results-dir ./results
"""

import os
import json
import argparse
from collections import defaultdict


# ── frequency split ────────────────────────────────────────────────────────────

def count_class_instances(labels_dir: str) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    counts[int(parts[0])] += 1
    return dict(counts)


def frequency_splits(counts: dict[int, int]):
    frequent, common, rare = [], [], []
    for cls, n in counts.items():
        if n > 100:
            frequent.append(cls)
        elif n >= 10:
            common.append(cls)
        else:
            rare.append(cls)
    return sorted(frequent), sorted(common), sorted(rare)


# ── YOLO per-class AP ──────────────────────────────────────────────────────────

def make_temp_yaml(val_dir: str) -> str:
    import yaml, tempfile
    val_images = os.path.join(os.path.abspath(val_dir), "images")
    cfg = {
        "train": val_images,
        "val":   val_images,
        "nc":    22,
        "names": {i: chr(ord("A") + i) for i in range(20)} | {20: "single", 21: "common"},
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def eval_yolo(best_pt: str, val_dir: str, device: str, conf_thresh: float = 0.55):
    """Returns (per_class_ap dict, full_metrics dict)."""
    from ultralytics import YOLO
    tmp_yaml = make_temp_yaml(val_dir)
    model    = YOLO(best_pt)
    metrics  = model.val(data=tmp_yaml, device=device, conf=conf_thresh, verbose=False)
    os.remove(tmp_yaml)
    box = metrics.box
    per_cls = {
        int(cls): float(ap)
        for cls, ap in zip(box.ap_class_index.tolist(), box.ap.tolist())
    }
    info = model.info(verbose=False)
    params_m = round(model.model.num_parameters() / 1e6, 2) if hasattr(model.model, "num_parameters") else None
    full = {
        "map50":    round(float(box.map50),  4),
        "map75":    round(float(box.map75),  4),
        "map50_95": round(float(box.map),    4),
        "mean_P":   round(float(box.mp),     4),
        "mean_R":   round(float(box.mr),     4),
    }
    return per_cls, full


# ── HF per-class AP ────────────────────────────────────────────────────────────

def _box_iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def eval_hf(ckpt: str, val_dir: str, device_str: str, conf_thresh: float = 0.5, iou_match: float = 0.5):
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from torchmetrics.detection import MeanAveragePrecision

    IMAGE_EXTS = (".jpg", ".jpeg", ".png")
    device     = torch.device(device_str if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(ckpt)
    model     = AutoModelForObjectDetection.from_pretrained(ckpt).to(device)
    model.eval()

    metric = MeanAveragePrecision(
        iou_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        class_metrics=True,
    )

    images_dir = os.path.join(val_dir, "images")
    labels_dir = os.path.join(val_dir, "labels")
    stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(images_dir))
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]

    tp_total = fp_total = fn_total = 0

    with torch.no_grad():
        for stem in tqdm(stems, desc=f"Eval HF ({os.path.basename(ckpt)})", leave=False):
            img_path = None
            for ext in IMAGE_EXTS:
                p = os.path.join(images_dir, stem + ext)
                if os.path.exists(p): img_path = p; break
            lbl_path = os.path.join(labels_dir, stem + ".txt")
            if img_path is None or not os.path.exists(lbl_path): continue

            pil  = Image.open(img_path).convert("RGB")
            W, H = pil.size
            gt_boxes, gt_labels = [], []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    gt_boxes.append([(xc-bw/2)*W, (yc-bh/2)*H,
                                     (xc+bw/2)*W, (yc+bh/2)*H])
                    gt_labels.append(cls)
            if not gt_boxes: continue

            import torch as _t
            gt_boxes_t  = _t.tensor(gt_boxes,  dtype=_t.float32)
            gt_labels_t = _t.tensor(gt_labels, dtype=_t.long)

            enc  = processor(images=pil, return_tensors="pt")
            outs = model(pixel_values=enc["pixel_values"].to(device))
            res  = processor.post_process_object_detection(
                       outs, threshold=0.0, target_sizes=[(H, W)]
                   )[0]

            metric.update(
                [{"boxes":  res["boxes"].cpu(),
                  "scores": res["scores"].cpu(),
                  "labels": res["labels"].cpu()}],
                [{"boxes":  gt_boxes_t, "labels": gt_labels_t}],
            )

            # ── P/R/F1: greedy IoU matching at conf_thresh ──────────────────
            scores  = res["scores"].cpu().tolist()
            labels  = res["labels"].cpu().tolist()
            boxes   = res["boxes"].cpu().tolist()

            # filter by confidence, sort descending
            preds = sorted(
                [(s, l, b) for s, l, b in zip(scores, labels, boxes) if s >= conf_thresh],
                key=lambda x: -x[0]
            )

            matched_gt = set()
            for score, pred_cls, pred_box in preds:
                hit = False
                for gi, (gt_cls, gt_box) in enumerate(zip(gt_labels, gt_boxes)):
                    if gi in matched_gt:
                        continue
                    if gt_cls == pred_cls and _box_iou(pred_box, gt_box) >= iou_match:
                        matched_gt.add(gi)
                        hit = True
                        break
                if hit:
                    tp_total += 1
                else:
                    fp_total += 1
            fn_total += len(gt_boxes) - len(matched_gt)

    r = metric.compute()
    classes = r.get("classes", _t.tensor([])).tolist()
    aps     = r["map_per_class"].tolist()
    per_cls = {int(c): float(v) for c, v in zip(classes, aps) if float(v) >= 0}

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0

    full = {
        "map50":    round(float(r["map_50"]),  4),
        "map75":    round(float(r["map_75"]),  4),
        "map50_95": round(float(r["map"]),     4),
        "mean_P":   round(precision, 4),
        "mean_R":   round(recall,    4),
    }
    return per_cls, full


# ── aggregate APr / APc / APf ──────────────────────────────────────────────────

def agg(per_class_ap: dict[int, float], class_list: list[int]) -> float:
    vals = [per_class_ap[c] for c in class_list if c in per_class_ap]
    return round(sum(vals) / len(vals), 4) if vals else -1.0


# ── main ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("Compute APr/APc/APf and patch results JSONs")
    p.add_argument("--val-dir",     default="../MatDetect/data/val")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--runs-dir",    default="./runs/detect/runs")
    p.add_argument("--device",      default="0")
    p.add_argument("--force",       action="store_true",
                   help="Recompute even if APr/APc/APf already exist in the JSON")
    p.add_argument("--models",      default=None,
                   help="Comma-separated list of model names to process (e.g. yolo12m,yolov10m). Default: all.")
    p.add_argument("--conf",        type=float, default=0.55, help="Confidence threshold for P/R/F1 (HF models)")
    p.add_argument("--iou-match",   type=float, default=0.45, help="IoU threshold for GT matching (HF models)")
    return p.parse_args()


def main():
    args = get_args()

    labels_dir = os.path.join(args.val_dir, "labels")
    counts     = count_class_instances(labels_dir)
    frequent, common, rare = frequency_splits(counts)

    print(f"Frequent classes ({len(frequent)}): {frequent}")
    print(f"Common   classes ({len(common)}):   {common}")
    print(f"Rare     classes ({len(rare)}):     {rare}\n")

    device_str  = f"cuda:{args.device}"

    allowed = set(args.models.split(",")) if args.models else None
    os.makedirs(args.results_dir, exist_ok=True)

    from models_registry_mydata import MODELS_MYDATA
    registry = {m["name"]: m for m in MODELS_MYDATA}
    existing = {os.path.splitext(f)[0] for f in os.listdir(args.results_dir) if f.endswith(".json")}

    for name, entry in registry.items():
        if allowed and name not in allowed:
            continue

        fpath = os.path.join(args.results_dir, f"{name}.json")

        # skip if JSON exists and already has apr/apc/apf (unless --force)
        if name in existing and not args.force:
            record = json.load(open(fpath))
            if all(k in record for k in ("apr", "apc", "apf")):
                print(f"{name}: already done, use --force to recompute.")
                continue

        print(f"Processing: {name} ...", end=" ", flush=True)
        try:
            ckpt = entry["model_id"]
            if entry["family"] == "yolo":
                per_cls, full = eval_yolo(ckpt, args.val_dir, args.device,
                                          conf_thresh=args.conf)
            else:
                per_cls, full = eval_hf(ckpt, args.val_dir, device_str,
                                        conf_thresh=args.conf,
                                        iou_match=args.iou_match)

            if name in existing:
                record = json.load(open(fpath))
            else:
                record = {"name": name, "checkpoint": ckpt}

            # always update full metrics (map50 etc.) in case they were missing
            record.update(full)

            record["apr"] = agg(per_cls, rare)
            record["apc"] = agg(per_cls, common)
            record["apf"] = agg(per_cls, frequent)

            with open(fpath, "w") as f:
                json.dump(record, f, indent=2)
            print(f"APr={record['apr']:.4f}  APc={record['apc']:.4f}  APf={record['apf']:.4f}")

        except Exception as e:
            print(f"FAILED — {e}")

    print("\nDone. Run 'python compare.py' to see the updated table.")


if __name__ == "__main__":
    main()
