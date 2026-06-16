#!/usr/bin/env python3
"""
Rebuild results JSONs from existing checkpoints (no retraining).
Run this once to recover lost results, then use eval_extended.py for APr/APc/APf.

Usage:
    python recover_results.py
    python recover_results.py --val-dir ../MatDetect/mydata_all/test
"""

import os
import json
import argparse


def get_args():
    p = argparse.ArgumentParser("Recover results from existing checkpoints")
    p.add_argument("--val-dir",     default="../MatDetect/data/val")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--runs-dir",    default="./runs/detect/runs")
    p.add_argument("--hf-ckpt",     default="../MatDetect/checkpoints/epoch046")
    p.add_argument("--device",      default="0")
    return p.parse_args()


args        = get_args()
RUNS_DIR    = args.runs_dir
HF_CKPT     = args.hf_ckpt
RESULTS_DIR = args.results_dir
DEVICE      = args.device
VAL_DIR     = args.val_dir

IMAGE_EXTS  = (".jpg", ".jpeg", ".png")

os.makedirs(RESULTS_DIR, exist_ok=True)


def make_temp_data_yaml(val_dir: str) -> str:
    """Write a temporary data.yaml pointing val to the given directory."""
    import yaml, tempfile
    val_images = os.path.join(os.path.abspath(val_dir), "images")
    cfg = {
        "train": val_images,  # required by ultralytics, unused during val
        "val":   val_images,
        "nc":    22,
        "names": {i: chr(ord("A") + i) for i in range(20)} | {20: "single", 21: "common"},
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def recover_yolo(name: str, best_pt: str):
    from ultralytics import YOLO
    print(f"  [{name}] evaluating ...", end=" ", flush=True)
    tmp_yaml = make_temp_data_yaml(VAL_DIR)
    model    = YOLO(best_pt)
    metrics  = model.val(data=tmp_yaml, device=DEVICE, verbose=False)
    os.remove(tmp_yaml)
    box     = metrics.box
    prec    = float(box.mp)
    rec     = float(box.mr)
    f1      = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
    record  = {
        "model":     name,
        "name":      name,
        "map50":     round(float(box.map50), 4),
        "map75":     round(float(box.map75), 4),
        "map50_95":  round(float(box.map),   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        f1,
    }
    out = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"mAP@50={record['map50']}  F1={record['f1']}")


def recover_hf():
    import torch
    from PIL import Image
    from tqdm import tqdm
    from collections import defaultdict
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from torchmetrics.detection import MeanAveragePrecision

    name   = "dabdetr"
    ckpt   = HF_CKPT
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  [{name}] evaluating ...", end=" ", flush=True)

    processor = AutoImageProcessor.from_pretrained(ckpt)
    model     = AutoModelForObjectDetection.from_pretrained(ckpt).to(device)
    model.eval()

    metric = MeanAveragePrecision(
        iou_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )

    images_dir = os.path.join(VAL_DIR, "images")
    labels_dir = os.path.join(VAL_DIR, "labels")
    stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(images_dir))
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    with torch.no_grad():
        for stem in tqdm(stems, desc="dabdetr", leave=False):
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
                    gt_boxes.append([(xc-bw/2)*W,(yc-bh/2)*H,(xc+bw/2)*W,(yc+bh/2)*H])
                    gt_labels.append(cls)
            if not gt_boxes: continue

            gt_boxes_t  = torch.tensor(gt_boxes,  dtype=torch.float32)
            gt_labels_t = torch.tensor(gt_labels, dtype=torch.long)

            enc  = processor(images=pil, return_tensors="pt")
            outs = model(pixel_values=enc["pixel_values"].to(device))
            res  = processor.post_process_object_detection(
                       outs, threshold=0.3, target_sizes=[(H, W)]
                   )[0]
            pred_boxes  = res["boxes"].cpu()
            pred_scores = res["scores"].cpu()
            pred_labels = res["labels"].cpu()

            metric.update(
                [{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
                [{"boxes": gt_boxes_t, "labels": gt_labels_t}],
            )

            matched = set()
            for i in pred_scores.argsort(descending=True).tolist():
                cls     = pred_labels[i].item()
                gt_mask = (gt_labels_t == cls).nonzero(as_tuple=True)[0]
                if gt_mask.numel() == 0: fp[cls] += 1; continue
                from torchmetrics.detection import MeanAveragePrecision as _
                iou = _box_iou(pred_boxes[i].unsqueeze(0), gt_boxes_t[gt_mask])[0]
                best_iou, best_j = iou.max(0)
                gj = gt_mask[best_j].item()
                if best_iou >= 0.5 and gj not in matched:
                    tp[cls] += 1; matched.add(gj)
                else:
                    fp[cls] += 1
            for i, cls in enumerate(gt_labels_t.tolist()):
                if i not in matched: fn[cls] += 1

    r      = metric.compute()
    ttp    = sum(tp.values()); tfp = sum(fp.values()); tfn = sum(fn.values())
    prec   = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0.0
    rec    = ttp / (ttp + tfn) if (ttp + tfn) > 0 else 0.0
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    record = {
        "model":     ckpt,
        "name":      name,
        "map50":     round(float(r["map_50"]),  4),
        "map75":     round(float(r["map_75"]),  4),
        "map50_95":  round(float(r["map"]),     4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
    }
    out = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"mAP@50={record['map50']}  F1={record['f1']}")


def _box_iou(a, b):
    import torch
    ax1,ay1,ax2,ay2 = a[:,0],a[:,1],a[:,2],a[:,3]
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    ix1 = torch.max(ax1.unsqueeze(1), bx1.unsqueeze(0))
    iy1 = torch.max(ay1.unsqueeze(1), by1.unsqueeze(0))
    ix2 = torch.min(ax2.unsqueeze(1), bx2.unsqueeze(0))
    iy2 = torch.min(ay2.unsqueeze(1), by2.unsqueeze(0))
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter).clamp(1e-6)


if __name__ == "__main__":
    print("Recovering YOLO results from best.pt checkpoints...\n")
    for name in sorted(os.listdir(RUNS_DIR)):
        best_pt = os.path.join(RUNS_DIR, name, "weights", "best.pt")
        if os.path.exists(best_pt):
            recover_yolo(name, best_pt)

    print("\nRecovering DAB-DETR results...\n")
    recover_hf()

    print(f"\nDone. {len(os.listdir(RESULTS_DIR))} results saved to {RESULTS_DIR}/")
    print("Now run: python eval_extended.py --force  (to add APr/APc/APf)")
