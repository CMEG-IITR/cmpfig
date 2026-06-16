#!/usr/bin/env python3
"""
Train any HuggingFace object detection model (DAB-DETR, Deformable-DETR,
Conditional-DETR, DETA) on YOLO-format data and save benchmark metrics.

Usage:
    python trainers/train_hf.py --model IDEA-Research/dab-detr-resnet-50 --name dabdetr
    python trainers/train_hf.py --model SenseTime/deformable-detr --name deformable_detr
"""

import os
import csv
import json
import argparse
import math
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection


NUM_CLASSES = 22
ID2LABEL = {i: chr(ord("A") + i) for i in range(20)}
ID2LABEL[20] = "single"
ID2LABEL[21] = "common"
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _find_image(images_dir: str, stem: str) -> str | None:
    for ext in IMAGE_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def _parse_yolo_labels(label_path: str, *, strict: bool = False):
    boxes, classes, errors = [], [], []
    with open(label_path) as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                errors.append(f"line {line_no}: expected 5 fields, got {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
            except ValueError:
                errors.append(f"line {line_no}: could not parse class/box")
                continue

            if not 0 <= cls < NUM_CLASSES:
                errors.append(f"line {line_no}: class {cls} outside [0,{NUM_CLASSES - 1}]")
                continue
            if not all(math.isfinite(v) for v in (xc, yc, bw, bh)):
                errors.append(f"line {line_no}: non-finite box value")
                continue
            if bw <= 0 or bh <= 0:
                errors.append(f"line {line_no}: non-positive box size ({bw}, {bh})")
                continue
            if strict and not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                errors.append(f"line {line_no}: normalized box outside valid range")
                continue

            bw = max(1e-6, min(1.0, bw))
            bh = max(1e-6, min(1.0, bh))
            xc = max(bw / 2, min(1.0 - bw / 2, xc))
            yc = max(bh / 2, min(1.0 - bh / 2, yc))
            boxes.append([xc, yc, bw, bh])
            classes.append(cls)
    return boxes, classes, errors


def _validate_targets(labels, stems):
    for stem, target in zip(stems, labels):
        classes = target["class_labels"]
        boxes = target["boxes"]
        if classes.numel() == 0 or boxes.numel() == 0:
            raise ValueError(f"{stem}: no valid labels")
        if classes.min().item() < 0 or classes.max().item() >= NUM_CLASSES:
            raise ValueError(f"{stem}: class id outside [0,{NUM_CLASSES - 1}]")
        if not torch.isfinite(boxes).all():
            raise ValueError(f"{stem}: non-finite box")
        if (boxes[:, 2:] <= 0).any():
            raise ValueError(f"{stem}: non-positive normalized box size")
        if (boxes < 0).any() or (boxes > 1).any():
            raise ValueError(f"{stem}: normalized box outside [0,1]")


# ── dataset ────────────────────────────────────────────────────────────────────

class YOLODetectionDataset(Dataset):
    """Mirrors MatDetect's PanelDataset exactly — processor handles image only,
    labels are built manually as {class_labels, boxes} in normalised cxcywh."""

    def __init__(self, root: str, processor, augment: bool = False):
        self.images_dir = os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels")
        self.processor  = processor
        self.samples = []
        skipped = []
        filtered = []

        for f in sorted(os.listdir(self.images_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMAGE_EXTS:
                continue
            img_path = os.path.join(self.images_dir, f)
            label_path = os.path.join(self.labels_dir, stem + ".txt")
            if not os.path.exists(label_path):
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                skipped.append((stem, f"image unreadable: {e}"))
                continue

            boxes, classes, errors = _parse_yolo_labels(label_path)
            if not boxes:
                reason = "; ".join(errors[:3]) if errors else "no labels"
                skipped.append((stem, reason))
                continue
            if errors:
                filtered.append((stem, "; ".join(errors[:3])))
            self.samples.append((stem, img_path, boxes, classes))

        if skipped:
            print(f"[dataset] skipped {len(skipped)} invalid samples under {root}")
            for stem, reason in skipped[:20]:
                print(f"  - {stem}: {reason}")
            if len(skipped) > 20:
                print(f"  ... {len(skipped) - 20} more")
        if filtered:
            print(f"[dataset] filtered bad label rows in {len(filtered)} samples under {root}")
            for stem, reason in filtered[:20]:
                print(f"  - {stem}: {reason}")
            if len(filtered) > 20:
                print(f"  ... {len(filtered) - 20} more")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        stem, img_path, boxes, classes = self.samples[idx]

        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[dataset] skipping unreadable image at load time: {img_path} ({e})")
            return None

        # processor handles image only — no annotations passed
        enc          = self.processor(images=pil, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)   # (3, H, W)

        return {
            "pixel_values": pixel_values,
            "labels": {
                "class_labels": torch.tensor(classes, dtype=torch.long),
                "boxes":        torch.tensor(boxes,   dtype=torch.float32),
            },
            "stem": stem,
        }


def collate_fn(batch):
    """Matches MatDetect collate_fn: drops None, pads to max H×W, builds pixel_mask."""
    import torch.nn.functional as F
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    pixel_values_list = [b["pixel_values"] for b in batch]
    max_h = max(pv.shape[1] for pv in pixel_values_list)
    max_w = max(pv.shape[2] for pv in pixel_values_list)

    padded_pixels, pixel_masks = [], []
    for pv in pixel_values_list:
        _, h, w = pv.shape
        padded  = F.pad(pv, (0, max_w - w, 0, max_h - h))
        mask    = torch.zeros(max_h, max_w, dtype=torch.long)
        mask[:h, :w] = 1
        padded_pixels.append(padded)
        pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_pixels),
        "pixel_mask":   torch.stack(pixel_masks),
        "labels":       [b["labels"] for b in batch],
        "stems":        [b["stem"] for b in batch],
    }


# ── evaluation helpers ─────────────────────────────────────────────────────────

def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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


def evaluate(model, processor, val_dir: str, device, conf: float = 0.3, iou: float = 0.5):
    try:
        from torchmetrics.detection import MeanAveragePrecision
        metric = MeanAveragePrecision(iou_thresholds=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95])
        has_tm = True
    except ImportError:
        has_tm = False

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    images_dir = os.path.join(val_dir, "images")
    labels_dir = os.path.join(val_dir, "labels")
    stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(images_dir))
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]

    model.eval()
    with torch.no_grad():
        for stem in tqdm(stems, desc="Evaluating", leave=False):
            img_path = None
            for ext in IMAGE_EXTS:
                p = os.path.join(images_dir, stem + ext)
                if os.path.exists(p): img_path = p; break
            lbl_path = os.path.join(labels_dir, stem + ".txt")
            if img_path is None or not os.path.exists(lbl_path): continue

            pil  = Image.open(img_path).convert("RGB")
            W, H = pil.size
            yolo_boxes, gt_labels, _ = _parse_yolo_labels(lbl_path)
            gt_boxes = [
                [(xc-bw/2)*W, (yc-bh/2)*H, (xc+bw/2)*W, (yc+bh/2)*H]
                for xc, yc, bw, bh in yolo_boxes
            ]
            if not gt_boxes: continue

            gt_boxes_t  = torch.tensor(gt_boxes,  dtype=torch.float32)
            gt_labels_t = torch.tensor(gt_labels, dtype=torch.long)

            enc  = processor(images=pil, return_tensors="pt")
            outs = model(pixel_values=enc["pixel_values"].to(device))
            res  = processor.post_process_object_detection(
                       outs, threshold=conf, target_sizes=[(H, W)]
                   )[0]

            pred_boxes  = res["boxes"].cpu()
            pred_scores = res["scores"].cpu()
            pred_labels = res["labels"].cpu()

            if pred_boxes.numel() > 0:
                from torchvision.ops import batched_nms
                keep        = batched_nms(pred_boxes, pred_scores, pred_labels, iou)
                pred_boxes  = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

            if has_tm:
                metric.update(
                    [{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
                    [{"boxes": gt_boxes_t, "labels": gt_labels_t}]
                )

            matched = set()
            for i in pred_scores.argsort(descending=True).tolist():
                cls = pred_labels[i].item()
                gt_mask = (gt_labels_t == cls).nonzero(as_tuple=True)[0]
                if gt_mask.numel() == 0: fp[cls] += 1; continue
                iou_mat = box_iou(pred_boxes[i].unsqueeze(0), gt_boxes_t[gt_mask])[0]
                best_iou, best_j = iou_mat.max(0)
                gj = gt_mask[best_j].item()
                if best_iou >= 0.5 and gj not in matched:
                    tp[cls] += 1; matched.add(gj)
                else:
                    fp[cls] += 1
            for i, cls in enumerate(gt_labels_t.tolist()):
                if i not in matched: fn[cls] += 1

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

    out = {"precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4),
           "map50": -1, "map75": -1, "map50_95": -1}

    if has_tm:
        r = metric.compute()
        out["map50"]    = round(float(r["map_50"]),  4)
        out["map75"]    = round(float(r["map_75"]),  4)
        out["map50_95"] = round(float(r["map"]),     4)

    return out


# ── training loop ──────────────────────────────────────────────────────────────

LOG_FIELDS = ["epoch", "split", "loss_total", "loss_ce", "loss_bbox", "loss_giou"]


class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    def log(self, epoch: int, split: str, losses: dict):
        row = {
            "epoch":      epoch,
            "split":      split,
            "loss_total": f"{losses['total']:.6f}",
            "loss_ce":    f"{losses.get('ce',   0.0):.6f}",
            "loss_bbox":  f"{losses.get('bbox', 0.0):.6f}",
            "loss_giou":  f"{losses.get('giou', 0.0):.6f}",
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


def run_epoch(model, loader, optimizer, device, train: bool,
              grad_accum: int = 4, max_grad_norm: float = 0.1):
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    totals = {"total": 0.0, "ce": 0.0, "bbox": 0.0, "giou": 0.0}
    n = 0
    if train: optimizer.zero_grad()

    with ctx:
        for step, batch in enumerate(tqdm(loader, desc="train" if train else "val", leave=False)):
            if batch is None:
                continue
            _validate_targets(batch["labels"], batch["stems"])
            pv = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]
            kwargs = {"pixel_values": pv, "labels": labels}
            if "pixel_mask" in batch:
                kwargs["pixel_mask"] = batch["pixel_mask"].to(device)

            outs = model(**kwargs)
            ld = outs.loss_dict if hasattr(outs, "loss_dict") and outs.loss_dict else {}

            if train:
                (outs.loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            totals["total"] += outs.loss.item()
            totals["ce"]    += ld.get("loss_ce",   ld.get("loss_vfl", ld.get("loss_focal", torch.tensor(0.0)))).item()
            totals["bbox"]  += ld.get("loss_bbox", torch.tensor(0.0)).item()
            totals["giou"]  += ld.get("loss_giou", torch.tensor(0.0)).item()
            n += 1

    denom = max(n, 1)
    return {k: v / denom for k, v in totals.items()}


# ── main ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("Train HuggingFace detection model")
    p.add_argument("--model",          required=True, help="HuggingFace model id")
    p.add_argument("--name",           required=True, help="Run name for results/<name>.json")
    p.add_argument("--train-dir",      default="../MatDetect/data/train")
    p.add_argument("--val-dir",        default="../MatDetect/data/val")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--lr-backbone",    type=float, default=1e-5)
    p.add_argument("--warmup-epochs",  type=int,   default=2)
    p.add_argument("--grad-accum",     type=int,   default=4)
    p.add_argument("--max-grad-norm",  type=float, default=0.1)
    p.add_argument("--conf",           type=float, default=0.3,
                   help="Confidence threshold for evaluation (default: 0.3)")
    p.add_argument("--iou",            type=float, default=0.5,
                   help="NMS IoU threshold for evaluation (default: 0.5)")
    p.add_argument("--patience",       type=int,   default=5,
                   help="Early stopping patience in epochs (default: 5)")
    p.add_argument("--cuda-device",    default="cuda:0")
    p.add_argument("--results-dir",    default="./results")
    p.add_argument("--ckpt-dir",       default="./checkpoints")
    p.add_argument("--eval-only",      action="store_true",
                   help="Skip training, evaluate --model checkpoint directly")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if args.eval_only:
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(f"\n{'='*60}")
        print(f"  Eval-only : {args.model}")
        print(f"  Name      : {args.name}")
        print(f"  Device    : {device}")
        print(f"{'='*60}\n")

        model     = AutoModelForObjectDetection.from_pretrained(args.model).to(device)
        processor = AutoImageProcessor.from_pretrained(args.model)
        metrics   = evaluate(model, processor, args.val_dir, device, args.conf, args.iou)

        record = {"model": args.model, "name": args.name, "epochs": "pretrained", **metrics}
        out = os.path.join(args.results_dir, f"{args.name}.json")
        with open(out, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\nResults saved → {out}")
        return

    ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Model  : {args.model}")
    print(f"  Name   : {args.name}")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}")
    print(f"{'='*60}\n")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForObjectDetection.from_pretrained(
        args.model,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds = YOLODetectionDataset(args.train_dir, processor, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)

    val_ds = YOLODetectionDataset(args.val_dir, processor, augment=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Split backbone vs head params — backbone gets 10× lower LR (matches MatDetect)
    backbone_params = [p for name, p in model.named_parameters()
                       if any(kw in name for kw in ("backbone", "vit", "patch_embed", "encoder"))
                       and p.requires_grad]
    backbone_ids    = {id(p) for p in backbone_params}
    other_params    = [p for p in model.parameters()
                       if id(p) not in backbone_ids and p.requires_grad]

    if backbone_params:
        print(f"  Backbone params : {sum(p.numel() for p in backbone_params):,}  lr={args.lr_backbone}")
        print(f"  Head params     : {sum(p.numel() for p in other_params):,}  lr={args.lr}")
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": other_params,    "lr": args.lr},
        ], weight_decay=1e-4)
    else:
        print(f"  [warn] no backbone params matched — using single lr={args.lr}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Linear warmup then cosine decay (matches MatDetect)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = os.path.join(ckpt_dir, "losses.csv")
    logger   = CSVLogger(log_path)
    print(f"Loss log : {log_path}\n")

    best_loss  = float("inf")
    best_ckpt  = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr  = run_epoch(model, train_dl, optimizer, device, train=True,
                        grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm)
        val = run_epoch(model, val_dl,   optimizer, device, train=False,
                        grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm)
        scheduler.step()

        improved = val["total"] < best_loss
        if improved:
            best_loss  = val["total"]
            no_improve = 0
        else:
            no_improve += 1

        logger.log(epoch, "train", tr)
        logger.log(epoch, "val",   val)
        lr_now = scheduler.get_last_lr()[-1]
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train={tr['total']:.4f}  val={val['total']:.4f}  "
            f"[ce={val['ce']:.3f}  bbox={val['bbox']:.3f}  giou={val['giou']:.3f}]"
            f"  lr={lr_now:.2e}"
            + (f" ✓ best" if improved else f"  (no improve {no_improve}/{args.patience})")
        )

        if improved:
            # remove previous best before saving new one
            if best_ckpt is not None and os.path.isdir(best_ckpt):
                import shutil
                shutil.rmtree(best_ckpt)
            best_ckpt = os.path.join(ckpt_dir, f"epoch{epoch:03d}")
            model.save_pretrained(best_ckpt)
            processor.save_pretrained(best_ckpt)
            with open(os.path.join(ckpt_dir, "best_path.txt"), "w") as f:
                f.write(best_ckpt)

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_loss:.4f}")
            break

    print(f"\nBest checkpoint: {best_ckpt}")
    print("Running evaluation...")

    best_model = AutoModelForObjectDetection.from_pretrained(best_ckpt).to(device)
    best_proc  = AutoImageProcessor.from_pretrained(best_ckpt)
    metrics = evaluate(best_model, best_proc, args.val_dir, device, args.conf, args.iou)

    record = {
        "model":    args.model,
        "name":     args.name,
        "epochs":   args.epochs,
        "checkpoint": best_ckpt,
        **metrics,
    }

    out = os.path.join(args.results_dir, f"{args.name}.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
