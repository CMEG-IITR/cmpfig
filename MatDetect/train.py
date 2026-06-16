#!/usr/bin/env python3
"""
Panel detector — DAB-DETR fine-tuned for compound figure panel detection.
Uses HuggingFace transformers DAB-DETR (IDEA-Research/dab-detr-resnet-50).

Pretrained on COCO 2017 → fine-tuned to detect panels A, B, C, ... (26 classes).
No VLM. No captions. Just boxes.

Usage:
    python train.py \
        --train-dir ./data/train \
        --val-dir   ./data/val \
        --test-dir  ./data/test \
        --output    ./checkpoints
"""

import os
import sys
import csv
import math
import json
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModelForObjectDetection

from dataset import PanelDataset, collate_fn

NUM_CLASSES   = 22        # A–T (0-19) + single (20) + common (21)
MODEL_HUB     = "IDEA-Research/dab-detr-resnet-50"

ID2LABEL = {i: chr(ord("A") + i) for i in range(20)}
ID2LABEL[20] = "single"
ID2LABEL[21] = "common"
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

LOG_FIELDS = [
    "epoch", "split",
    "loss_total", "loss_ce", "loss_bbox", "loss_giou",
]


# ── training / eval loop ───────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool,
              grad_accum: int = 8, max_grad_norm: float = 0.1):
    """Run one epoch. Returns dict of mean losses: total, ce, bbox, giou."""
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()

    totals = {"total": 0.0, "ce": 0.0, "bbox": 0.0, "giou": 0.0}
    n = 0

    if train:
        optimizer.zero_grad()

    with ctx:
        label = "Train" if train else "Eval "
        pbar = tqdm(loader, leave=False, desc=label)
        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device)
            pixel_mask   = batch["pixel_mask"].to(device)
            labels = [
                {"class_labels": l["class_labels"].to(device),
                 "boxes":        l["boxes"].to(device)}
                for l in batch["labels"]
            ]

            outputs = model(pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels)
            loss = outputs.loss
            ld   = outputs.loss_dict

            ce   = ld.get("loss_ce",   torch.tensor(0.0)).item()
            bbox = ld.get("loss_bbox", torch.tensor(0.0)).item()
            giou = ld.get("loss_giou", torch.tensor(0.0)).item()

            if train:
                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            totals["total"] += loss.item()
            totals["ce"]    += ce
            totals["bbox"]  += bbox
            totals["giou"]  += giou
            n += 1

            pbar.set_postfix(
                total=f"{loss.item():.3f}",
                ce=f"{ce:.3f}",
                bbox=f"{bbox:.3f}",
                giou=f"{giou:.3f}",
            )

    denom = max(n, 1)
    return {k: v / denom for k, v in totals.items()}


# ── CSV logger ─────────────────────────────────────────────────────────────────

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
            "loss_ce":    f"{losses['ce']:.6f}",
            "loss_bbox":  f"{losses['bbox']:.6f}",
            "loss_giou":  f"{losses['giou']:.6f}",
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


# ── args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("Panel detector — DAB-DETR (transformers)")
    p.add_argument("--train-dir",     required=True)
    p.add_argument("--val-dir",       required=True)
    p.add_argument("--output",        default="./checkpoints")
    p.add_argument("--model-hub",     default=MODEL_HUB)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--lr-backbone",   type=float, default=1e-5)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--grad-accum",    type=int,   default=2)
    p.add_argument("--max-grad-norm", type=float, default=0.1)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--warmup-epochs", type=int,   default=2)
    p.add_argument("--num-workers",   type=int,   default=2)
    p.add_argument("--max-samples",   type=int,   default=None)
    p.add_argument("--cuda-device",   default="cuda:0")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device(args.cuda_device)
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading pretrained DAB-DETR from: {args.model_hub}")
    processor = AutoImageProcessor.from_pretrained(args.model_hub)
    model     = AutoModelForObjectDetection.from_pretrained(
        args.model_hub,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        attention_dropout=0.1,
        dropout=0.1,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    backbone_params    = list(model.model.backbone.parameters())
    backbone_ids       = {id(p) for p in backbone_params}
    other_params       = [p for p in model.parameters()
                          if id(p) not in backbone_ids and p.requires_grad]
    backbone_trainable = [p for p in backbone_params if p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": backbone_trainable, "lr": args.lr_backbone},
            {"params": other_params,       "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = PanelDataset(args.train_dir, processor, augment=True,
                            max_samples=args.max_samples)
    val_ds   = PanelDataset(args.val_dir,   processor, augment=False,
                            max_samples=args.max_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── CSV logger ────────────────────────────────────────────────────────────
    log_path = os.path.join(args.output, "losses.csv")
    logger   = CSVLogger(log_path)
    print(f"Logging all losses to: {log_path}\n")

    # ── training loop ─────────────────────────────────────────────────────────
    best_val   = float("inf")
    no_improve = 0
    best_dir   = None

    for epoch in range(1, args.epochs + 1):
        #----freezing/unfrezzingbackbone
        if epoch <= args.warmup_epochs:
            for p in model.model.backbone.parameters():
                p.requires_grad_(False)
        else:
            for p in model.model.backbone.parameters():
                p.requires_grad_(True)
        #-------------------------------------
        tr  = run_epoch(model, train_loader, optimizer, device, train=True,
                        grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm)
        val = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        logger.log(epoch, "train", tr)
        logger.log(epoch, "val",   val)

        improved = val["total"] < best_val
        if improved:
            best_val   = val["total"]
            no_improve = 0
        else:
            no_improve += 1

        lr_now = scheduler.get_last_lr()[1]
        print(
            f"Epoch {epoch:03d} | "
            f"train={tr['total']:.4f} [ce={tr['ce']:.3f} bbox={tr['bbox']:.3f} giou={tr['giou']:.3f}]  "
            f"val={val['total']:.4f} [ce={val['ce']:.3f} bbox={val['bbox']:.3f} giou={val['giou']:.3f}]"
            + f"  lr={lr_now:.2e}"
            + (" ✓ best" if improved else f" (no improve {no_improve}/{args.patience})")
        )

        # ── save checkpoint ───────────────────────────────────────────────────
        save_dir = os.path.join(args.output, f"epoch{epoch:03d}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "losses.json"), "w") as f:
            json.dump({"train": tr, "val": val}, f, indent=2)

        if improved:
            best_dir  = save_dir
            best_link = os.path.join(args.output, "best")
            if os.path.islink(best_link):
                os.remove(best_link)
            os.symlink(os.path.abspath(save_dir), best_link)

        if no_improve >= args.patience:
            print(f"Early stopping. Best: {best_dir}")
            break

    with open(os.path.join(args.output, "best_path.txt"), "w") as f:
        f.write(best_dir or "")
    print(f"\nDone. Best val={best_val:.4f}  →  {best_dir}")
    print(f"Full loss log: {log_path}")


if __name__ == "__main__":
    main()
