#!/usr/bin/env python3
"""
Train YOLO12m with a custom per-image class Uniqueness Loss.

Problem being solved:
    YOLO predicts each anchor independently — it has no global constraint that
    a given class (e.g. "B") should appear at most once per compound figure.
    The uniqueness loss adds a penalty when the same class fires with high
    confidence on multiple spatially-separate anchors in the same image.

How the loss works:
    For each (image, class) pair in the batch:
      - Take all anchor confidence scores for that class (after sigmoid)
      - Sort descending
      - Allow the top-1 prediction to exist freely
      - Penalise every additional prediction above `uloss_thresh`
    Penalty = sum of duplicate confidence scores / (B * nc)
    This is added to the standard (box + cls + dfl) loss scaled by `uloss_weight`.

Hook into Ultralytics 8.4.x:
    v8DetectionLoss.loss(preds_dict, batch) is overridden in a subclass.
    The subclass is injected into model.criterion via the on_train_start callback
    so the rest of the trainer is completely untouched.

Usage:
    python trainers/train_yolo12m_unique.py
    python trainers/train_yolo12m_unique.py --epochs 100 --uloss-weight 0.1
    python trainers/train_yolo12m_unique.py --uloss-weight 0.05 --uloss-thresh 0.4 --warmup-epochs 5
"""

import os
import json
import shutil
import argparse

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model


# ── Uniqueness Loss ────────────────────────────────────────────────────────────

class UniquenessDetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss + per-image class uniqueness penalty.

    preds["scores"] shape coming into loss(): [B, nc, total_anchors]
    After sigmoid these are class probabilities across all anchor positions.
    For each (image, class) we keep the best prediction free and penalise
    every additional prediction that exceeds uloss_thresh.
    """

    def __init__(self, model, uloss_weight: float = 0.05, uloss_thresh: float = 0.5,
                 warmup_epochs: int = 5):
        super().__init__(model)
        self.uloss_weight   = uloss_weight
        self.uloss_thresh   = uloss_thresh
        self.warmup_epochs  = warmup_epochs
        self.current_epoch  = 0          # updated by callback each epoch
        self._last_uloss    = 0.0        # for logging

    def loss(self, preds: dict, batch: dict):
        total_loss, loss_items = super().loss(preds, batch)

        if self.uloss_weight > 0 and self.current_epoch >= self.warmup_epochs:
            u = self._uniqueness_penalty(preds["scores"])
            self._last_uloss = float(u.detach())
            total_loss = total_loss + self.uloss_weight * u

        return total_loss, loss_items

    def _uniqueness_penalty(self, scores: torch.Tensor) -> torch.Tensor:
        """
        scores: [B, nc, total_anchors]  (raw logits, pre-sigmoid)

        Returns scalar penalty:
            sum of confidence of all duplicate predictions (above thresh, rank > 0)
            normalised by B * nc so it stays scale-invariant.
        """
        probs = scores.sigmoid()                              # [B, nc, N]

        # Sort each (image, class) slice from highest to lowest confidence
        sorted_probs, _ = probs.sort(dim=2, descending=True) # [B, nc, N]

        # Binary mask: 1 where prediction is above threshold
        above = (sorted_probs > self.uloss_thresh).float()

        # Allow rank-0 (the best prediction) — only penalise rank >= 1
        above[:, :, 0] = 0.0

        penalty = (sorted_probs * above).sum()

        B, nc, _ = probs.shape
        return penalty / max(B * nc, 1)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def make_callbacks(uloss_weight: float, uloss_thresh: float, warmup_epochs: int):
    """Return the three callbacks needed to inject and monitor the uniqueness loss."""

    def on_train_start(trainer):
        """Replace model.criterion with UniquenessDetectionLoss after normal setup."""
        m = unwrap_model(trainer.model)
        m.criterion = UniquenessDetectionLoss(
            m,
            uloss_weight=uloss_weight,
            uloss_thresh=uloss_thresh,
            warmup_epochs=warmup_epochs,
        )
        LOGGER.info(
            f"\n  [UniqueLoss] Injected — weight={uloss_weight}, "
            f"thresh={uloss_thresh}, warmup={warmup_epochs} epochs\n"
        )

    def on_train_epoch_start(trainer):
        """Keep criterion's epoch counter in sync with trainer."""
        m = unwrap_model(trainer.model)
        if isinstance(m.criterion, UniquenessDetectionLoss):
            m.criterion.current_epoch = trainer.epoch

    def on_train_epoch_end(trainer):
        """Print uniqueness loss value at end of each epoch."""
        m = unwrap_model(trainer.model)
        if isinstance(m.criterion, UniquenessDetectionLoss):
            crit = m.criterion
            if crit.current_epoch >= crit.warmup_epochs:
                LOGGER.info(
                    f"  [UniqueLoss] epoch {trainer.epoch+1}  "
                    f"u_loss_raw={crit._last_uloss:.6f}  "
                    f"weighted={crit._last_uloss * crit.uloss_weight:.6f}"
                )
            else:
                remaining = crit.warmup_epochs - crit.current_epoch - 1
                LOGGER.info(
                    f"  [UniqueLoss] warmup — {remaining} epoch(s) remaining before activation"
                )

    return on_train_start, on_train_epoch_start, on_train_epoch_end


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("Train YOLO12m with uniqueness loss")
    p.add_argument("--weights",        default="./runs/detect/runs/yolo12m/weights/best.pt",
                   help="Starting checkpoint (50-epoch pretrained on compound data)")
    p.add_argument("--data",           default="./data_mydata.yaml")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--imgsz",          type=int,   default=1024)
    p.add_argument("--batch",          type=int,   default=8)
    p.add_argument("--workers",        type=int,   default=2)
    p.add_argument("--device",         default="0")
    p.add_argument("--name",           default="yolo12m_unique")
    p.add_argument("--runs-dir",       default="./runs_mydata")
    p.add_argument("--results-dir",    default="./results_mydata")
    # uniqueness loss hyper-params
    p.add_argument("--uloss-weight",   type=float, default=0.05,
                   help="Scale factor for uniqueness loss (λ). Start small — 0.05.")
    p.add_argument("--uloss-thresh",   type=float, default=0.5,
                   help="Confidence threshold above which a duplicate is penalised.")
    p.add_argument("--warmup-epochs",  type=int,   default=5,
                   help="Epochs before uniqueness loss is activated (let detection stabilise).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  YOLO12m + Uniqueness Loss")
    print(f"  Weights        : {args.weights}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  uloss_weight   : {args.uloss_weight}")
    print(f"  uloss_thresh   : {args.uloss_thresh}")
    print(f"  warmup_epochs  : {args.warmup_epochs}")
    print(f"{'='*60}\n")

    model = YOLO(args.weights)

    cb_start, cb_epoch_start, cb_epoch_end = make_callbacks(
        args.uloss_weight, args.uloss_thresh, args.warmup_epochs
    )
    model.add_callback("on_train_start",       cb_start)
    model.add_callback("on_train_epoch_start", cb_epoch_start)
    model.add_callback("on_train_epoch_end",   cb_epoch_end)

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
        # keep the rest of hyper-params from the pretrained checkpoint
        resume=False,
    )

    print("\nRunning final validation...")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )

    box   = metrics.box
    prec  = float(box.mp)
    rec   = float(box.mr)
    f1    = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
    record = {
        "model":         args.weights,
        "name":          args.name,
        "epochs":        args.epochs,
        "uloss_weight":  args.uloss_weight,
        "uloss_thresh":  args.uloss_thresh,
        "warmup_epochs": args.warmup_epochs,
        "map50":         round(float(box.map50), 4),
        "map75":         round(float(box.map75), 4),
        "map50_95":      round(float(box.map),   4),
        "precision":     round(prec, 4),
        "recall":        round(rec,  4),
        "f1":            f1,
    }

    out = os.path.join(args.results_dir, f"{args.name}.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResults saved → {out}")

    run_csv = os.path.join(args.runs_dir, args.name, "results.csv")
    if os.path.exists(run_csv):
        dest = os.path.join(args.results_dir, f"{args.name}_losses.csv")
        shutil.copy2(run_csv, dest)
        print(f"Loss log   → {dest}")

    print(f"\n  mAP@50={record['map50']}  P={record['precision']}  "
          f"R={record['recall']}  F1={record['f1']}")


if __name__ == "__main__":
    main()
