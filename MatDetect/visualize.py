#!/usr/bin/env python3
"""
Visualise MatDetect (DAB-DETR) predictions on a folder of images.

Draws predicted bounding boxes (panel labels A–Z + confidence score).
Optionally overlays ground-truth boxes in a different colour if a labels/
folder is present alongside the images/ folder.

Usage:
    python visualize.py \
        --checkpoint  ./checkpoints/epoch046 \
        --image-dir   ./data/val/images \
        --output-dir  ./vis_output \
        --conf        0.3 \
        --max-images  20
"""

import os
import argparse
import random

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection


NUM_CLASSES = 22
ID2LABEL    = {i: chr(ord("A") + i) for i in range(20)}
ID2LABEL[20] = "single"
ID2LABEL[21] = "common"

# Distinct colours for each panel label (A–Z)
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
    "#e6beff", "#008080", "#ff6600", "#cc00cc", "#00cc66",
    "#6666ff",
]


def get_font(size: int = 14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


def yolo_to_xyxy_pil(boxes_raw, W, H):
    """List of [xc,yc,bw,bh] (norm) → list of (x1,y1,x2,y2) (abs pixel)."""
    out = []
    for xc, yc, bw, bh in boxes_raw:
        x1 = (xc - bw / 2) * W
        y1 = (yc - bh / 2) * H
        x2 = (xc + bw / 2) * W
        y2 = (yc + bh / 2) * H
        out.append((x1, y1, x2, y2))
    return out


def read_labels(path):
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


def draw_boxes(pil_img: Image.Image,
               pred_boxes, pred_scores, pred_labels,
               gt_boxes=None, gt_classes=None,
               line_width: int = 2) -> Image.Image:
    """Draw predictions (solid) and GT (dashed-look via double rect) onto image."""
    draw = ImageDraw.Draw(pil_img)
    font_big  = get_font(15)
    font_small = get_font(12)
    W, H = pil_img.size

    # ── ground truth (thin white + label name, no fill) ──────────────────────
    if gt_boxes is not None and gt_classes is not None:
        for (x1, y1, x2, y2), cls in zip(gt_boxes, gt_classes):
            color = PALETTE[cls % len(PALETTE)]
            draw.rectangle([x1, y1, x2, y2], outline="white",   width=1)
            draw.rectangle([x1+1, y1+1, x2-1, y2-1], outline=color, width=1)
            label = ID2LABEL.get(cls, str(cls))
            draw.text((x1 + 2, y1 + 2), f"GT:{label}", fill="white", font=font_small)

    # ── predictions (thick coloured box + label + score) ─────────────────────
    for box, score, cls in zip(pred_boxes, pred_scores, pred_labels):
        x1, y1, x2, y2 = box.tolist()
        cls   = int(cls)
        color = PALETTE[cls % len(PALETTE)]
        label = ID2LABEL.get(cls, str(cls))
        text  = f"{label} {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # badge background
        try:
            bbox = font_big.getbbox(text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = font_big.getsize(text)

        bx1 = x1
        by1 = max(0, y1 - th - 4)
        bx2 = bx1 + tw + 6
        by2 = by1 + th + 4
        draw.rectangle([bx1, by1, bx2, by2], fill=color)
        draw.text((bx1 + 3, by1 + 2), text, fill="black", font=font_big)

    return pil_img


# ── main ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("MatDetect — visualisation")
    p.add_argument("--checkpoint",   default="./checkpoints/epoch046")
    p.add_argument("--image-dir",    default="./data/val/images",
                   help="folder containing .jpg/.png files to visualise")
    p.add_argument("--output-dir",   default="./vis_output")
    p.add_argument("--conf",         type=float, default=0.3,
                   help="Confidence score threshold (default: 0.3)")
    p.add_argument("--iou",          type=float, default=0.5,
                   help="NMS IoU threshold (default: 0.5)")
    p.add_argument("--max-images",   type=int,   default=50,
                   help="limit how many images to process (0 = all)")
    p.add_argument("--show-gt",      action="store_true",
                   help="overlay GT boxes if a ../labels/ folder exists")
    p.add_argument("--cuda-device",  default="cuda:0")
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
    print(f"Device : {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    model     = AutoModelForObjectDetection.from_pretrained(args.checkpoint).to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # optional GT labels
    labels_dir = os.path.join(os.path.dirname(args.image_dir.rstrip("/")), "labels")
    has_labels = args.show_gt and os.path.isdir(labels_dir)
    if args.show_gt and not has_labels:
        print(f"[warn] --show-gt set but no labels dir found at {labels_dir}")

    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if args.max_images and args.max_images > 0:
        stems = stems[:args.max_images]
    print(f"Images : {len(stems)}   →   saving to {args.output_dir}/")

    n_saved = 0
    with torch.no_grad():
        for stem in tqdm(stems, desc="Visualising"):
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(args.image_dir, stem + ext)
                if os.path.exists(p):
                    img_path = p; break
            if img_path is None:
                continue

            pil  = Image.open(img_path).convert("RGB")
            W, H = pil.size

            enc          = processor(images=pil, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device)

            outputs = model(pixel_values=pixel_values)
            results = processor.post_process_object_detection(
                outputs, threshold=args.conf, target_sizes=[(H, W)]
            )[0]

            pred_boxes  = results["boxes"].cpu()
            pred_scores = results["scores"].cpu()
            pred_labels = results["labels"].cpu()

            if pred_boxes.numel() > 0:
                from torchvision.ops import batched_nms
                keep        = batched_nms(pred_boxes, pred_scores, pred_labels, args.iou)
                pred_boxes  = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

            # sort by score descending for drawing order
            order       = pred_scores.argsort(descending=True)
            pred_boxes  = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_labels = pred_labels[order]

            gt_boxes_pil = gt_classes_list = None
            if has_labels:
                lbl_path = os.path.join(labels_dir, stem + ".txt")
                if os.path.exists(lbl_path):
                    gt_raw, gt_cls = read_labels(lbl_path)
                    gt_boxes_pil    = yolo_to_xyxy_pil(gt_raw, W, H)
                    gt_classes_list = gt_cls

            annotated = draw_boxes(
                pil.copy(),
                pred_boxes, pred_scores, pred_labels,
                gt_boxes=gt_boxes_pil, gt_classes=gt_classes_list,
            )

            n_pred = len(pred_boxes)
            out_name = f"{stem}_pred{n_pred}.jpg"
            annotated.save(os.path.join(args.output_dir, out_name), quality=90)
            n_saved += 1

    print(f"\nSaved {n_saved} annotated images to: {args.output_dir}/")
    print("Legend: coloured box = prediction (Label Score)   white outline = GT (if --show-gt)")


if __name__ == "__main__":
    main()
