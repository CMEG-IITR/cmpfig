#!/usr/bin/env python3
"""
Batch inference for MatDetect (DAB-DETR panel detector).
Reads every image from a folder, runs the best checkpoint, and writes
a single JSON file with detections + metadata to a separate output folder.

Usage:
    python infer.py --image-dir ./testing --output-dir ./results
    python infer.py --image-dir ./testing --output-dir ./results --checkpoint ./checkpoints/epoch046
    python infer.py --image-dir ./testing --output-dir ./results --best-of checkpoints_mydata_all
"""

import os
import json
import argparse
import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection


NUM_CLASSES = 22
ID2LABEL = {i: chr(ord("A") + i) for i in range(20)}
ID2LABEL[20] = "single"
ID2LABEL[21] = "common"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_checkpoint(checkpoint: str) -> str:
    best_txt = os.path.join(checkpoint, "best_path.txt")
    if os.path.isfile(best_txt):
        with open(best_txt) as f:
            resolved = f.read().strip()
        print(f"[info] best_path.txt → {resolved}")
        return resolved
    return checkpoint


def find_images(folder: str) -> list[str]:
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]


def get_args():
    p = argparse.ArgumentParser("MatDetect — inference → JSON")
    p.add_argument("--image-dir",   required=True,
                   help="Folder of images to run inference on")
    p.add_argument("--output-dir",  required=True,
                   help="Folder where the output JSON will be saved (created if missing)")
    p.add_argument("--checkpoint",  default=None,
                   help="HuggingFace checkpoint dir (overrides --best-of)")
    p.add_argument("--best-of",     default="checkpoints",
                   help="Checkpoint root with best_path.txt (default: checkpoints)")
    p.add_argument("--conf",        type=float, default=0.55,
                   help="Confidence score threshold (default: 0.3)")
    p.add_argument("--iou",         type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.5)")
    p.add_argument("--cuda-device", default="cuda:0")
    return p.parse_args()


def main():
    args = get_args()

    # ── output dir ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── resolve checkpoint ──────────────────────────────────────────────────────
    ckpt_path = resolve_checkpoint(args.checkpoint or args.best_of)
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    # ── device ──────────────────────────────────────────────────────────────────
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Conf       : {args.conf}")
    print(f"IoU (NMS)  : {args.iou}")
    print(f"Output dir : {args.output_dir}")

    # ── load model ──────────────────────────────────────────────────────────────
    processor = AutoImageProcessor.from_pretrained(ckpt_path)
    model = AutoModelForObjectDetection.from_pretrained(ckpt_path).to(device)
    model.eval()

    # ── collect images ──────────────────────────────────────────────────────────
    image_paths = find_images(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.image_dir}")
    print(f"Images     : {len(image_paths)}\n")

    # ── run inference ───────────────────────────────────────────────────────────
    n_saved = 0

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Inferring"):
            pil = Image.open(img_path).convert("RGB")
            W, H = pil.size

            enc  = processor(images=pil, return_tensors="pt")
            outs = model(pixel_values=enc["pixel_values"].to(device))
            res  = processor.post_process_object_detection(
                       outs, threshold=args.conf, target_sizes=[(H, W)]
                   )[0]

            boxes  = res["boxes"].cpu()
            scores = res["scores"].cpu()
            labels = res["labels"].cpu()
            if boxes.numel() > 0:
                from torchvision.ops import batched_nms
                keep   = batched_nms(boxes, scores, labels, args.iou)
                boxes  = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            detections = []
            for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                detections.append({
                    "bbox":       [round(v, 2) for v in box],  # [x1, y1, x2, y2]
                    "score":      round(score, 6),
                    "label_id":   label,
                    "label_name": ID2LABEL.get(label, str(label)),
                })

            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(args.output_dir, f"{stem}.json")

            record = {
                "meta": {
                    "checkpoint": ckpt_path,
                    "conf":       args.conf,
                    "iou":        args.iou,
                    "device":     str(device),
                    "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
                    "id2label":   {str(k): v for k, v in ID2LABEL.items()},
                },
                "file":         os.path.basename(img_path),
                "path":         img_path,
                "width":        W,
                "height":       H,
                "n_detections": len(detections),
                "detections":   detections,
            }

            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            n_saved += 1

    print(f"\nSaved {n_saved} JSON files → {args.output_dir}/")


if __name__ == "__main__":
    main()
