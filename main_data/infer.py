#!/usr/bin/env python3
"""
Batch inference for MatDetect using a YOLO best.pt checkpoint.
Reads every image from a folder, runs the model, and writes one JSON file per image
(plus a combined summary JSON) to the output folder.

Usage:
    python infer.py --image-dir ./testing --output-dir ./results
    python infer.py --image-dir ./testing --output-dir ./results --weights ./best.pt
    python infer.py --image-dir ./testing --output-dir ./results --conf 0.4 --iou 0.45
"""

import argparse
import datetime
import json
import os

from PIL import Image
from tqdm import tqdm

NUM_CLASSES = 22
ID2LABEL = {i: chr(ord("A") + i) for i in range(20)}
ID2LABEL[20] = "single"
ID2LABEL[21] = "common"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(folder: str) -> list[str]:
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]


def get_args():
    p = argparse.ArgumentParser("MatDetect — YOLO inference → JSON")
    p.add_argument("--image-dir",   required=True,
                   help="Folder of images to run inference on")
    p.add_argument("--output-dir",  required=True,
                   help="Folder where output JSON files will be saved (created if missing)")
    p.add_argument("--weights",     default="best.pt",
                   help="Path to YOLO best.pt weights (default: best.pt)")
    p.add_argument("--conf",        type=float, default=0.6,
                   help="Confidence threshold (default: 0.55)")
    p.add_argument("--iou",         type=float, default=0.4,
                   help="NMS IoU threshold (default: 0.4)")
    p.add_argument("--imgsz",       type=int,   default=1024,
                   help="Inference image size (default: 1024)")
    p.add_argument("--device",      default="",
                   help="Device: '' for auto, 'cpu', '0', '0,1', etc.")
    return p.parse_args()


def main():
    args = get_args()

    from ultralytics import YOLO

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    print(f"Weights    : {args.weights}")
    print(f"Conf       : {args.conf}")
    print(f"IoU (NMS)  : {args.iou}")
    print(f"Imgsz      : {args.imgsz}")
    print(f"Output dir : {args.output_dir}")

    model = YOLO(args.weights)

    image_paths = find_images(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.image_dir}")
    print(f"Images     : {len(image_paths)}\n")

    all_records = []

    for img_path in tqdm(image_paths, desc="Inferring"):
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size

        results = model.predict(
            source=img_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        r = results[0]

        detections = []
        if r.boxes is not None and len(r.boxes):
            for box, score, cls in zip(
                r.boxes.xyxy.tolist(),
                r.boxes.conf.tolist(),
                r.boxes.cls.int().tolist(),
            ):
                detections.append({
                    "bbox":       [round(v, 2) for v in box],  # [x1, y1, x2, y2]
                    "score":      round(score, 6),
                    "label_id":   cls,
                    "label_name": ID2LABEL.get(cls, str(cls)),
                })

        stem = os.path.splitext(os.path.basename(img_path))[0]
        record = {
            "meta": {
                "weights":   args.weights,
                "conf":      args.conf,
                "iou":       args.iou,
                "imgsz":     args.imgsz,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "id2label":  {str(k): v for k, v in ID2LABEL.items()},
            },
            "file":         os.path.basename(img_path),
            "path":         img_path,
            "width":        W,
            "height":       H,
            "n_detections": len(detections),
            "detections":   detections,
        }

        out_path = os.path.join(args.output_dir, f"{stem}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        all_records.append(record)

    summary_path = os.path.join(args.output_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\nSaved {len(all_records)} JSON files → {args.output_dir}/")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
