#!/usr/bin/env python3
"""
Batch inference for MatDetect with either:
  - Hugging Face object detectors such as DAB-DETR/DETR checkpoints
  - Ultralytics YOLO .pt checkpoints

Writes one JSON file per input image.

Examples:
    python infer_any.py --image-dir ./testing --output-dir ./results --checkpoint ./checkpoints
    python infer_any.py --image-dir ./testing --output-dir ./results --checkpoint ./checkpoints/epoch046 --model-type hf
    python infer_any.py --image-dir ./testing --output-dir ./results --checkpoint ../ModelBench/runs/detect/runs/yolo11s/weights/best.pt --model-type yolo
"""

import argparse
import datetime
import json
import os
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


NUM_CLASSES = 22
ID2LABEL = {i: chr(ord("A") + i) for i in range(20)}
ID2LABEL[20] = "single"
ID2LABEL[21] = "common"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_checkpoint(path: str) -> str:
    path = os.path.normpath(path)
    best_txt = os.path.join(path, "best_path.txt")
    if os.path.isfile(best_txt):
        with open(best_txt) as f:
            resolved = f.read().strip()
        if not os.path.isabs(resolved):
            resolved = os.path.normpath(os.path.join(os.getcwd(), resolved))
        print(f"[info] best_path.txt -> {resolved}")
        return resolved

    if os.path.isdir(path):
        for candidate in (
            os.path.join(path, "weights", "best.pt"),
            os.path.join(path, "best.pt"),
            os.path.join(path, "weights", "last.pt"),
            os.path.join(path, "last.pt"),
        ):
            if os.path.isfile(candidate):
                print(f"[info] YOLO checkpoint -> {candidate}")
                return candidate

    return path


def infer_model_type(checkpoint: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if os.path.isfile(checkpoint) and checkpoint.lower().endswith(".pt"):
        return "yolo"
    if os.path.isdir(checkpoint) and os.path.isfile(os.path.join(checkpoint, "config.json")):
        return "hf"
    raise ValueError(
        "Could not infer model type. Use --model-type yolo or --model-type hf."
    )


def find_images(folder: str) -> list[str]:
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]


def label_name(label_id: int, names: Any = None) -> str:
    if isinstance(names, dict):
        return str(names.get(label_id, names.get(str(label_id), ID2LABEL.get(label_id, label_id))))
    if isinstance(names, list) and 0 <= label_id < len(names):
        return str(names[label_id])
    return ID2LABEL.get(label_id, str(label_id))


def load_hf(checkpoint: str, device: torch.device):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device)
    model.eval()
    return {"processor": processor, "model": model}


def predict_hf(handle: dict[str, Any], image: Image.Image, conf: float, iou: float, device: torch.device):
    width, height = image.size
    processor = handle["processor"]
    model = handle["model"]

    enc = processor(images=image, return_tensors="pt")
    outs = model(pixel_values=enc["pixel_values"].to(device))
    res = processor.post_process_object_detection(
        outs, threshold=conf, target_sizes=[(height, width)]
    )[0]

    pb = res["boxes"].cpu()
    ps = res["scores"].cpu()
    pl = res["labels"].cpu()
    if pb.numel() > 0:
        from torchvision.ops import batched_nms
        keep = batched_nms(pb, ps, pl, iou)
        pb = pb[keep]; ps = ps[keep]; pl = pl[keep]

    detections = []
    for box, score, label in zip(pb.tolist(), ps.tolist(), pl.tolist()):
        label = int(label)
        detections.append({
            "bbox": [round(float(v), 2) for v in box],
            "score": round(float(score), 6),
            "label_id": label,
            "label_name": label_name(label),
        })
    return detections


def load_yolo(checkpoint: str):
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    return {"model": model}


def predict_yolo(handle: dict[str, Any], image_path: str, conf: float, iou: float, device: torch.device):
    model = handle["model"]
    device_arg = device.index if device.type == "cuda" and device.index is not None else str(device)
    results = model.predict(image_path, conf=conf, iou=iou, device=device_arg, verbose=False)
    result = results[0]
    names = getattr(result, "names", getattr(model, "names", None))

    detections = []
    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    labels = result.boxes.cls.cpu().tolist()
    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        detections.append({
            "bbox": [round(float(v), 2) for v in box],
            "score": round(float(score), 6),
            "label_id": label,
            "label_name": label_name(label, names),
        })
    return detections


def write_record(
    out_path: str,
    image_path: str,
    width: int,
    height: int,
    detections: list[dict[str, Any]],
    model_type: str,
    checkpoint: str,
    conf: float,
    iou: float,
    device: torch.device,
) -> None:
    record = {
        "meta": {
            "model_type": model_type,
            "checkpoint": checkpoint,
            "conf": conf,
            "iou": iou,
            "device": str(device),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
        },
        "file": os.path.basename(image_path),
        "path": image_path,
        "width": width,
        "height": height,
        "n_detections": len(detections),
        "detections": detections,
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)


def get_args():
    p = argparse.ArgumentParser("MatDetect inference for YOLO or DAB-DETR/HF -> JSON")
    p.add_argument("--image-dir", required=True, help="Folder of images to run inference on")
    p.add_argument("--output-dir", required=True, help="Folder where JSON files will be saved")
    p.add_argument("--checkpoint", default="checkpoints", help="Checkpoint dir, best_path root, or YOLO .pt")
    p.add_argument("--model-type", choices=["auto", "hf", "yolo"], default="auto")
    p.add_argument("--conf", type=float, default=0.55, help="Confidence threshold (default: 0.3)")
    p.add_argument("--iou",  type=float, default=0.45, help="NMS IoU threshold (default: 0.5)")
    p.add_argument("--cuda-device", default="cuda:0")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = resolve_checkpoint(args.checkpoint)
    model_type = infer_model_type(checkpoint, args.model_type)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    image_paths = find_images(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.image_dir}")

    print(f"Model type : {model_type}")
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Conf       : {args.conf}")
    print(f"IoU (NMS)  : {args.iou}")
    print(f"Images     : {len(image_paths)}")
    print(f"Output dir : {args.output_dir}\n")

    if model_type == "hf":
        handle = load_hf(checkpoint, device)
    else:
        handle = load_yolo(checkpoint)

    n_saved = 0
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Inferring"):
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                width, height = image.size

                if model_type == "hf":
                    detections = predict_hf(handle, image, args.conf, args.iou, device)
                else:
                    detections = predict_yolo(handle, image_path, args.conf, args.iou, device)

            stem = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(args.output_dir, f"{stem}.json")
            write_record(
                out_path,
                image_path,
                width,
                height,
                detections,
                model_type,
                checkpoint,
                args.conf,
                args.iou,
                device,
            )
            n_saved += 1

    print(f"\nSaved {n_saved} JSON files -> {args.output_dir}/")


if __name__ == "__main__":
    main()
