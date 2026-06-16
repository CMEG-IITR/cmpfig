"""
Registry of all models to benchmark.
Each entry drives both train_yolo.py and train_hf.py automatically.

Fields:
  name       — unique run identifier (used as results/<name>.json)
  family     — "yolo" | "hf" | "gdino"
  model_id   — ultralytics model name OR HuggingFace model id
  variant    — short display label for the compare table
  eval_only  — skip training, run evaluation on the pretrained checkpoint
"""

MODELS = [
    # ── YOLO family (ultralytics) ────────────────────────────────────────────
    {"name": "yolov8n",  "family": "yolo", "model_id": "yolov8n.pt",   "variant": "YOLOv8-n"},
    {"name": "yolov8s",  "family": "yolo", "model_id": "yolov8s.pt",   "variant": "YOLOv8-s"},
    {"name": "yolov8m",  "family": "yolo", "model_id": "yolov8m.pt",   "variant": "YOLOv8-m"},

    {"name": "yolov9c",  "family": "yolo", "model_id": "yolov9c.pt",   "variant": "YOLOv9-c"},
    {"name": "yolov9e",  "family": "yolo", "model_id": "yolov9e.pt",   "variant": "YOLOv9-e"},

    {"name": "yolov10n", "family": "yolo", "model_id": "yolov10n.pt",  "variant": "YOLOv10-n"},
    {"name": "yolov10s", "family": "yolo", "model_id": "yolov10s.pt",  "variant": "YOLOv10-s"},
    {"name": "yolov10m", "family": "yolo", "model_id": "yolov10m.pt",  "variant": "YOLOv10-m"},
    {"name": "yolov10l", "family": "yolo", "model_id": "yolov10l.pt",  "variant": "YOLOv10-l"},

    {"name": "yolo11n",  "family": "yolo", "model_id": "yolo11n.pt",   "variant": "YOLO11-n"},
    {"name": "yolo11s",  "family": "yolo", "model_id": "yolo11s.pt",   "variant": "YOLO11-s"},
    {"name": "yolo11m",  "family": "yolo", "model_id": "yolo11m.pt",   "variant": "YOLO11-m"},
    {"name": "yolo11l",  "family": "yolo", "model_id": "yolo11l.pt",   "variant": "YOLO11-l"},

    {"name": "yolo12n",  "family": "yolo", "model_id": "yolo12n.pt",   "variant": "YOLO12-n"},
    {"name": "yolo12s",  "family": "yolo", "model_id": "yolo12s.pt",   "variant": "YOLO12-s"},
    {"name": "yolo12m",  "family": "yolo", "model_id": "yolo12m.pt",   "variant": "YOLO12-m"},
    {"name": "yolo12l",  "family": "yolo", "model_id": "yolo12l.pt",   "variant": "YOLO12-l"},

    # ── DAB-DETR — already trained in MatDetect, eval only ──────────────────
    {"name": "dabdetr",    "family": "hf", "model_id": "../MatDetect/checkpoints/epoch046",        "variant": "DAB-DETR",      "eval_only": True},

    # ── YOLOS (ViT-based, HuggingFace) ───────────────────────────────────────
    {"name": "yolos_ti",        "family": "hf", "model_id": "hustvl/yolos-tiny",          "variant": "YOLOS-Ti"},

    # ── RT-DETRv2 (ResNet-18 backbone, HuggingFace) ───────────────────────────
    {"name": "rtdetr_v2_r18vd", "family": "hf", "model_id": "PekingU/rtdetr_v2_r18vd",   "variant": "RT-DETRv2-R18"},
    {"name": "rtdetr_r18vd_coco", "family": "hf", "model_id": "PekingU/rtdetr_r18vd_coco_o365",   "variant": "RT-DETRv2-R18-Coco"},
]
