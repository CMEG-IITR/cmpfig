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
    {"name": "yolov8m",  "family": "yolo", "model_id": "yolov8m.pt",   "variant": "YOLOv8-m"},

    {"name": "yolov9c",  "family": "yolo", "model_id": "yolov9c.pt",   "variant": "YOLOv9-c"},
    {"name": "yolov10m", "family": "yolo", "model_id": "yolov10m.pt",  "variant": "YOLOv10-m"},
    {"name": "yolo11m",  "family": "yolo", "model_id": "yolo11m.pt",   "variant": "YOLO11-m"},
    {"name": "yolo12m",  "family": "yolo", "model_id": "yolo12m.pt",   "variant": "YOLO12-m"},

    # ── DAB-DETR — already trained in MatDetect, eval only ──────────────────
    {"name": "dabdetr",    "family": "hf", "model_id": "../MatDetect/checkpoints/epoch046",        "variant": "DAB-DETR",      "eval_only": True},
]
