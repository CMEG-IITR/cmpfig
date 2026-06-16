"""
Registry for mydata_all retraining.
Each model_id points to the best fine-tuned checkpoint from the original
benchmark run (trained on MatDetect/data/train) so training resumes from
those weights rather than starting from COCO pretrained.
"""

MODELS_MYDATA = [
    # ── YOLO family — resuming from best fine-tuned .pt ─────────────────────
    {"name": "yolov8m",  "family": "yolo", "model_id": "./runs/detect/runs/yolov8m/weights/best.pt",  "variant": "YOLOv8-m"},

    {"name": "yolov9c",  "family": "yolo", "model_id": "./runs/detect/runs/yolov9c/weights/best.pt",  "variant": "YOLOv9-c"},
    {"name": "yolov10m", "family": "yolo", "model_id": "./runs/detect/runs/yolov10m/weights/best.pt", "variant": "YOLOv10-m"},
    {"name": "yolo11m",  "family": "yolo", "model_id": "./runs/detect/runs/yolo11m/weights/best.pt",  "variant": "YOLO11-m"},
    {"name": "yolo12m",  "family": "yolo", "model_id": "./runs/detect/runs/yolo12m/weights/best.pt",  "variant": "YOLO12-m"},

    # ── DAB-DETR — trained on mydata_all in MatDetect, eval only ─────────────
    {"name": "dabdetr",  "family": "hf",   "model_id": "../MatDetect/checkpoints/epoch046", "variant": "DAB-DETR", "eval_only": True},
]
