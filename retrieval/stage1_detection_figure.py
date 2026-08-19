"""
Stage 1 only: the original compound figure with YOLO12-m detection boxes
drawn on it (color-coded per panel letter), cropped tight for direct use in
a paper. No title, no crop grid, no annotation cards -- just the boxed image.

Bounding boxes are the real cached YOLO12-m output already logged to disk
(`*_prod_json/*.json`); falls back to re-running the checkpoint only if a
chosen figure has no cached detection JSON.

Usage:
    python -m retrieval.stage1_detection_figure
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────
MAIN_DATA_DIR = "/mnt/d/Subham/Compoun_img_01/main_data"
DETECTION_JSON_DIR = os.path.join(MAIN_DATA_DIR, "ceramics_prod_json")
SOURCE_FIGURES_DIR = os.path.join(MAIN_DATA_DIR, "ceramics_production_images", "ceramics_production_images")
YOLO_CKPT_PATH = os.path.join(MAIN_DATA_DIR, "yolo12_unique_multimat.pt")  # fallback only
OUTPUT_PATH = "/mnt/d/Subham/Compoun_img_01/retrieval/paper_figures/stage1_detection.pdf"

IMAGE_ID = "ceramics_prod_img38203"

LETTER_COLORS = {
    "A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd",
    "E": "#ff7f0e", "F": "#17becf",
}


def color_for(letter):
    return LETTER_COLORS.get(letter, "gray")


def run_yolo_fallback(image_path, ckpt_path):
    from ultralytics import YOLO
    model = YOLO(ckpt_path)
    result = model.predict(source=image_path, conf=0.6, iou=0.4, imgsz=1024, verbose=False)[0]
    id2label = result.names
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(box.conf[0]),
            "label_id": int(box.cls[0]),
            "label_name": id2label[int(box.cls[0])],
        })
    return detections


def main():
    det_json_path = os.path.join(DETECTION_JSON_DIR, f"{IMAGE_ID}.json")
    if os.path.exists(det_json_path):
        det_data = json.load(open(det_json_path))
        detections = det_data["detections"]
        source_image_path = os.path.join(SOURCE_FIGURES_DIR, det_data["file"])
        print(f"Using cached YOLO12-m detections: {det_json_path}")
    else:
        source_image_path = os.path.join(SOURCE_FIGURES_DIR, f"{IMAGE_ID}.jpg")
        print(f"No cached detection JSON for {IMAGE_ID}; re-running YOLO12-m checkpoint ...")
        detections = run_yolo_fallback(source_image_path, YOLO_CKPT_PATH)

    orig_img = Image.open(source_image_path).convert("RGB")
    print(f"Source image: {source_image_path}  ({orig_img.width}x{orig_img.height}px)")

    fig, ax = plt.subplots(figsize=(orig_img.width / 200, orig_img.height / 200))
    ax.imshow(orig_img)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for det in detections:
        letter = det["label_name"]
        x1, y1, x2, y2 = det["bbox"]
        c = color_for(letter)
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         fill=False, edgecolor=c, linewidth=3))
        ax.text(x1 + 6, y1 + 6, f"({letter.lower()})", fontsize=13, weight="bold",
                color="white", va="top",
                bbox=dict(boxstyle="square,pad=0.15", facecolor=c, edgecolor="none"))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.savefig(os.path.splitext(OUTPUT_PATH)[0] + ".png", format="png", dpi=300,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
