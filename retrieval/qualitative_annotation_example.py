"""
End-to-end MatSciFig pipeline qualitative figure: one compound figure, all
three pipeline stages shown side by side.

  Stage 1 (YOLO12-m detector)  -> bounding boxes + panel letters on the
                                   original compound figure
  Stage 2 (crop)               -> individual panel crops
  Stage 3 (Gemini annotation)  -> per-panel category / subtype / subcaption /
                                   summary, pulled verbatim from the dataset

Data provenance / what is and isn't fabricated
------------------------------------------------
- Bounding boxes are NOT re-run through YOLO -- they're the actual detector
  output already logged to disk at inference time (`*_prod_json/*.json`,
  produced by `ModelBench/runs/detect/runs_mydata/yolo12m_unique/weights/best.pt`
  == `main_data/yolo12_unique_multimat.pt`). YOLO_CKPT_PATH is kept below only
  as a fallback for re-running detection on a figure that has no cached JSON.
- Category/subtype/subcaption/summary are read verbatim from
  `dataset_index.json` (the flattened per-panel MatSciFig index; no separate
  parquet/HF export exists locally, this IS the dataset).
- The original figure's full caption + in-text reference sentence (the text
  Gemini was conditioned on) is NOT persisted anywhere in this pipeline's
  output on disk -- it was only ever consumed transiently as LLM input and
  discarded afterward. Every JSON file under main_data/ was inspected
  (detection jsons, generated_subcaptions_*, linked_dataset.csv) and none of
  them retain it. Rather than fabricate that text, the figure explicitly
  labels it "not persisted in this pipeline's stored outputs" instead of
  printing invented caption text.

Usage:
    python -m retrieval.qualitative_annotation_example
"""
import glob
import json
import os
import random
import textwrap
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────
MAIN_DATA_DIR = "/mnt/d/Subham/Compoun_img_01/main_data"
DATASET_INDEX_PATH = os.path.join(MAIN_DATA_DIR, "dataset_index.json")   # per-panel MatSciFig index (no local parquet export)
DETECTION_JSON_DIR = os.path.join(MAIN_DATA_DIR, "ceramics_prod_json")   # cached real YOLO12-m outputs, one JSON per source figure
SOURCE_FIGURES_DIR = os.path.join(MAIN_DATA_DIR, "ceramics_production_images", "ceramics_production_images")
YOLO_CKPT_PATH = os.path.join(MAIN_DATA_DIR, "yolo12_unique_multimat.pt")  # fallback only, used if a chosen figure has no cached detection JSON
OUTPUT_PATH = "/mnt/d/Subham/Compoun_img_01/retrieval/paper_figures/qualitative_annotation_example_01.pdf"

IMAGE_ID = "ceramics_prod_img607"  # set to None to auto-select via `auto_select_candidate()` below
AUTO_SELECT_SAMPLE_SIZE = 400
AUTO_SELECT_SEED = 3

# Recovered from the source paper (not stored anywhere in this pipeline's own
# outputs -- see the provenance note in the module docstring). Supplied by
# the user, who located the actual paper text for this figure.
ORIGINAL_CAPTION = ( 
    "Graphical illustration of (a) L-PBF and (b) LP-DED specimens during fabrication along with their coordinate system. Note that red boxes are used to indicate to areas where SEM images of the surface were collected, which are depicted for (c) L-PBF and (d) LP-DED specimens."
)
ORIGINAL_REFERENCE = (
    "Fig. 1 offers a graphical representations of the specimenâ€™s orientation relative to the build plate for (a) L-PBF and (b) LP-DED specimens. || Furthermore, Fig. 1 displays SEM images of the surface for (c) L-PBF spcimens and (d) LP-DED specimens."
)

LETTER_COLORS = {
    "A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd",
    "E": "#ff7f0e", "F": "#17becf",
}


# ── Candidate selection (mirrors the exploration used to pick IMAGE_ID) ─────

def auto_select_candidate(detection_dir, index_by_id, sample_size, seed):
    """Scan cached detection JSONs for a clean 3-5 panel figure with aligned
    dataset annotations and, ideally, more than one visualization_category
    (taxonomy diversity) -- clean detection quality is weighted first."""
    files = sorted(glob.glob(os.path.join(detection_dir, "*.json")))
    rng = random.Random(seed)
    sample = rng.sample(files, min(sample_size, len(files)))

    best = None
    for f in sample:
        d = json.load(open(f))
        img_id = os.path.splitext(os.path.basename(f))[0]
        dets = d.get("detections", [])
        labels = [x["label_name"] for x in dets]
        if not (3 <= len(dets) <= 5):
            continue
        if any(l in ("single", "common") for l in labels) or len(set(labels)) != len(labels):
            continue
        rows = index_by_id.get(img_id, [])
        if len(rows) != len(dets):
            continue  # need full alignment between detections and annotations
        n_cats = len(set(r["visualization_category"] for r in rows))
        min_score = min(x["score"] for x in dets)
        key = (n_cats, min_score)
        if best is None or key > best[0]:
            best = (key, img_id)
    if best is None:
        raise RuntimeError("No clean 3-5 panel candidate found in the sampled detection JSONs.")
    return best[1]


def run_yolo_fallback(image_path, ckpt_path):
    """Only used if a chosen figure has no cached detection JSON."""
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


# ── Helpers ────────────────────────────────────────────────────────────────

def wrap(text, width, max_words=None):
    text = str(text)
    if max_words is not None:
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "..."
    return "\n".join(textwrap.wrap(text, width=width))


def make_text_cursor(fig, ax, start_y=0.95, indent=0.0):
    """Returns draw(text, size, color, weight, gap_after_pt) that places text
    at a descending y-cursor on `ax` (axes-fraction coords) and advances the
    cursor by the text's ACTUAL rendered height (measured via the renderer),
    not a hand-tuned estimate -- exact regardless of font size, axes height,
    or how many lines the text wraps to, so it never over/under-fills."""
    y = [start_y]

    def draw(text, size, color, weight="normal", gap_after_pt=6.0):
        txt = ax.text(indent, y[0], text, fontsize=size, weight=weight, color=color,
                      va="top", ha="left", linespacing=1.25, transform=ax.transAxes)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_axes = txt.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        ax_height_in = ax.get_window_extent(renderer=renderer).height / fig.dpi
        gap_frac = (gap_after_pt / 72) / ax_height_in
        y[0] = bbox_axes.y0 - gap_frac

    return draw


def color_for(letter):
    return LETTER_COLORS.get(letter, "gray")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Loading dataset index: {DATASET_INDEX_PATH}")
    index = json.load(open(DATASET_INDEX_PATH))
    index_by_id = defaultdict(list)
    for r in index:
        index_by_id[r["image_id"]].append(r)

    image_id = IMAGE_ID
    if image_id is None:
        print(f"Auto-selecting a candidate from {DETECTION_JSON_DIR} ...")
        image_id = auto_select_candidate(DETECTION_JSON_DIR, index_by_id, AUTO_SELECT_SAMPLE_SIZE, AUTO_SELECT_SEED)
    print(f"Selected image_id: {image_id}")

    det_json_path = os.path.join(DETECTION_JSON_DIR, f"{image_id}.json")
    det_meta = {"weights": os.path.basename(YOLO_CKPT_PATH), "conf": "-", "iou": "-"}
    if os.path.exists(det_json_path):
        det_data = json.load(open(det_json_path))
        detections = det_data["detections"]
        source_image_path = os.path.join(SOURCE_FIGURES_DIR, det_data["file"])
        det_meta = {"weights": os.path.basename(det_data["meta"]["weights"]),
                    "conf": det_data["meta"]["conf"], "iou": det_data["meta"]["iou"]}
        print(f"Using cached YOLO12-m detections: {det_json_path}")
    else:
        source_image_path = os.path.join(SOURCE_FIGURES_DIR, f"{image_id}.jpg")
        print(f"No cached detection JSON for {image_id}; re-running YOLO12-m checkpoint ...")
        detections = run_yolo_fallback(source_image_path, YOLO_CKPT_PATH)

    rows = {r["panel_suffix"]: r for r in index_by_id[image_id]}
    detections = sorted(detections, key=lambda d: d["label_name"])
    panel_letters = [d["label_name"] for d in detections]
    missing = [l for l in panel_letters if l not in rows]
    if missing:
        raise RuntimeError(f"Detected panels {missing} have no matching dataset_index.json annotation for {image_id}.")

    print(f"Panels: {panel_letters}  (categories: "
          f"{Counter(rows[l]['visualization_category'] for l in panel_letters)})")

    orig_img = Image.open(source_image_path).convert("RGB")
    n_panels = len(panel_letters)

    # ── Figure: col0 = original + boxes (spans all rows), col1 = crops, col2 = annotation cards
    TOP_H = 5.0          # inches -- fits the original image at this column width, with headroom for the caption/reference text
    ROW_H = 3.2           # inches per panel row for the annotation-card grid below
    fig = plt.figure(figsize=(13.5, TOP_H + ROW_H * n_panels))
    outer = fig.add_gridspec(2, 1, height_ratios=[TOP_H, ROW_H * n_panels], hspace=0.06)

    # -- Part A (top): original figure with detector boxes, plus provenance note --
    top_gs = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.06)
    ax_orig = fig.add_subplot(top_gs[0, 0])
    ax_orig.imshow(orig_img)
    ax_orig.set_xticks([]); ax_orig.set_yticks([])
    for det in detections:
        letter = det["label_name"]
        x1, y1, x2, y2 = det["bbox"]
        c = color_for(letter)
        ax_orig.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor=c, linewidth=3))
        ax_orig.text(x1 + 6, y1 + 6, f"({letter.lower()})", fontsize=13, weight="bold",
                     color="white", va="top",
                     bbox=dict(boxstyle="square,pad=0.15", facecolor=c, edgecolor="none"))
    ax_note = fig.add_subplot(top_gs[0, 1])
    ax_note.axis("off")
    ax_note.set_xlim(0, 1); ax_note.set_ylim(0, 1)

    draw_note = make_text_cursor(fig, ax_note, start_y=0.98)
    draw_note("Original caption:", 12.5, "black", weight="bold", gap_after_pt=4)
    draw_note(wrap(ORIGINAL_CAPTION, 44), 11.5, "black", gap_after_pt=16)
    draw_note("In-text reference:", 12.5, "black", weight="bold", gap_after_pt=4)
    draw_note(wrap(ORIGINAL_REFERENCE, 44), 11.5, "black", gap_after_pt=16)

    # -- Part C (bottom grid): one annotation card per panel, full width --
    # (no repeated crop thumbnail here -- the crops are already visible with
    # their color-matched boxes in the Stage 1 panel above)
    bottom_gs = outer[1].subgridspec(n_panels, 1, hspace=0.22)

    WRAP_WIDTH = 85

    for r, letter in enumerate(panel_letters):
        c = color_for(letter)
        row = rows[letter]
        ax_card = fig.add_subplot(bottom_gs[r, 0])
        ax_card.axis("off")
        ax_card.set_xlim(0, 1); ax_card.set_ylim(0, 1)
        ax_card.add_patch(mpatches.FancyBboxPatch(
            (0.005, 0.04), 0.99, 0.92, boxstyle="round,pad=0.02", transform=ax_card.transAxes,
            fill=False, edgecolor=c, linewidth=2.5))

        draw_block = make_text_cursor(fig, ax_card, start_y=0.88, indent=0.025)
        draw_block(f"({letter.lower()})  {row['visualization_category']}  —  {row['visualization_subtype']}",
                   13.0, c, weight="bold", gap_after_pt=5)
        draw_block("Subcaption:  " + wrap(row["subcaption"], WRAP_WIDTH), 11.0, "black", gap_after_pt=8)
        draw_block("Summary:  " + wrap(row["summary"], WRAP_WIDTH), 10.5, "black", gap_after_pt=8)

    plt.tight_layout(rect=[0, 0, 1, 1])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # bbox_inches="tight" trims any residual whitespace margin so the saved
    # figure is cropped tight to its content -- journal figures shouldn't
    # carry blank page margin baked into the image.
    plt.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(os.path.splitext(OUTPUT_PATH)[0] + ".png", format="png", dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
