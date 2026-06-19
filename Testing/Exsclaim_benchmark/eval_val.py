"""
Evaluate yolov11_finetuned_augmentation_best.pt on val/images + val/labels.

Two outputs:
  eval_results/metrics.txt     — mAP50, mAP50-95, P, R, F1 per class and summary
  eval_results/vis/            — one JPG per val image: GT (green) vs predicted (red)
  eval_results/summary_grid.jpg — all val images tiled in a 2-column grid

Model classes (ids 0–7): a b c d e f g h
GT labels may contain class ids > 7 (shown in orange, model cannot predict these).

Usage:
    python eval_val.py
"""

from pathlib import Path
import sys
import tempfile
import time

SCRIPT_DIR   = Path(__file__).parent
EXSCLAIM_DIR = SCRIPT_DIR / "exsclaim2.0"
if EXSCLAIM_DIR.exists():
    sys.path.insert(0, str(EXSCLAIM_DIR))

from ultralytics import YOLO
import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT = EXSCLAIM_DIR / "exsclaim/figures/checkpoints/yolov11_finetuned_augmentation_best.pt"
VAL_IMAGES = SCRIPT_DIR / "test/images"
VAL_LABELS = SCRIPT_DIR / "test/labels"
OUT_DIR    = SCRIPT_DIR / "test_results"
VIS_DIR    = OUT_DIR / "vis"

CONF  = 0.55
IOU   = 0.45
IMGSZ = 640

CLASS_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]   # model's 8 classes

# Colours (BGR): GT known class = green, GT unknown class = orange, Pred = red
GT_COLOR      = (0, 200, 0)
GT_UNK_COLOR  = (0, 140, 255)   # orange — GT class id > 7, model can't predict
PRED_COLOR    = (0, 0, 220)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ── Helpers ───────────────────────────────────────────────────────────────────

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Normalised YOLO → absolute pixel xyxy."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return (
        max(0, x1), max(0, y1),
        min(img_w, x2), min(img_h, y2),
    )


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """Return list of (class_id, x1, y1, x2, y2)."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_label(img, text, x1, y1, color):
    """Draw a small filled label box above the bounding box."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
    lx1, ly1 = x1, max(0, y1 - th - baseline - 2)
    lx2, ly2 = x1 + tw + 4, y1
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.putText(img, text, (lx1 + 2, ly2 - baseline), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_image(img, gt_boxes, pred_boxes):
    """
    Draw GT and predicted boxes on a copy of img.
    gt_boxes:   list of (class_id, x1, y1, x2, y2)
    pred_boxes: list of (class_id, x1, y1, x2, y2, conf)
    """
    out = img.copy()

    for cls, x1, y1, x2, y2 in gt_boxes:
        if cls < len(CLASS_NAMES):
            color = GT_COLOR
            label = CLASS_NAMES[cls].upper()
        else:
            color = GT_UNK_COLOR
            label = f"cls{cls}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        draw_label(out, label, x1, y1, color)

    for cls, x1, y1, x2, y2, conf in pred_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), PRED_COLOR, 2)
        name = CLASS_NAMES[cls].upper() if cls < len(CLASS_NAMES) else str(cls)
        draw_label(out, f"{name} {conf:.2f}", x1, y1, PRED_COLOR)

    return out


def add_legend(img):
    h, w = img.shape[:2]
    panel = np.full((46, w, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "GREEN = ground truth (A B C…)", (10, 18), FONT, 0.48, GT_COLOR,     1, cv2.LINE_AA)
    cv2.putText(panel, "ORANGE = GT class unknown to model", (10, 38), FONT, 0.48, GT_UNK_COLOR, 1, cv2.LINE_AA)
    cv2.putText(panel, "RED = predicted (A 0.92…)", (w // 2 + 20, 28), FONT, 0.48, PRED_COLOR, 1, cv2.LINE_AA)
    return np.vstack([img, panel])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not CHECKPOINT.exists():
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)
    if not VAL_IMAGES.exists():
        print(f"[ERROR] Val images folder not found: {VAL_IMAGES}")
        sys.exit(1)

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model  : {CHECKPOINT.name}")
    print(f"Val images     : {VAL_IMAGES}  ({len(list(VAL_IMAGES.glob('*.jpg')))} images)")
    print(f"Val labels     : {VAL_LABELS}")
    model = YOLO(str(CHECKPOINT))

    # ── 1. Official YOLO val metrics ──────────────────────────────────────────
    # 31 of the 142 images have GT boxes with class id > 7 (e.g. class 20, 21).
    # YOLO marks the whole image "corrupt" and skips it when nc=8.
    #
    # Fix: build a proper temp dataset directory:
    #   tmp_val/images/  — per-file symlinks (path stays tmp_val/... so YOLO
    #                       derives labels from tmp_val/labels/, not val/labels/)
    #   tmp_val/labels/  — filtered copies keeping only class id < 8
    # Also delete any stale .cache files so YOLO rescans with the filtered labels.
    import shutil as _shutil

    tmp_val = OUT_DIR / "_tmp_val"
    tmp_img = tmp_val / "images"
    tmp_lbl = tmp_val / "labels"

    # Rebuild from scratch to avoid stale state
    if tmp_val.exists():
        _shutil.rmtree(tmp_val)
    tmp_img.mkdir(parents=True)
    tmp_lbl.mkdir(parents=True)

    # Per-file image symlinks so paths are tmp_val/images/imgXXX.jpg
    for img in VAL_IMAGES.iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            (tmp_img / img.name).symlink_to(img.resolve())

    # Filtered label files
    skipped_boxes = 0
    for lf in VAL_LABELS.glob("*.txt"):
        lines = []
        for l in lf.read_text().splitlines():
            if not l.strip():
                continue
            if int(l.split()[0]) < len(CLASS_NAMES):
                lines.append(l)
            else:
                skipped_boxes += 1
        (tmp_lbl / lf.name).write_text("\n".join(lines) + ("\n" if lines else ""))

    # Delete any stale YOLO label cache
    for cache in [tmp_lbl / "labels.cache", VAL_LABELS / "labels.cache"]:
        cache.unlink(missing_ok=True)

    print(f"  Filtered out {skipped_boxes} GT boxes with class id ≥ 8 (shown as orange in vis).")

    tmp_yaml = tmp_val / "dataset.yaml"
    names_str = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    tmp_yaml.write_text(
        f"path: {tmp_val.resolve()}\n"
        f"train: images\n"
        f"val:   images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_str}\n"
    )

    n_imgs = len(list(tmp_img.iterdir()))
    print(f"\nRunning official YOLO validation on all {n_imgs} images …")
    metrics = model.val(
        data=str(tmp_yaml),
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        split="val",
        verbose=True,
        project=str(OUT_DIR),
        name="yolo_val",
        exist_ok=True,
    )

    # Per-class arrays from YOLO metrics
    ap50      = metrics.box.ap50        # (nc,)  AP @ IoU=0.50
    ap5095    = metrics.box.ap          # (nc,)  AP @ IoU=0.50:0.95
    all_ap    = metrics.box.all_ap      # (nc,10) AP at IoU 0.50,0.55,...,0.95
    ap75      = all_ap[:, 5]            # (nc,)  AP @ IoU=0.75  (index 5 = 0.5+5*0.05)
    prec      = metrics.box.p           # (nc,)
    rec       = metrics.box.r           # (nc,)
    f1        = metrics.box.f1          # (nc,)

    # TP / FP / FN / GT from the confusion matrix
    # matrix layout (ultralytics): matrix[pred_cls][gt_cls]
    #   TP[i]  = matrix[i][i]                        — correctly detected
    #   FP[i]  = sum(row i) - matrix[i][i]           — pred i not matching GT i
    #              (includes unmatched preds + cross-class confusions)
    #   FN[i]  = sum(col i) - matrix[i][i]           — GT i not matched to pred i
    #              (includes missed GT + GT matched to wrong class)
    #   GT[i]  = sum(col i) = TP[i] + FN[i]          — actual GT instances of class i
    cm   = np.array(metrics.confusion_matrix.matrix)   # (nc+1, nc+1)
    nc   = len(CLASS_NAMES)
    tp_c = [int(cm[i, i])                      for i in range(nc)]
    fp_c = [int(cm[i, :].sum() - cm[i, i])     for i in range(nc)]
    fn_c = [int(cm[:, i].sum() - cm[i, i])     for i in range(nc)]
    gt_c = [int(cm[:, i].sum())                for i in range(nc)]  # TP + FN

    val_imgs = sorted(VAL_IMAGES.glob("*.jpg")) + sorted(VAL_IMAGES.glob("*.png"))

    def predict_boxes(img_path: Path):
        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            verbose=False,
        )
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                pred_boxes.append((cls, x1, y1, x2, y2, conf))
        return pred_boxes

    # Manual wall-clock timing, matching save_results-style FPS:
    # warm up 3 images, then average model.predict() elapsed ms per image.
    for img_path in val_imgs[:3]:
        predict_boxes(img_path)

    timed_predictions = {}
    infer_times_ms = []
    for img_path in val_imgs:
        t0 = time.perf_counter()
        pred_boxes = predict_boxes(img_path)
        infer_times_ms.append((time.perf_counter() - t0) * 1000.0)
        timed_predictions[img_path] = pred_boxes

    # ── Print & save table ────────────────────────────────────────────────────
    header = (
        f"\n{'Class':>6} | {'GT':>5} | {'AP@50':>7} | {'AP@75':>7} | {'AP@50:95':>9} | "
        f"{'P':>7} | {'R':>7} | {'F1':>7} | "
        f"{'TP':>6} | {'FP':>6} | {'FN':>6}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for i in range(nc):
        name = CLASS_NAMES[i].upper()
        rows.append(
            f"{name:>6} | {gt_c[i]:>5d} | {ap50[i]:>7.4f} | {ap75[i]:>7.4f} | {ap5095[i]:>9.4f} | "
            f"{prec[i]:>7.4f} | {rec[i]:>7.4f} | {f1[i]:>7.4f} | "
            f"{tp_c[i]:>6d} | {fp_c[i]:>6d} | {fn_c[i]:>6d}"
        )

    rows.append(sep)
    def _mean(arr): return float(sum(arr) / len(arr)) if len(arr) else 0.0
    mean_ap5095 = _mean(ap5095)
    mean_ap50 = _mean(ap50)
    mean_f1 = _mean(f1)
    mean_p = _mean(prec)
    mean_r = _mean(rec)
    total_tp = sum(tp_c)
    total_fp = sum(fp_c)
    total_fn = sum(fn_c)
    infer_ms = float(np.mean(infer_times_ms)) if infer_times_ms else 0.0
    fps = round(1000.0 / infer_ms, 2) if infer_ms > 0 else 0.0
    params_m = sum(p.numel() for p in model.model.parameters()) / 1_000_000
    gflops = 0.0
    model_info = getattr(model.model, "info", None)
    if callable(model_info):
        try:
            info = model_info(verbose=False, imgsz=IMGSZ)
        except TypeError:
            info = model_info(verbose=False)
        if isinstance(info, (tuple, list)) and info:
            gflops = float(info[-1])
    rows.append(
        f"{'MEAN':>6} | {sum(gt_c):>5d} | {mean_ap50:>7.4f} | {_mean(ap75):>7.4f} | {mean_ap5095:>9.4f} | "
        f"{mean_p:>7.4f} | {mean_r:>7.4f} | {mean_f1:>7.4f} | "
        f"{total_tp:>6d} | {total_fp:>6d} | {total_fn:>6d}"
    )
    summary_text = (
        "mAP@50:95\tmAP@50\tmean_F1\tmean_P\tmean_R\tparams_M\tgflops\tinfer_ms\tfps\ttotal_TP\ttotal_FP\ttotal_FN\n"
        f"{mean_ap5095:.4f}\t{mean_ap50:.4f}\t{mean_f1:.4f}\t{mean_p:.4f}\t{mean_r:.4f}\t"
        f"{params_m:.2f}\t{gflops:.2f}\t{infer_ms:.2f}\t{fps:.2f}\t{total_tp}\t{total_fp}\t{total_fn}"
    )
    rows.append("\nsummary\n" + summary_text)

    table = "\n".join(rows)
    print(table)
    (OUT_DIR / "metrics.txt").write_text(table)
    (OUT_DIR / "summary.tsv").write_text(summary_text)
    print(f"\nMetrics saved → {OUT_DIR / 'metrics.txt'}")
    print(f"Summary saved → {OUT_DIR / 'summary.tsv'}")

    # ── 2. Per-image visualisation ────────────────────────────────────────────
    print("\nGenerating per-image visualisations …")

    for img_path in val_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] cannot read {img_path.name}")
            continue

        ih, iw = img.shape[:2]

        # Ground truth
        label_path = VAL_LABELS / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(label_path, iw, ih)

        pred_boxes = timed_predictions.get(img_path, [])

        vis = annotate_image(img, gt_boxes, pred_boxes)
        vis = add_legend(vis)

        out_path = VIS_DIR / f"{img_path.stem}_eval.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {img_path.name}  GT={len(gt_boxes)} boxes  Pred={len(pred_boxes)} boxes → {out_path.name}")

    # ── 3. Summary grid (all val images tiled) ───────────────────────────────
    print("\nBuilding summary grid …")
    vis_files = sorted(VIS_DIR.glob("*_eval.jpg"))
    tiles = []
    TARGET_W = 640
    for vf in vis_files:
        tile = cv2.imread(str(vf))
        if tile is None:
            continue
        h, w = tile.shape[:2]
        scale = TARGET_W / w
        tile  = cv2.resize(tile, (TARGET_W, int(h * scale)), interpolation=cv2.INTER_AREA)
        tiles.append(tile)

    if tiles:
        # Stack into 2-column grid
        cols = 2
        rows_needed = (len(tiles) + cols - 1) // cols
        # Pad to even count
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))

        rows_imgs = []
        for r in range(rows_needed):
            row_tiles = tiles[r * cols: r * cols + cols]
            # equalise heights
            max_h = max(t.shape[0] for t in row_tiles)
            padded = []
            for t in row_tiles:
                diff = max_h - t.shape[0]
                if diff:
                    t = np.vstack([t, np.zeros((diff, t.shape[1], 3), dtype=np.uint8)])
                padded.append(t)
            rows_imgs.append(np.hstack(padded))

        grid = np.vstack(rows_imgs)
        grid_path = OUT_DIR / "summary_grid.jpg"
        cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"Summary grid saved → {grid_path}")

    print("\nAll done!")
    print(f"  Metrics  : {OUT_DIR / 'metrics.txt'}")
    print(f"  Per-image: {VIS_DIR}/")
    print(f"  Grid     : {OUT_DIR / 'summary_grid.jpg'}")


if __name__ == "__main__":
    main()
