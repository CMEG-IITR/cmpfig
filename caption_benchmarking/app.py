import csv
import json
import threading
from collections import OrderedDict
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
RESULTS_PATH = BASE_DIR / "benchmark_results.json"
MODEL_DIRS = [
    "gemini_3.1_flash_lite_outputs",
    "gemini_3.5_flash_outputs",
    "gpt54nano_outputs",
    "gpt54minioutputs",
    "DeepSeek_V4_Flash_outputs",
    "mistral_large_3_outputs",
]
MODEL_LABELS = {
    "gemini_3.1_flash_lite_outputs": "Gemini 3.1 Flash Lite",
    "gemini_3.5_flash_outputs": "Gemini 3.5 Flash",
    "gpt54nano_outputs": "GPT-5.4 Nano",
    "gpt54minioutputs": "GPT-5.4 Mini",
    "DeepSeek_V4_Flash_outputs": "DeepSeek V4 Flash",
    "mistral_large_3_outputs": "Mistral Large 3",
}

app = Flask(__name__)
_results_lock = threading.Lock()


def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as f:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(f)]


ROWS = load_dataset()
IMAGE_INDEX = {row["image_name"]: idx for idx, row in enumerate(ROWS)}


def safe_read_json(path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def write_results(results):
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results():
    return safe_read_json(RESULTS_PATH, {})


def get_row(index):
    if index < 0 or index >= len(ROWS):
        abort(404)
    return ROWS[index]


def normalize_panel_id(pid):
    """Canonicalize panel IDs so 'A', 'a', '(a)' all map to the same key."""
    return str(pid).strip().strip("()").strip().upper()


def panel_sort_key(p):
    v = normalize_panel_id(p.get("panel", ""))
    return (len(v), v)


def _image_status(image_name, results):
    saved = results.get(image_name, {})
    cat = saved.get("category_benchmark", {})
    summ = saved.get("summary_benchmark", {})
    cat_done = sum(1 for m in MODEL_DIRS if cat.get(m))
    sum_done = sum(1 for m in MODEL_DIRS if summ.get(m))
    n = len(MODEL_DIRS)
    if cat_done == n and sum_done == n:
        status = "complete"
    elif cat_done > 0 or sum_done > 0:
        status = "partial"
    else:
        status = "empty"
    return status, bool(saved.get("flagged", False))


def build_review_context(index):
    row = get_row(index)
    results = load_results()
    image_name = row["image_name"]
    saved = results.get(image_name, {})
    cat_saved = saved.get("category_benchmark", {})
    sum_saved = saved.get("summary_benchmark", {})

    all_panel_ids = OrderedDict()
    model_panels = {}
    for model_dir in MODEL_DIRS:
        path = BASE_DIR / model_dir / f"{image_name}.json"
        data = safe_read_json(path, {})
        panels = data.get("panels", []) if isinstance(data, dict) else []
        sorted_panels = sorted(panels, key=panel_sort_key)
        model_panels[model_dir] = {normalize_panel_id(p["panel"]): p for p in sorted_panels}
        for p in sorted_panels:
            all_panel_ids[normalize_panel_id(p["panel"])] = True

    # Panel-centric: one row per panel, one column per model
    cat_panel_groups = []
    for panel_id in all_panel_ids:
        cells = []
        for model_dir in MODEL_DIRS:
            p = model_panels[model_dir].get(panel_id)
            if p:
                cells.append({
                    "model": model_dir,
                    "category": p.get("visualization_category", ""),
                    "subtype": p.get("visualization_subtype", ""),
                    "cat_saved": cat_saved.get(model_dir, {}).get(panel_id, {}).get("category_correct", None),
                    "sub_saved": cat_saved.get(model_dir, {}).get(panel_id, {}).get("subtype_correct", None),
                })
            else:
                cells.append(None)
        cat_panel_groups.append({"panel_id": panel_id, "cells": cells})

    # Panel-centric for summary: one row per panel, one column per model (parallel view)
    sum_panel_groups = []
    for panel_id in all_panel_ids:
        cells = []
        for model_dir in MODEL_DIRS:
            p = model_panels[model_dir].get(panel_id)
            cell = {
                "model": model_dir,
                "model_label": MODEL_LABELS.get(model_dir, model_dir),
                "available": p is not None,
            }
            if p:
                cell.update({
                    "subcaption": p.get("subcaption", ""),
                    "summary": p.get("summary", ""),
                    "subcap_saved": sum_saved.get(model_dir, {}).get(panel_id, {}).get("subcaption_quality", ""),
                    "sum_saved": sum_saved.get(model_dir, {}).get(panel_id, {}).get("summary_quality", ""),
                    "hallu_saved": sum_saved.get(model_dir, {}).get(panel_id, {}).get("hallucination", ""),
                })
            cells.append(cell)
        sum_panel_groups.append({"panel_id": panel_id, "cells": cells})

    flagged = bool(saved.get("flagged", False))
    prev_index = index - 1 if index > 0 else None
    next_index = index + 1 if index + 1 < len(ROWS) else None

    return {
        "row": row,
        "index": index,
        "total": len(ROWS),
        "model_labels": [MODEL_LABELS.get(m, m) for m in MODEL_DIRS],
        "cat_panel_groups": cat_panel_groups,
        "sum_panel_groups": sum_panel_groups,
        "flagged": flagged,
        "prev_index": prev_index,
        "next_index": next_index,
        "next_url": url_for("review_page", index=next_index) if next_index is not None else None,
        "prev_url": url_for("review_page", index=prev_index) if prev_index is not None else None,
    }


def merge_annotation(image_name, section, payload):
    with _results_lock:
        results = load_results()
        results.setdefault(image_name, {}).setdefault(section, {}).update(payload)
        write_results(results)


@app.route("/")
def index():
    return redirect(url_for("review_page", index=0))


@app.route("/review/<int:index>")
def review_page(index):
    return render_template("review.html", **build_review_context(index))


@app.route("/category/<int:index>")
def category_page(index):
    return redirect(url_for("review_page", index=index))


@app.route("/summary/<int:index>")
def summary_page(index):
    return redirect(url_for("review_page", index=index))


@app.route("/progress")
def progress():
    results = load_results()
    items = []
    for idx, row in enumerate(ROWS):
        status, flagged = _image_status(row["image_name"], results)
        items.append({"index": idx, "image_name": row["image_name"], "status": status, "flagged": flagged})
    n_complete = sum(1 for it in items if it["status"] == "complete")
    n_partial = sum(1 for it in items if it["status"] == "partial")
    return render_template("progress.html", items=items, total=len(ROWS),
                           n_complete=n_complete, n_partial=n_partial)


@app.route("/image/<image_name>")
def image(image_name):
    idx = IMAGE_INDEX.get(image_name)
    if idx is None:
        abort(404)
    image_path = (BASE_DIR / ROWS[idx]["img_path"]).resolve()
    if not image_path.is_file() or BASE_DIR not in image_path.parents:
        abort(404)
    return send_file(image_path)


@app.route("/save/category", methods=["POST"])
def save_category():
    payload = request.get_json(force=True)
    image_name = payload.get("image_name")
    if image_name not in IMAGE_INDEX:
        abort(400)
    merge_annotation(image_name, "category_benchmark", payload.get("annotations", {}))
    return jsonify({"ok": True})


@app.route("/save/summary", methods=["POST"])
def save_summary():
    payload = request.get_json(force=True)
    image_name = payload.get("image_name")
    if image_name not in IMAGE_INDEX:
        abort(400)
    merge_annotation(image_name, "summary_benchmark", payload.get("annotations", {}))
    return jsonify({"ok": True})


@app.route("/save/flag", methods=["POST"])
def save_flag():
    payload = request.get_json(force=True)
    image_name = payload.get("image_name")
    if image_name not in IMAGE_INDEX:
        abort(400)
    with _results_lock:
        results = load_results()
        results.setdefault(image_name, {})["flagged"] = bool(payload.get("flagged", True))
        write_results(results)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
