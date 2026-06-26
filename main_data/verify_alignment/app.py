"""
Panel letter alignment verification tool.
Run build_sample.py once first, then start this.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

APP_DIR      = Path(__file__).parent
IMG_DIR      = APP_DIR / "data" / "images"
SAMPLE_FILE  = APP_DIR / "sample.json"
RESULTS_FILE = APP_DIR / "results.jsonl"
LABELS       = ["aligned", "misaligned", "uncertain"]

app = Flask(__name__)

if not SAMPLE_FILE.exists():
    raise FileNotFoundError("Run build_sample.py first.")

SAMPLE     = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
SAMPLE_MAP = {r["image_filename"]: i for i, r in enumerate(SAMPLE)}


def load_done():
    """Returns {filename: label} — last label wins (supports re-annotation)."""
    done = {}
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["image_filename"]] = r["label"]
    return done


def save_record(item, label):
    record = {
        "image_filename": item["image_filename"],
        "image_id":       item["image_id"],
        "panel_suffix":   item["panel_suffix"],
        "domain":         item["domain"],
        "subcaption":     item["subcaption"],
        "label":          label,
        "timestamp":      datetime.utcnow().isoformat(),
    }
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.route("/")
def index():
    done = load_done()
    current = next((r for r in SAMPLE if r["image_filename"] not in done), None)
    if current is None:
        return render_template("index.html", done=True,
                               total=len(SAMPLE), n_done=len(done))
    return render_template("index.html",
                           item=current,
                           idx=SAMPLE_MAP[current["image_filename"]],
                           total=len(SAMPLE),
                           n_done=len(done),
                           current_label=None,
                           revisit=False,
                           done=False)


@app.route("/revisit/<int:idx>")
def revisit(idx):
    if idx < 0 or idx >= len(SAMPLE):
        return "not found", 404
    done = load_done()
    item = SAMPLE[idx]
    return render_template("index.html",
                           item=item,
                           idx=idx,
                           total=len(SAMPLE),
                           n_done=len(done),
                           current_label=done.get(item["image_filename"]),
                           revisit=True,
                           done=False)


@app.route("/annotate", methods=["POST"])
def annotate():
    data     = request.get_json()
    label    = data.get("label", "").lower()
    filename = data.get("image_filename", "")
    revisit  = data.get("revisit", False)
    if label not in LABELS or not filename:
        return jsonify({"error": "invalid"}), 400
    done = load_done()
    if filename not in done or revisit:
        item = next((r for r in SAMPLE if r["image_filename"] == filename), None)
        if item:
            save_record(item, label)
    return jsonify({"ok": True})


@app.route("/skip", methods=["POST"])
def skip():
    data     = request.get_json()
    filename = data.get("image_filename", "")
    revisit  = data.get("revisit", False)
    done     = load_done()
    if filename not in done or revisit:
        item = next((r for r in SAMPLE if r["image_filename"] == filename), None)
        if item:
            save_record(item, "skip")
    return jsonify({"ok": True})


@app.route("/done_list")
def done_list():
    done = load_done()
    result = []
    for i, r in enumerate(SAMPLE):
        if r["image_filename"] in done:
            result.append({
                "idx":          i,
                "image_id":     r["image_id"],
                "panel_suffix": r["panel_suffix"],
                "domain":       r["domain"],
                "label":        done[r["image_filename"]],
            })
    return jsonify(result)


@app.route("/stats")
def stats():
    done = load_done()
    return jsonify({
        "total": len(SAMPLE), "done": len(done),
        "remaining": len(SAMPLE) - len(done),
        "counts": dict(Counter(done.values())),
    })


@app.route("/image/<filename>")
def serve_image(filename):
    path = IMG_DIR / filename
    if not path.exists():
        return "not found", 404
    return send_file(path)


if __name__ == "__main__":
    print(f"Loaded {len(SAMPLE)} samples | Results → {RESULTS_FILE}")
    app.run(debug=True, port=5050)
