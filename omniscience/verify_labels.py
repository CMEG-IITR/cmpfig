#!/usr/bin/env python3
"""
Label verification viewer.
Opens a browser UI showing each image with YOLO bounding boxes + panel letters overlaid.

Usage:
    python verify_labels.py
    python verify_labels.py --data-dir ./omni_materials --port 5050
"""

import os
import json
import argparse
from pathlib import Path
from flask import Flask, jsonify, send_file, abort, render_template_string

# ── args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./omni_materials")
    p.add_argument("--port",     type=int, default=5050)
    return p.parse_args()

args = get_args()

IMAGES_DIR = os.path.join(args.data_dir, "images")
LABELS_DIR = os.path.join(args.data_dir, "labels")
LOG_FILE   = os.path.join(args.data_dir, "verified_correct.txt")
ID2LETTER  = {i: chr(ord("A") + i) for i in range(20)}

# load already-verified stems into a set (persists across server restarts)
def _load_verified():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE) as f:
        return set(line.strip() for line in f if line.strip())

verified_set = _load_verified()

stems = sorted(
    Path(f).stem for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

app = Flask(__name__)

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Label Verifier</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a2e; color: #eee; font-family: Arial, sans-serif; }

  #topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; background: #16213e; border-bottom: 1px solid #0f3460;
  }
  #topbar button {
    padding: 6px 18px; border: none; border-radius: 4px;
    background: #0f3460; color: #eee; font-size: 14px; cursor: pointer;
  }
  #topbar button:hover { background: #e94560; }
  #counter { font-size: 14px; min-width: 100px; text-align: center; }
  #imgname { font-size: 13px; color: #aaa; flex: 1; }
  #legend-bar { display: flex; flex-wrap: wrap; gap: 6px; }
  .legend-chip {
    padding: 2px 8px; border-radius: 3px; font-size: 12px; font-weight: bold;
  }

  #main { display: flex; justify-content: center; padding: 16px; }
  #canvas-wrap { position: relative; }
  canvas { display: block; border: 1px solid #333; }

  #info-panel {
    max-width: 900px; margin: 0 auto; padding: 0 16px 24px;
    font-size: 13px; color: #bbb; line-height: 1.6;
  }
  #info-panel .field { margin-bottom: 6px; }
  #info-panel .label { color: #e94560; font-weight: bold; margin-right: 6px; }

  #no-label { color: #e94560; text-align: center; padding: 10px; }
</style>
</head>
<body>

<div id="topbar">
  <button onclick="navigate(-1)">&#8592; Prev</button>
  <button onclick="navigate(1)">Next &#8594;</button>
  <button id="btn-correct" onclick="markCorrect()" style="background:#27ae60;">&#10003; All Correct</button>
  <span id="counter">0 / 0</span>
  <span id="verified-count" style="font-size:13px; color:#2ecc71;"></span>
  <span id="imgname"></span>
  <div id="legend-bar"></div>
</div>

<div id="main">
  <div id="canvas-wrap">
    <canvas id="canvas"></canvas>
  </div>
</div>
<div id="no-label" style="display:none">No labels for this image</div>

<div id="info-panel">
  <div class="field"><span class="label">Caption:</span><span id="f-caption"></span></div>
  <div class="field"><span class="label">Raw caption:</span><span id="f-rawcap"></span></div>
  <div class="field"><span class="label">Model:</span><span id="f-model"></span></div>
</div>

<script>
const TOTAL = {{ total }};
let idx = 0;

// 20 distinct colors for A-T
const COLORS = [
  '#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
  '#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4',
  '#8bc34a','#ff5722','#607d8b','#795548','#cddc39',
  '#ff9800','#03a9f4','#4caf50','#9c27b0','#f44336',
];

function classColor(cid) { return COLORS[cid % COLORS.length]; }

async function markCorrect() {
  const btn = document.getElementById('btn-correct');
  const res = await fetch(`/api/mark_correct/${idx}`, {method: 'POST'});
  const data = await res.json();
  if (data.ok) {
    btn.style.background = '#145a32';
    btn.textContent = '✓ Marked';
    document.getElementById('verified-count').textContent =
      `✓ ${data.total_verified} verified`;
  }
}

async function load(i) {
  idx = ((i % TOTAL) + TOTAL) % TOTAL;
  document.getElementById('counter').textContent = `${idx+1} / ${TOTAL}`;

  const res  = await fetch(`/api/sample/${idx}`);
  const data = await res.json();

  // update All Correct button state
  const btn = document.getElementById('btn-correct');
  if (data.verified) {
    btn.style.background = '#145a32';
    btn.textContent = '✓ Marked';
  } else {
    btn.style.background = '#27ae60';
    btn.textContent = '✓ All Correct';
  }
  document.getElementById('verified-count').textContent =
    data.total_verified > 0 ? `✓ ${data.total_verified} verified` : '';

  document.getElementById('imgname').textContent  = data.stem;
  document.getElementById('f-caption').textContent = data.caption || '';
  document.getElementById('f-rawcap').textContent  = data.raw_caption || '';
  document.getElementById('f-model').textContent   = data.recaption_model || '';

  const noLbl = document.getElementById('no-label');
  noLbl.style.display = data.labels.length === 0 ? 'block' : 'none';

  // draw image + boxes
  const img = new Image();
  img.src = `/img/${data.stem}?t=${Date.now()}`;
  img.onload = () => {
    const canvas = document.getElementById('canvas');
    const maxW = window.innerWidth - 40;
    const scale = Math.min(1.0, maxW / img.width);
    const W = Math.round(img.width  * scale);
    const H = Math.round(img.height * scale);
    canvas.width  = W;
    canvas.height = H;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, W, H);

    // draw each box
    const legendBar = document.getElementById('legend-bar');
    legendBar.innerHTML = '';
    const seen = new Set();

    for (const lb of data.labels) {
      const [cid, xc, yc, w, h] = lb;
      const letter = String.fromCharCode(65 + cid);
      const color  = classColor(cid);

      const x1 = (xc - w/2) * W;
      const y1 = (yc - h/2) * H;
      const bw = w * W;
      const bh = h * H;

      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      ctx.strokeRect(x1, y1, bw, bh);

      // filled label tag
      const tag   = letter;
      ctx.font     = 'bold 14px Arial';
      const tw     = ctx.measureText(tag).width + 8;
      const ty     = y1 > 20 ? y1 - 20 : y1 + bh + 2;
      ctx.fillStyle = color;
      ctx.fillRect(x1, ty, tw, 18);
      ctx.fillStyle = '#fff';
      ctx.fillText(tag, x1 + 4, ty + 14);

      if (!seen.has(cid)) {
        seen.add(cid);
        const chip = document.createElement('span');
        chip.className = 'legend-chip';
        chip.style.background = color;
        chip.style.color = '#fff';
        chip.textContent = letter;
        legendBar.appendChild(chip);
      }
    }
  };
}

function navigate(dir) { load(idx + dir); }

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'd') navigate(1);
  if (e.key === 'ArrowLeft'  || e.key === 'a') navigate(-1);
});

load(0);
</script>
</body>
</html>
"""

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML, total=len(stems))


@app.route("/img/<stem>")
def serve_image(stem):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(IMAGES_DIR, stem + ext)
        if os.path.exists(p):
            return send_file(p)
    abort(404)


@app.route("/api/mark_correct/<int:idx>", methods=["POST"])
def mark_correct(idx):
    if idx < 0 or idx >= len(stems):
        abort(404)
    stem = stems[idx]
    if stem not in verified_set:
        verified_set.add(stem)
        with open(LOG_FILE, "a") as f:
            f.write(stem + "\n")
    return jsonify({"ok": True, "stem": stem, "total_verified": len(verified_set)})


@app.route("/api/sample/<int:idx>")
def sample(idx):
    if idx < 0 or idx >= len(stems):
        abort(404)
    stem = stems[idx]

    # read YOLO labels
    lbl_path = os.path.join(LABELS_DIR, stem + ".txt")
    labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([int(parts[0])] + [float(x) for x in parts[1:]])

    # read metadata from CSV (lazy: re-read each time, fine for small dataset)
    caption = raw_caption = recaption_model = ""
    csv_path = os.path.join(args.data_dir, "metadata.csv")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if Path(row.get("image_name", "")).stem == stem:
                    caption         = row.get("caption", "")
                    raw_caption     = row.get("raw_caption", "")
                    recaption_model = row.get("recaption_model", "")
                    break

    return jsonify({
        "stem":            stem,
        "labels":          labels,
        "caption":         caption,
        "raw_caption":     raw_caption,
        "recaption_model": recaption_model,
        "verified":        stem in verified_set,
        "total_verified":  len(verified_set),
    })


if __name__ == "__main__":
    print(f"Loaded {len(stems)} images from {IMAGES_DIR}")
    print(f"Open http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
