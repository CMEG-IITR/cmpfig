"""
Build a linked CSV dataset pairing each cropped image with its JSON metadata.

Linking rules (only these two cases handled):
  imgXXXX_single.jpg  ->  JSON panel "main"
  imgXXXX_A.jpg       ->  JSON panel "a" or "A"  (case-insensitive match)
  imgXXXX_B_2.jpg     ->  JSON panel "b" or "B"  (variant crop, same panel)

Complex JSON panel keys (a1, a1-a6, d-e, etc.) are intentionally skipped.

Outputs:
  linked_dataset.csv
  build_dataset.log
"""

import csv
import json
import os
import re
from datetime import datetime

IMAGES_DIR = "crops_allloys_img_0.55"
JSON_DIR   = "generated_subcaptions"
OUTPUT_CSV = "linked_dataset.csv"
OUTPUT_LOG = "build_dataset.log"

IMG_RE = re.compile(r"^(img\d+)_(single|[A-Z])(?:_(\d+))?\.jpg$", re.IGNORECASE)

CSV_FIELDS = [
    "image_filename",
    "image_id",
    "panel_suffix",
    "variant",
    "json_panel",
    "visualization_type",
    "subcaption",
    "summary",
    "matched",
]


def load_json_index(json_dir):
    index = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        img_id = fname[:-5]
        with open(os.path.join(json_dir, fname), encoding="utf-8") as fh:
            try:
                index[img_id] = json.load(fh)
            except json.JSONDecodeError:
                pass
    return index


def build_panel_lookup(json_obj):
    # Only keep simple single-letter keys and "main"; skip a1, a1-a6, d-e, etc.
    lookup = {}
    for panel in json_obj.get("panels", []):
        key = panel.get("panel", "").strip()
        if key == "main" or (len(key) == 1 and key.isalpha()):
            lookup[key.lower()] = panel
    return lookup


def main():
    json_index = load_json_index(JSON_DIR)
    image_files = sorted(os.listdir(IMAGES_DIR))

    rows = []
    skipped_pattern  = []   # filenames that didn't match IMG_RE
    no_json          = []   # image has no corresponding JSON file
    panel_not_found  = []   # JSON exists but panel key missing / complex
    matched_rows     = []

    for fname in image_files:
        m = IMG_RE.match(fname)
        if not m:
            skipped_pattern.append(fname)
            continue

        img_id       = m.group(1)
        panel_letter = m.group(2)
        variant      = m.group(3) or ""

        if panel_letter.lower() == "single":
            panel_suffix = "single"
            lookup_key   = "main"
        else:
            panel_suffix = f"{panel_letter.upper()}_{variant}" if variant else panel_letter.upper()
            lookup_key   = panel_letter.lower()

        row = {
            "image_filename"  : fname,
            "image_id"        : img_id,
            "panel_suffix"    : panel_suffix,
            "variant"         : variant,
            "json_panel"      : "",
            "visualization_type": "",
            "subcaption"      : "",
            "summary"         : "",
            "matched"         : False,
        }

        json_obj = json_index.get(img_id)
        if json_obj is None:
            no_json.append(fname)
        else:
            panel_data = build_panel_lookup(json_obj).get(lookup_key)
            if panel_data:
                row["json_panel"]          = panel_data.get("panel", "")
                row["visualization_type"]  = panel_data.get("visualization_type", "")
                row["subcaption"]          = panel_data.get("subcaption", "")
                row["summary"]             = panel_data.get("summary", "")
                row["matched"]             = True
                matched_rows.append(fname)
            else:
                panel_not_found.append(fname)

        rows.append(row)

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ── Stats ──────────────────────────────────────────────────────────────────
    total_images  = len(image_files)
    total_rows    = len(rows)           # after skipping bad patterns
    n_matched     = len(matched_rows)
    n_no_json     = len(no_json)
    n_no_panel    = len(panel_not_found)
    n_skipped     = len(skipped_pattern)

    pct_matched   = n_matched  / total_rows * 100 if total_rows else 0
    pct_no_json   = n_no_json  / total_rows * 100 if total_rows else 0
    pct_no_panel  = n_no_panel / total_rows * 100 if total_rows else 0

    # Caption linkage: matched rows that have both subcaption and summary filled
    n_has_caption = sum(
        1 for r in rows if r["matched"] and r["subcaption"] and r["summary"]
    )
    pct_caption   = n_has_caption / total_rows * 100 if total_rows else 0

    log_lines = [
        f"build_dataset.py  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Total image files scanned  : {total_images}",
        f"Skipped (bad filename)     : {n_skipped}",
        f"Processed rows             : {total_rows}",
        "",
        "── Image → JSON linkage ──────────────────────────────────",
        f"  Matched (image + panel)  : {n_matched:>6}  ({pct_matched:.1f}%)",
        f"  No JSON file             : {n_no_json:>6}  ({pct_no_json:.1f}%)",
        f"  JSON found, panel missing: {n_no_panel:>6}  ({pct_no_panel:.1f}%)",
        "",
        "── Caption linkage (of total processed) ─────────────────",
        f"  Has subcaption + summary : {n_has_caption:>6}  ({pct_caption:.1f}%)",
        "",
        "── Output ────────────────────────────────────────────────",
        f"  CSV  : {OUTPUT_CSV}",
        f"  Log  : {OUTPUT_LOG}",
    ]

    if skipped_pattern:
        log_lines += ["", "── Skipped filenames (bad pattern) ──────────────────────"]
        log_lines += [f"  {f}" for f in skipped_pattern]

    log_text = "\n".join(log_lines)

    with open(OUTPUT_LOG, "w", encoding="utf-8") as fh:
        fh.write(log_text + "\n")

    print(log_text)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
