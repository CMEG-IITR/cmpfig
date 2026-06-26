"""
Appends single-panel crops to the existing sample.json.
Does NOT touch already-annotated items.
Takes first PER_DOMAIN singles from each domain.
"""

import json, shutil
from pathlib import Path

PER_DOMAIN = 25
APP_DIR    = Path(__file__).parent
BASE       = APP_DIR.parent
IMG_DIR    = APP_DIR / "data" / "images"
SAMPLE_OUT = APP_DIR / "sample.json"

IMG_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = {
    "alloy": {
        "crops": BASE / "alloy_prod_crops",
        "llm":   BASE / "generated_subcaptions_alloy_prod",
    },
    "ceramics": {
        "crops": BASE / "ceramics_prod_crops",
        "llm":   BASE / "generated_subcaptions_ceramics_prod",
    },
    "composite": {
        "crops": BASE / "composite_prod_crops",
        "llm":   BASE / "generated_subcaptions_composite_prod",
    },
    "ni_alloy": {
        "crops": BASE / "ni_prod_crops",
        "llm":   BASE / "generated_subcaptions_ni_alloy_prod",
    },
}

# load existing sample so we don't duplicate
existing = json.loads(SAMPLE_OUT.read_text(encoding="utf-8")) if SAMPLE_OUT.exists() else []
existing_filenames = {r["image_filename"] for r in existing}

new_items = []

for domain, cfg in DOMAINS.items():
    crops_dir = cfg["crops"]
    llm_dir   = cfg["llm"]

    llm_stems = {f.stem for f in llm_dir.iterdir() if f.suffix == ".json"}

    picked = []
    for crop in crops_dir.iterdir():
        if len(picked) >= PER_DOMAIN:
            break
        if crop.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        stem  = crop.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        img_id, panel = parts[0], parts[1].upper()
        if panel != "SINGLE":
            continue
        if img_id not in llm_stems:
            continue
        picked.append((crop, img_id, panel))

    print(f"  {domain}: {len(picked)} singles found")

    for crop, img_id, panel in picked:
        dst_img = IMG_DIR / f"{domain}__{crop.name}"

        if dst_img.name in existing_filenames:
            continue

        llm_file = llm_dir / f"{img_id}.json"
        llm = json.loads(llm_file.read_text(encoding="utf-8"))
        if "error" in llm:
            continue

        subcaption = ""
        for p in llm.get("panels", []):
            if p.get("panel", "").strip().lower() == "main":
                subcaption = p.get("subcaption", "").strip()
                break
        if not subcaption:
            continue

        shutil.copy2(crop, dst_img)
        new_items.append({
            "image_filename": dst_img.name,
            "image_id":       img_id,
            "panel_suffix":   "single",
            "domain":         domain,
            "subcaption":     subcaption,
        })

combined = existing + new_items
SAMPLE_OUT.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"\nAdded {len(new_items)} singles → total {len(combined)} in sample.json")
