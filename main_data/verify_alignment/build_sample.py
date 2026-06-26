"""
Run once. Copies 25 crop images + matching LLM subcaptions from each domain
into verify_alignment/data/images/ and builds sample.json.
Total: 100 items (first 25 x 4 domains).
"""

import json, shutil
from pathlib import Path

PER_DOMAIN = 250
APP_DIR    = Path(__file__).parent
BASE       = APP_DIR.parent          # main_data/
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

sample = []

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
        if panel in ("SINGLE", "COMMON"):
            continue
        if img_id not in llm_stems:
            continue
        picked.append((crop, img_id, panel, llm_dir / f"{img_id}.json"))

    print(f"  {domain}: {len(picked)} picked")

    for crop, img_id, panel, llm_file in picked:
        # copy image
        dst_img = IMG_DIR / f"{domain}__{crop.name}"
        shutil.copy2(crop, dst_img)

        # read subcaption
        llm = json.loads(llm_file.read_text(encoding="utf-8"))
        subcaption = ""
        for p in llm.get("panels", []):
            if p.get("panel", "").strip().upper() == panel:
                subcaption = p.get("subcaption", "").strip()
                break

        if not subcaption:
            dst_img.unlink()
            continue

        sample.append({
            "image_filename": dst_img.name,
            "image_id":       img_id,
            "panel_suffix":   panel,
            "domain":         domain,
            "subcaption":     subcaption,
        })

SAMPLE_OUT.write_text(json.dumps(sample, indent=2), encoding="utf-8")
print(f"\nDone — {len(sample)} items saved to {SAMPLE_OUT}")
print(f"Images in {IMG_DIR}")
