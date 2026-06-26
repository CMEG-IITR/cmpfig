"""
Counts non-standard panel keys across all generated_subcaptions_* JSONs.
Standard keys (single letters a-z/A-Z and "main") are excluded from the output.
Saves results to audit_panel_keys.txt.
"""

import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

BASE    = Path(__file__).parent
OUT_TXT = BASE / "audit_panel_keys.txt"

def is_standard(key):
    k = key.strip()
    return k == "main" or (len(k) == 1 and k.isalpha())

subcap_dirs = sorted(d for d in BASE.iterdir()
                     if d.is_dir() and d.name.startswith("generated_subcaptions_"))

total_panels   = 0
standard_count = 0
nonstandard    = Counter()

for d in subcap_dirs:
    files = list(d.glob("*.json"))
    for f in tqdm(files, desc=d.name, unit="file"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "error" in data:
            continue
        for p in data.get("panels", []):
            key = p.get("panel", "").strip()
            total_panels += 1
            if is_standard(key):
                standard_count += 1
            else:
                nonstandard[key] += 1

ns_count = sum(nonstandard.values())
pct_std  = 100 * standard_count / total_panels if total_panels else 0
pct_ns   = 100 * ns_count       / total_panels if total_panels else 0

lines = [
    f"Total panels          : {total_panels:,}",
    f"Standard (a-z / main) : {standard_count:,}  ({pct_std:.1f}%)",
    f"Non-standard (dropped): {ns_count:,}  ({pct_ns:.1f}%)",
    f"Unique non-std keys   : {len(nonstandard)}",
    "",
    "All non-standard keys (sorted by count):",
]
for k, v in nonstandard.most_common():
    lines.append(f"  {v:>6}  {repr(k)}")

output = "\n".join(lines)
print(output)

OUT_TXT.write_text(output + "\n", encoding="utf-8")
print(f"\nSaved → {OUT_TXT}")
