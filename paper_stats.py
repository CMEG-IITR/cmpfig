#!/usr/bin/env python3
"""
paper_stats.py — produces all dataset statistics needed for the paper.

Covers:
  1. Detection→annotation join evaluation  (from build_dataset logs)
  2. Out-of-taxonomy subtype rate          (from dataset_index.json)
  3. Inter-annotator agreement             (from cohen_kappa_log.txt)

Run from the cmpfig root:
    python paper_stats.py
"""

import json
import os
import re
import builtins
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_FILE = ROOT / "paper_stats_results.txt"


_results_fp = open(RESULTS_FILE, "w", encoding="utf-8")


def print(*args, **kwargs):
    builtins.print(*args, **kwargs)
    builtins.print(*args, **kwargs, file=_results_fp)

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_FILES = {
    "alloy":    ROOT / "main_data" / "alloy_build_dataset.log",
    "ceramics": ROOT / "main_data" / "ceramics_build_dataset.log",
    "composite":ROOT / "main_data" / "composite_build_dataset.log",
    "ni_alloy": ROOT / "main_data" / "ni_alloy_build_dataset.log",
}
DATASET_INDEX = ROOT / "Visulization" / "dataset_index.json"
KAPPA_LOG     = ROOT / "Visulization" / "kappa" / "cohen_kappa_log.txt"

# ── Valid taxonomy ─────────────────────────────────────────────────────────────
VALID_SUBTYPES = {
    "SEM","TEM","STEM","HAADF-STEM","BF-TEM","DF-TEM","Optical Micrograph",
    "Confocal Microscopy","AFM","Fluorescence Microscopy","Live/Dead Staining",
    "XRD Pattern","SAED","EBSD Map","Pole Figure","Inverse Pole Figure",
    "Neutron Diffraction","Synchrotron Diffraction",
    "XPS Spectrum","Raman Spectrum","FTIR Spectrum","EDX Spectrum","EELS Spectrum",
    "NMR Spectrum","Mass Spectrum","UV-Vis Spectrum","Photoluminescence Spectrum",
    "XANES Spectrum","EXAFS Spectrum","Mössbauer Spectrum",
    "DSC Curve","TGA Curve","DMA Curve","TMA Curve",
    "Binary Phase Diagram","Ternary Phase Diagram","TTT Diagram","CCT Diagram",
    "CALPHAD Diagram","Pourbaix Diagram",
    "Stress-Strain Curve","Load-Displacement Curve","Nanoindentation Curve",
    "Hardness Map","Fatigue/S-N Curve","Creep Curve","Fracture Toughness Plot",
    "DIC Strain Map","Wear/Tribology Plot",
    "Cyclic Voltammogram","Charge-Discharge Curve","Capacity Retention Plot",
    "Coulombic Efficiency Plot","Nyquist Plot","Bode Plot","Tafel Plot",
    "Polarization Curve","Polarisation Curve","GITT/PITT Curve","Rate Capability Plot",
    "M-H Hysteresis Loop","M-T Curve","ZFC/FC Curve","I-V Curve","C-V Curve",
    "Band Structure","Density of States","Hall Effect Plot",
    "Absorbance Spectrum","Transmittance Spectrum","Reflectance Spectrum",
    "EQE/IQE Plot","J-V Curve","Ellipsometry Plot","Refractive Index Plot",
    "APT Reconstruction","Micro-CT","FIB-SEM Tomography","3D Reconstruction",
    "EDS Map","WDS Map","EBSD IPF Map","Elemental Distribution Map",
    "DFT Result","MD Snapshot","MD Trajectory","Phase-Field Simulation",
    "FEA/FEM Result","Monte Carlo Result",
    "Parity Plot","Confusion Matrix","ROC Curve","Learning Curve",
    "Feature Importance Plot","SHAP Plot","t-SNE/UMAP/PCA Plot",
    "Unit Cell","Atomic Model","Supercell",
    "Bar Chart","Scatter Plot","Line Graph","Box Plot","Contour Plot","Heatmap",
    "Radar Chart","Ashby Plot","Arrhenius Plot","Histogram",
    "Process Schematic","Flowchart","Mechanism Diagram","Experimental Setup",
    "Sample Photo","Equipment Photo","In-situ Photo",
    "Data Table","other",
}


def sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(title)
        print("=" * 60)


# ── 1. Join evaluation from logs ───────────────────────────────────────────────
def parse_log(path):
    text = Path(path).read_text(encoding="utf-8")
    def extract(pattern):
        m = re.search(pattern, text)
        return int(m.group(1).replace(",", "")) if m else 0
    return {
        "scanned":        extract(r"Total image files scanned\s*:\s*([\d,]+)"),
        "skipped":        extract(r"Skipped \(bad filename\)\s*:\s*([\d,]+)"),
        "matched":        extract(r"Matched \(image \+ panel\)\s*:\s*([\d,]+)"),
        "no_json":        extract(r"No JSON file\s*:\s*([\d,]+)"),
        "panel_missing":  extract(r"JSON found, panel missing\s*:\s*([\d,]+)"),
    }


sep("1. DETECTION → ANNOTATION JOIN EVALUATION")

totals = {"scanned":0, "skipped":0, "matched":0, "no_json":0, "panel_missing":0}
print(f"\n{'Domain':<12} {'Scanned':>9} {'Matched':>9} {'Match%':>7} {'No JSON':>8} {'PanelMiss':>10}")
print("-" * 60)
for domain, path in LOG_FILES.items():
    if not path.exists():
        print(f"  [skip] {path} not found")
        continue
    s = parse_log(path)
    proc = s["scanned"] - s["skipped"]
    pct  = 100 * s["matched"] / proc if proc else 0
    print(f"{domain:<12} {s['scanned']:>9,} {s['matched']:>9,} {pct:>6.1f}% "
          f"{s['no_json']:>8,} {s['panel_missing']:>10,}")
    for k in totals:
        totals[k] += s[k]

proc = totals["scanned"] - totals["skipped"]
matched       = totals["matched"]
no_json       = totals["no_json"]
panel_missing = totals["panel_missing"]
unmatched     = proc - matched

print("-" * 60)
print(f"{'TOTAL':<12} {totals['scanned']:>9,} {matched:>9,} "
      f"{100*matched/proc:>6.1f}% {no_json:>8,} {panel_missing:>10,}")

print(f"""
Summary:
  Detected panels processed : {proc:,}
  Successfully joined       : {matched:,}  ({100*matched/proc:.1f}%)
  Unmatched — no LLM JSON   : {no_json:,}  ({100*no_json/proc:.1f}%)
  Unmatched — panel key miss: {panel_missing:,}  ({100*panel_missing/proc:.1f}%)
  Total unmatched (excluded): {unmatched:,}  ({100*unmatched/proc:.1f}%)
  Common-label panels skipped (bad filename): {totals['skipped']}

Join mechanism:
  Crop filename letter  (e.g. imgXXXX_B.jpg → key "b")
  matched case-insensitively to LLM JSON panel key.
  Single-panel figures use key "main".
  Fallback: none — unmatched panels are excluded from the dataset.
""")


# ── 2. Out-of-taxonomy subtype rate ───────────────────────────────────────────
sep("2. OUT-OF-TAXONOMY SUBTYPE RATE")

if not DATASET_INDEX.exists():
    print(f"  [skip] {DATASET_INDEX} not found")
else:
    with open(DATASET_INDEX, encoding="utf-8") as f:
        data = json.load(f)

    total   = len(data)
    invalid = [d for d in data if d.get("visualization_subtype", "") not in VALID_SUBTYPES]
    counts  = Counter(d.get("visualization_subtype", "") for d in invalid)

    print(f"\n  Total panels in index : {total:,}")
    print(f"  Out-of-taxonomy       : {len(invalid):,}  ({100*len(invalid)/total:.2f}%)")
    print(f"  Unique invalid values : {len(counts)}")
    print(f"\n  Top out-of-taxonomy subtypes:")
    for s, c in counts.most_common(15):
        print(f"    {c:>5}  {s}")

    print(f"""
  Note:
    visualization_category is schema-enforced (JSON enum) → always valid.
    visualization_subtype  is prompt-guided only           → {100*len(invalid)/total:.2f}% out-of-taxonomy.
    These panels are retained but routed to training only
    via the rare-subtype threshold (< 10 panels per subtype).
""")


# ── 3. Inter-annotator agreement ──────────────────────────────────────────────
sep("3. INTER-ANNOTATOR AGREEMENT (Cohen's κ)")

if not KAPPA_LOG.exists():
    print(f"  [skip] {KAPPA_LOG} not found")
else:
    text = KAPPA_LOG.read_text(encoding="utf-8")
    def grab(pattern):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else "N/A"

    print(f"""
  Images annotated by both  : {grab(r'Common images\s*:\s*(\d+)')}
  Agreed (exact label set)  : {grab(r'Agreed\s*:\s*(\d+)')}
  Disagreed                 : {grab(r'Disagreed\s*:\s*(\d+)')}
  Observed agreement (Po)   : {grab(r'Observed agreement Po\s*:\s*([\d.]+)')}
  Cohen's κ                 : {grab(r"Cohen's Kappa\s*:\s*([\d.]+)")}
  Interpretation            : {grab(r'Interpretation\s*:\s*(.+)')}

  Unit: per-image exact match of the complete panel label set.
""")
sep("DONE")

_results_fp.close()
