#!/usr/bin/env python3
"""
paper_stats.py — produces all dataset statistics needed for the paper.

Covers:
  1. Detection→annotation join evaluation  (from build_dataset logs)
  2. Taxonomy validity (category/subtype pairing) (from dataset_index.json)
  3. Inter-annotator agreement             (from cohen_kappa_log.txt)

Run from the cmpfig root:
    python paper_stats.py
"""

import contextlib
import json
import os
import re
import sys
import builtins
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
    "steel":    ROOT / "main_data" / "steel_prod_dataset.log",
    "additive": ROOT / "main_data" / "additive_elsevier_dataset.log",
}
DATASET_INDEX = ROOT / "Visulization" / "dataset_index.json"
KAPPA_LOG     = ROOT / "Visulization" / "kappa" / "cohen_kappa_log.txt"
TAXONOMY_SOURCE = ROOT / "caption_benchmarking" / "gen_subcaption_azure.py"


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


# ── 2. Taxonomy validity (category/subtype pairing) ───────────────────────────
sep("2. TAXONOMY VALIDITY (category/subtype pairing)")

if not DATASET_INDEX.exists():
    print(f"  [skip] {DATASET_INDEX} not found")
elif not TAXONOMY_SOURCE.exists():
    print(f"  [skip] {TAXONOMY_SOURCE} not found")
else:
    sys.path.insert(0, str(ROOT / "Visulization"))
    from taxonomy_audit import load_taxonomy, load_metadata, audit, report as taxonomy_report, _Tee

    taxonomy = load_taxonomy(TAXONOMY_SOURCE)
    df = load_metadata(DATASET_INDEX)
    df = audit(df, taxonomy)

    # taxonomy_audit.report() prints via plain print(); route it through the
    # same stdout+file mirroring this script's own print() override uses, so
    # both land in this one results file instead of a second, separate report.
    with contextlib.redirect_stdout(_Tee(sys.stdout, _results_fp)):
        taxonomy_report(df, header=False)

    print(f"\n  visualization_category is schema-enforced (JSON enum) → always valid.")
    print(f"  visualization_subtype  is prompt-guided only           → see breakdown above.")
    print(
        "  Correction: the routing claim in earlier drafts (\"out-of-taxonomy panels"
        "\n  are routed to training only, via the rare-subtype threshold\") does not"
        "\n  hold. retrieval/data.py's compute_split() force-routes a figure to train"
        "\n  only when its DOMINANT subtype string has < rare_subtype_threshold (10)"
        "\n  panels summed across the whole corpus — a check on string rarity, not"
        "\n  taxonomy validity. Cross-referencing confirms only ~68 invalid panels"
        "\n  (<1% of either the old or new invalid set) are actually force-routed;"
        "\n  the rest pass through the normal stratified split and can land in val/test."
    )


# ── 3. Inter-annotator agreement ──────────────────────────────────────────────
sep("3. INTER-ANNOTATOR AGREEMENT (Cohen's κ)")

if not KAPPA_LOG.exists():
    print(f"  [skip] {KAPPA_LOG} not found")
else:
    text = KAPPA_LOG.read_text(encoding="utf-8")
    def grab(pattern):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else "N/A"

    common_images = grab(r'Common images\s*:\s*(\d+)')
    agreed        = grab(r'Agreed\s*:\s*(\d+)')
    disagreed     = grab(r'Disagreed\s*:\s*(\d+)')
    po            = grab(r'Observed agreement Po\s*:\s*([\d.]+)')
    kappa         = grab(r"Cohen's Kappa\s*:\s*([\d.]+)")
    interpretation = grab(r'Interpretation\s*:\s*(.+)')

    print(f"""
  Images annotated by both  : {common_images}
  Agreed (exact label set)  : {agreed}
  Disagreed                 : {disagreed}
  Observed agreement (Po)   : {po}
  Cohen's κ                 : {kappa}
  Interpretation            : {interpretation}

  Unit: per-image exact match of the complete panel label set.
""")
sep("DONE")

_results_fp.close()
