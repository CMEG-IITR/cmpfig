#!/usr/bin/env python3
"""
taxonomy_audit.py — audits visualization_category/visualization_subtype pairs
in dataset_index.json against the Appendix B taxonomy actually injected into
the annotation prompt (SUBTYPES_BY_CATEGORY in
caption_benchmarking/gen_subcaption_azure.py).

Classifies every panel into exactly one state:
    valid               — subtype is a child of the assigned category
    category_as_subtype — subtype string is itself a category name
    wrong_parent        — subtype is valid, but under a different category
    unknown_subtype     — subtype appears nowhere in the taxonomy
    null_subtype        — subtype field is missing/blank
    null_category       — category field is missing/blank (subtype present)

The canonical combined text report is paper_stats.py's paper_stats_results.txt
(section 2), which calls report() in this module directly — run that for the
one authoritative .txt. --report here is an optional standalone .txt, off by
default, for ad-hoc runs of this module alone.

Usage:
    python taxonomy_audit.py \
        --metadata dataset_index.json \
        --taxonomy ../caption_benchmarking/gen_subcaption_azure.py \
        --out taxonomy_audit.parquet \
        --report taxonomy_audit_report.txt   # optional, off by default
"""

import argparse
import ast
import contextlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA = SCRIPT_DIR / "dataset_index.json"
DEFAULT_TAXONOMY = SCRIPT_DIR.parent / "caption_benchmarking" / "gen_subcaption_azure.py"
DEFAULT_OUT = SCRIPT_DIR / "taxonomy_audit.parquet"
DEFAULT_REPORT = SCRIPT_DIR / "taxonomy_audit_report.txt"

PAPER_INVALID_COUNT = 5633
PAPER_INVALID_PCT = 1.44


# ── null handling ────────────────────────────────────────────────────────────
def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


# ── taxonomy loading ─────────────────────────────────────────────────────────
def load_taxonomy(path) -> dict:
    """Extract SUBTYPES_BY_CATEGORY out of the source file that builds the
    annotation prompt, without importing/executing it (avoids requiring its
    runtime deps, e.g. the openai package, and avoids any import side effects).
    """
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "SUBTYPES_BY_CATEGORY" in targets:
            raw = ast.literal_eval(node.value)
            return {category: set(subtypes) for category, subtypes in raw.items()}

    raise ValueError(f"SUBTYPES_BY_CATEGORY assignment not found in {path}")


def build_lookup(taxonomy: dict):
    valid_pairs = {
        (category, subtype)
        for category, subtypes in taxonomy.items()
        for subtype in subtypes
    }
    subtype_to_categories = {}
    for category, subtypes in taxonomy.items():
        for subtype in subtypes:
            subtype_to_categories.setdefault(subtype, set()).add(category)
    return valid_pairs, subtype_to_categories


# ── classification ───────────────────────────────────────────────────────────
def classify_subtype(category, subtype, taxonomy, valid_pairs, subtype_to_categories) -> str:
    if is_missing(subtype):
        return "null_subtype"
    if is_missing(category):
        return "null_category"
    if (category, subtype) in valid_pairs:
        return "valid"
    if subtype in taxonomy:
        return "category_as_subtype"
    if subtype in subtype_to_categories:
        return "wrong_parent"
    return "unknown_subtype"


# ── metadata loading ─────────────────────────────────────────────────────────
def load_metadata(path) -> pd.DataFrame:
    """Metadata-only load, following the plain json.load pattern used
    elsewhere in the repo (paper_stats.py, retrieval/data.py) — the index
    file contains no image bytes, only paths."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame.from_records(
        records,
        columns=["image_id", "panel_suffix", "visualization_category", "visualization_subtype"],
    )
    return df


# ── reporting ─────────────────────────────────────────────────────────────────
def audit(df: pd.DataFrame, taxonomy: dict) -> pd.DataFrame:
    valid_pairs, subtype_to_categories = build_lookup(taxonomy)

    df = df.copy()
    df["violation_class"] = [
        classify_subtype(c, s, taxonomy, valid_pairs, subtype_to_categories)
        for c, s in zip(df["visualization_category"], df["visualization_subtype"])
    ]
    df["subtype_valid"] = df["violation_class"] == "valid"
    return df


def report(df: pd.DataFrame, header: bool = True) -> None:
    total = len(df)
    counts = Counter(df["violation_class"])

    if header:
        print("=" * 60)
        print("TAXONOMY AUDIT")
        print("=" * 60)
    print(f"\nTotal panels: {total:,}\n")

    print("State breakdown:")
    for state in ["valid", "category_as_subtype", "wrong_parent", "unknown_subtype"]:
        n = counts.get(state, 0)
        pct = 100 * n / total if total else 0.0
        print(f"  {state:<22} {n:>8,}  ({pct:>6.2f}%)")

    null_subtype_n = counts.get("null_subtype", 0)
    null_category_n = counts.get("null_category", 0)
    print(f"\nNull handling (excluded from the four states above):")
    print(f"  null_subtype           {null_subtype_n:>8,}")
    print(f"  null_category          {null_category_n:>8,}")

    invalid_n = counts.get("category_as_subtype", 0) + counts.get("unknown_subtype", 0)
    invalid_pct = 100 * invalid_n / total if total else 0.0
    print("\n" + "-" * 60)
    print("Reconciliation vs paper (category_as_subtype + unknown_subtype)")
    print("-" * 60)
    print(f"  Computed : {invalid_n:,}  ({invalid_pct:.2f}%)")
    print(f"  Paper    : {PAPER_INVALID_COUNT:,}  ({PAPER_INVALID_PCT:.2f}%)")
    matches = invalid_n == PAPER_INVALID_COUNT
    if matches:
        print("  MATCH")
    else:
        diff = invalid_n - PAPER_INVALID_COUNT
        print(f"  MISMATCH — diff = {diff:+,} panels.")
        other_cross_category = int(
            (
                (df["visualization_subtype"] == "other")
                & (df["visualization_category"] != "other")
                & (df["violation_class"] == "category_as_subtype")
            ).sum()
        )
        if other_cross_category:
            print(
                f"  Root cause: {other_cross_category:,} panels have subtype \"other\" under a"
                f" category other than \"other\". The taxonomy defines \"other\": [\"other\"] —"
                f" a valid subtype only under category \"other\" — but the paper's flat"
                f" membership check treated \"other\" as universally valid regardless of"
                f" category, so it did not count these as invalid."
            )
        if other_cross_category != diff:
            print(f"  Residual unexplained diff: {diff - other_cross_category:+,} panels.")

    df_invalid = df[df["violation_class"].isin(["category_as_subtype", "wrong_parent"])]

    print("\n" + "-" * 60)
    print("Top 20 (category, subtype) pairs — category_as_subtype")
    print("-" * 60)
    cas = Counter(
        zip(
            df.loc[df["violation_class"] == "category_as_subtype", "visualization_category"],
            df.loc[df["violation_class"] == "category_as_subtype", "visualization_subtype"],
        )
    )
    for (category, subtype), n in cas.most_common(20):
        print(f"  {n:>7,}  {category!r:<30} -> {subtype!r}")

    cas_total = counts.get("category_as_subtype", 0)
    cas_other = int(
        (
            (df["visualization_subtype"] == "other")
            & (df["violation_class"] == "category_as_subtype")
        ).sum()
    )
    cas_named = cas_total - cas_other
    print("\n" + "-" * 60)
    print("category_as_subtype split")
    print("-" * 60)
    print(f"  Total category_as_subtype:                                   {cas_total:>6,}")
    if cas_total:
        print(
            f'  subtype == "other" (used under a category != "other"):     '
            f" {cas_other:>6,}  ({100*cas_other/cas_total:.2f}%)"
        )
        print(
            f"  subtype == another category name (e.g. \"Generic Plot\"):    "
            f" {cas_named:>6,}  ({100*cas_named/cas_total:.2f}%)"
        )

    print("\n" + "-" * 60)
    print("Top 20 (category, subtype) pairs — wrong_parent")
    print("-" * 60)
    wp = Counter(
        zip(
            df.loc[df["violation_class"] == "wrong_parent", "visualization_category"],
            df.loc[df["violation_class"] == "wrong_parent", "visualization_subtype"],
        )
    )
    for (category, subtype), n in wp.most_common(20):
        print(f"  {n:>7,}  {category!r:<30} -> {subtype!r}")

    print("\n" + "-" * 60)
    print("Per-category consistency rate (worst first)")
    print("-" * 60)
    attributable = df[df["violation_class"] != "null_category"]
    rows = []
    for category, group in attributable.groupby("visualization_category"):
        n = len(group)
        valid_n = int(group["subtype_valid"].sum())
        rate = valid_n / n if n else 0.0
        rows.append((category, rate, n, valid_n))
    rows.sort(key=lambda r: r[1])
    for category, rate, n, valid_n in rows:
        print(f"  {rate:>6.2%}  n={n:>7,}  valid={valid_n:>7,}  {category}")

    unique_invalid = df.loc[
        df["violation_class"].isin(["category_as_subtype", "unknown_subtype"]),
        "visualization_subtype",
    ].nunique()
    print("\n" + "-" * 60)
    print(f"Unique invalid subtype strings (category_as_subtype + unknown_subtype): {unique_invalid}")
    print("-" * 60)


class _Tee:
    """Mirrors writes to multiple streams — used to duplicate the printed
    report into a saved .txt file without altering print() globally."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to dataset_index.json")
    parser.add_argument(
        "--taxonomy",
        default=str(DEFAULT_TAXONOMY),
        help="Path to the source file containing SUBTYPES_BY_CATEGORY (Appendix B taxonomy)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write the audit parquet file")
    parser.add_argument(
        "--report",
        nargs="?",
        const=str(DEFAULT_REPORT),
        default="",
        help="Optional path to save a copy of the printed report as .txt (off by"
        " default — the canonical combined report is paper_stats.py's"
        " paper_stats_results.txt, which calls this module's report() directly;"
        " bare --report uses this module's own default filename)",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    df = load_metadata(args.metadata)
    df = audit(df, taxonomy)

    out_df = df[
        [
            "image_id",
            "panel_suffix",
            "visualization_category",
            "visualization_subtype",
            "violation_class",
            "subtype_valid",
        ]
    ]

    def emit():
        report(df)
        out_df.to_parquet(args.out, index=False)
        print(f"\nWrote {len(out_df):,} rows to {args.out}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f, contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            emit()
        print(f"Saved report to {args.report}")
    else:
        emit()


if __name__ == "__main__":
    main()
