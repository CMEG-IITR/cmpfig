"""
evaluation/compare_results.py
------------------------------
Loads metric JSON files from all conditions (zero_shot, few_shot, finetuned)
and prints a clean LaTeX-ready comparison table plus a per-subtype breakdown.

Usage:
  python -m evaluation.compare_results \
      --results_dirs \
          ./results/zero_shot \
          ./results/few_shot \
          ./results/finetuned \
      --output_dir ./results/final
"""

import argparse
import json
from pathlib import Path


METRIC_KEYS = ["bleu1", "bleu4", "meteor", "rougeL", "bertscore_f1"]
METRIC_LABELS = {
    "bleu1":        "BLEU-1",
    "bleu4":        "BLEU-4",
    "meteor":       "METEOR",
    "rougeL":       "ROUGE-L",
    "bertscore_f1": "BERTScore-F1",
}
CONDITION_LABELS = {
    "zero_shot":  "Zero-shot",
    "few_shot":   "3-shot",
    "finetuned":  "Fine-tuned (ours)",
}


def load_metrics(results_dir: Path) -> tuple[str, dict]:
    """Find the *_metrics.json in this dir and return (condition_name, metrics)."""
    for f in results_dir.glob("*_metrics.json"):
        condition = f.stem.replace("_metrics", "")
        data = json.loads(f.read_text())
        return condition, data
    raise FileNotFoundError(f"No *_metrics.json found in {results_dir}")


def print_overall_table(all_results: list[tuple[str, dict]]):
    print("\n" + "=" * 70)
    print("  Overall comparison table")
    print("=" * 70)

    header = f"{'Condition':<22}" + "".join(f"{METRIC_LABELS[k]:>14}" for k in METRIC_KEYS)
    print(header)
    print("-" * 70)

    for cond, data in all_results:
        overall = data.get("overall", {})
        label   = CONDITION_LABELS.get(cond, cond)
        row = f"{label:<22}"
        for k in METRIC_KEYS:
            v = overall.get(k, None)
            row += f"{v:>14.2f}" if v is not None else f"{'—':>14}"
        print(row)

    print("=" * 70)


def print_subtype_table(all_results: list[tuple[str, dict]]):
    # collect all subtypes
    subtypes = set()
    for _, data in all_results:
        subtypes.update(data.get("per_subtype", {}).keys())
    subtypes = sorted(subtypes)

    print(f"\n\nPer-subtype BLEU-4  |  BERTScore-F1")
    col_w = 16
    cond_labels = [CONDITION_LABELS.get(c, c) for c, _ in all_results]

    # header
    header = f"{'Subtype':<30}" + "".join(
        f"{f'BLEU4/{l[:6]}':>{col_w}}  {f'BScr/{l[:6]}':>{col_w}}"
        for l in cond_labels
    )
    print(header)
    print("-" * (30 + len(all_results) * (col_w * 2 + 4)))

    for st in subtypes:
        row = f"{st:<30}"
        for cond, data in all_results:
            m = data.get("per_subtype", {}).get(st, {})
            b4 = m.get("bleu4",        None)
            bs = m.get("bertscore_f1", None)
            row += f"{b4 if b4 else '—':>{col_w}}  {bs if bs else '—':>{col_w}}"
        print(row)


def latex_table(all_results: list[tuple[str, dict]]) -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    cols = "l" + "r" * len(METRIC_KEYS)
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")

    header = "Model & " + " & ".join(METRIC_LABELS[k] for k in METRIC_KEYS) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for cond, data in all_results:
        overall = data.get("overall", {})
        label   = CONDITION_LABELS.get(cond, cond)
        values  = [
            f"{overall[k]:.2f}" if overall.get(k) is not None else "—"
            for k in METRIC_KEYS
        ]
        lines.append(f"{label} & " + " & ".join(values) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Panel subcaption generation results on the test split.}")
    lines.append(r"\label{tab:exp3_captioning}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir",   required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = []
    for d in args.results_dirs:
        try:
            cond, metrics = load_metrics(Path(d))
            all_results.append((cond, metrics))
        except FileNotFoundError as e:
            print(f"[WARN] {e}")

    if not all_results:
        print("No metric files found.")
        return

    print_overall_table(all_results)
    print_subtype_table(all_results)

    latex = latex_table(all_results)
    latex_path = out / "table_captioning.tex"
    latex_path.write_text(latex)
    print(f"\nLaTeX table saved → {latex_path}")

    # also save combined JSON
    combined = {cond: metrics for cond, metrics in all_results}
    (out / "combined_metrics.json").write_text(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
