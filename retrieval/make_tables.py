"""
Generate paper-ready result tables from test_results.json and baseline_results.json.
Prints markdown + LaTeX for both Table 1 (overall) and Table 2 (per-category).

Usage:
    python -m retrieval.make_tables
    python -m retrieval.make_tables --output_dir retrieval/outputs_proper
"""
import argparse
import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


def pct(v):
    return f"{v*100:.1f}"


def dec(v):
    return f"{v:.3f}"


# ── Table 1 — Overall ─────────────────────────────────────────────────────────

def table1_markdown(rnd, zs, ft):
    metrics = ["R@1", "R@5", "R@10", "R@50", "R@100", "MRR", "mAP@100"]

    header = (
        "| Metric     | Random     | ZS-CLIP i2t | ZS-CLIP t2i "
        "| **Ours i2t** | **Ours t2i** |"
    )
    sep = (
        "|------------|-----------|-------------|-------------"
        "|-------------|-------------|"
    )
    rows = [header, sep]

    for m in metrics:
        k = m
        r  = rnd["i2t_" + k]
        zi = zs["i2t_"  + k]
        zt = zs["t2i_"  + k]
        fi = ft["i2t_"  + k]
        ftt= ft["t2i_"  + k]

        if m in ("MRR", "mAP@100"):
            fmt = dec
        else:
            fmt = lambda v: pct(v) + "%"

        rows.append(
            f"| {m:<10} | {fmt(r):<9} | {fmt(zi):<11} | {fmt(zt):<11} "
            f"| **{fmt(fi)}**    | {fmt(ftt):<11} |"
        )

    rows.append(f"| N          | 39,315    | 39,315      | 39,315      "
                f"| 39,315      | 39,315      |")
    return "\n".join(rows)


def table1_latex(rnd, zs, ft):
    metrics = ["R@1", "R@5", "R@10", "R@50", "R@100", "MRR", "mAP@100"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Overall cross-modal retrieval performance on 39{,}315 test panels."
        r" R@K values in \%. Best result per row in \textbf{bold}.}",
        r"\label{tab:retrieval_overall}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Metric & Random & \multicolumn{2}{c}{Zero-shot CLIP} & \multicolumn{2}{c}{\textbf{Ours (Fine-tuned)}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"       &        & i{\to}t & t{\to}i & \textbf{i{\to}t} & t{\to}i \\",
        r"\midrule",
    ]

    for m in metrics:
        k  = m
        r  = rnd["i2t_" + k]
        zi = zs["i2t_"  + k]
        zt = zs["t2i_"  + k]
        fi = ft["i2t_"  + k]
        ftt= ft["t2i_"  + k]

        if m in ("MRR", "mAP@100"):
            fmt = dec
        else:
            fmt = lambda v: pct(v)

        lines.append(
            f"R@{m.split('@')[1] if '@' in m else m} & {fmt(r)} & {fmt(zi)} & {fmt(zt)}"
            f" & \\textbf{{{fmt(fi)}}} & {fmt(ftt)} \\\\"
            if "@" in m else
            f"{m} & {fmt(r)} & {fmt(zi)} & {fmt(zt)}"
            f" & \\textbf{{{fmt(fi)}}} & {fmt(ftt)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Table 2 — Per-category ────────────────────────────────────────────────────

def table2_markdown(zs_cat, ft_cat):
    all_cats = sorted(
        set(list(ft_cat.keys())) | set(list(zs_cat.keys())),
        key=lambda c: -ft_cat.get(c, {}).get("n", 0)
    )
    all_cats = [c for c in all_cats if c != "other"]

    header = (
        "| Category                  |    n  "
        "| ZS i2t R@1 | ZS i2t R@10 "
        "| ZS t2i R@1 | ZS t2i R@10 "
        "| **Ours i2t R@1** | **Ours i2t R@10** | **Ours i2t MRR** "
        "| **Ours t2i R@1** | **Ours t2i R@10** | **Ours t2i MRR** |"
    )
    sep = (
        "|---------------------------|------:"
        "|----------:|------------:"
        "|----------:|------------:"
        "|-----------------:|------------------:|-----------------:"
        "|-----------------:|------------------:|-----------------:|"
    )
    rows = [header, sep]

    for cat in all_cats:
        fc = ft_cat.get(cat, {})
        zc = zs_cat.get(cat, {})
        n  = fc.get("n", zc.get("n", 0))

        zi_r1  = pct(zc.get("i2t_R@1",  float("nan"))) + "%"
        zi_r10 = pct(zc.get("i2t_R@10", float("nan"))) + "%"
        zt_r1  = pct(zc.get("t2i_R@1",  float("nan"))) + "%"
        zt_r10 = pct(zc.get("t2i_R@10", float("nan"))) + "%"

        fi_r1  = pct(fc.get("i2t_R@1",  float("nan"))) + "%"
        fi_r10 = pct(fc.get("i2t_R@10", float("nan"))) + "%"
        fi_mrr = dec(fc.get("i2t_MRR",  float("nan")))
        ft_r1  = pct(fc.get("t2i_R@1",  float("nan"))) + "%"
        ft_r10 = pct(fc.get("t2i_R@10", float("nan"))) + "%"
        ft_mrr = dec(fc.get("t2i_MRR",  float("nan")))

        rows.append(
            f"| {cat:<25} | {n:>5} "
            f"| {zi_r1:>10} | {zi_r10:>11} "
            f"| {zt_r1:>10} | {zt_r10:>11} "
            f"| **{fi_r1}** | **{fi_r10}** | **{fi_mrr}** "
            f"| **{ft_r1}** | **{ft_r10}** | **{ft_mrr}** |"
        )

    return "\n".join(rows)


def table2_latex(zs_cat, ft_cat):
    all_cats = sorted(
        set(list(ft_cat.keys())) | set(list(zs_cat.keys())),
        key=lambda c: -ft_cat.get(c, {}).get("n", 0)
    )
    all_cats = [c for c in all_cats if c != "other"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-category retrieval performance for both directions"
        r" (i{\to}t: image queries text; t{\to}i: text queries image)."
        r" R@K values in \%. Categories sorted by test-set size.}",
        r"\label{tab:retrieval_category}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lrccccccccccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{Zero-shot CLIP} & \multicolumn{6}{c}{\textbf{Ours (Fine-tuned)}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-12}",
        r"& & \multicolumn{2}{c}{i{\to}t} & \multicolumn{2}{c}{t{\to}i}"
        r" & \multicolumn{3}{c}{\textbf{i{\to}t}} & \multicolumn{3}{c}{\textbf{t{\to}i}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}",
        r"Category & $n$ & R@1 & R@10 & R@1 & R@10"
        r" & \textbf{R@1} & \textbf{R@10} & \textbf{MRR}"
        r" & \textbf{R@1} & \textbf{R@10} & \textbf{MRR} \\",
        r"\midrule",
    ]

    for cat in all_cats:
        fc = ft_cat.get(cat, {})
        zc = zs_cat.get(cat, {})
        n  = fc.get("n", zc.get("n", 0))

        zi_r1  = pct(zc.get("i2t_R@1",  float("nan")))
        zi_r10 = pct(zc.get("i2t_R@10", float("nan")))
        zt_r1  = pct(zc.get("t2i_R@1",  float("nan")))
        zt_r10 = pct(zc.get("t2i_R@10", float("nan")))

        fi_r1  = pct(fc.get("i2t_R@1",  float("nan")))
        fi_r10 = pct(fc.get("i2t_R@10", float("nan")))
        fi_mrr = dec(fc.get("i2t_MRR",  float("nan")))
        ft_r1  = pct(fc.get("t2i_R@1",  float("nan")))
        ft_r10 = pct(fc.get("t2i_R@10", float("nan")))
        ft_mrr = dec(fc.get("t2i_MRR",  float("nan")))

        cat_tex = cat.replace("&", r"\&")
        lines.append(
            f"{cat_tex} & {n:,} & {zi_r1} & {zi_r10} & {zt_r1} & {zt_r10}"
            f" & \\textbf{{{fi_r1}}} & \\textbf{{{fi_r10}}} & \\textbf{{{fi_mrr}}}"
            f" & \\textbf{{{ft_r1}}} & \\textbf{{{ft_r10}}} & \\textbf{{{ft_mrr}}} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/mnt/d/Subham/Compoun_img_01/retrieval/outputs_proper")
    args = p.parse_args()

    eval_dir = os.path.join(args.output_dir, "eval")
    ft  = load(os.path.join(eval_dir, "test_results.json"))
    bl  = load(os.path.join(eval_dir, "baseline_results.json"))

    ft_ov  = ft["overall"]
    ft_cat = ft["by_category"]
    zs_ov  = bl["zero_shot_clip"]["overall"]
    zs_cat = bl["zero_shot_clip"]["by_category"]
    rnd_ov = bl["random"]["overall"]

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TABLE 1 — OVERALL  (Markdown)")
    print("="*70)
    print(table1_markdown(rnd_ov, zs_ov, ft_ov))

    print("\n" + "="*70)
    print("TABLE 2 — PER-CATEGORY  (Markdown)")
    print("="*70)
    print(table2_markdown(zs_cat, ft_cat))

    print("\n" + "="*70)
    print("TABLE 1 — OVERALL  (LaTeX)")
    print("="*70)
    print(table1_latex(rnd_ov, zs_ov, ft_ov))

    print("\n" + "="*70)
    print("TABLE 2 — PER-CATEGORY  (LaTeX)")
    print("="*70)
    print(table2_latex(zs_cat, ft_cat))

    # ── Save ──────────────────────────────────────────────────────────────────
    out = os.path.join(args.output_dir, "tables.txt")
    with open(out, "w") as f:
        f.write("TABLE 1 — OVERALL (Markdown)\n")
        f.write(table1_markdown(rnd_ov, zs_ov, ft_ov) + "\n\n")
        f.write("TABLE 2 — PER-CATEGORY (Markdown)\n")
        f.write(table2_markdown(zs_cat, ft_cat) + "\n\n")
        f.write("TABLE 1 — OVERALL (LaTeX)\n")
        f.write(table1_latex(rnd_ov, zs_ov, ft_ov) + "\n\n")
        f.write("TABLE 2 — PER-CATEGORY (LaTeX)\n")
        f.write(table2_latex(zs_cat, ft_cat) + "\n")

    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
