"""
evaluation/metrics.py
---------------------
Computes BLEU-1/4, METEOR, ROUGE-L, and BERTScore (F1)
on a list of (reference, prediction) pairs.
Also computes per-subtype breakdowns.

Usage:
  python -m evaluation.metrics \
      --predictions_json ./results/zero_shot/zero_shot_predictions.json \
      --output_dir       ./results/zero_shot
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4",  quiet=True)
nltk.download("punkt",    quiet=True)
nltk.download("punkt_tab", quiet=True)


# ── word count stats ──────────────────────────────────────────────────────────

def word_count_stats(texts: list[str]) -> dict:
    counts = [len(t.split()) for t in texts]
    return {
        "mean":   round(float(np.mean(counts)),  2),
        "std":    round(float(np.std(counts)),   2),
        "min":    int(np.min(counts)),
        "max":    int(np.max(counts)),
        "in_range_20_30_pct": round(
            sum(1 for c in counts if 20 <= c <= 30) / len(counts) * 100, 1
        ),
    }


# ── BLEU ──────────────────────────────────────────────────────────────────────

def compute_bleu(references: list[str], predictions: list[str]) -> dict:
    smooth = SmoothingFunction().method1
    refs_tok  = [[r.lower().split()] for r in references]
    preds_tok = [p.lower().split()   for p in predictions]

    bleu1 = corpus_bleu(refs_tok, preds_tok, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        "bleu1": round(bleu1 * 100, 2),
        "bleu4": round(bleu4 * 100, 2),
    }


# ── METEOR ────────────────────────────────────────────────────────────────────

def compute_meteor(references: list[str], predictions: list[str]) -> dict:
    scores = [
        meteor_score([r.lower().split()], p.lower().split())
        for r, p in zip(references, predictions)
    ]
    return {"meteor": round(float(np.mean(scores)) * 100, 2)}


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def compute_rouge(references: list[str], predictions: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(r, p)["rougeL"].fmeasure
        for r, p in zip(references, predictions)
    ]
    return {"rougeL": round(float(np.mean(scores)) * 100, 2)}


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bertscore(
    references: list[str], predictions: list[str],
    model_type: str = "allenai/scibert_scivocab_uncased",  # domain-relevant
) -> dict:
    """
    Uses SciBERT rather than BERT-base for better domain alignment.
    Falls back to bert-base-uncased if SciBERT is unavailable.
    """
    try:
        P, R, F1 = bert_score(
            predictions, references,
            model_type=model_type,
            lang="en", verbose=False,
        )
    except Exception:
        P, R, F1 = bert_score(
            predictions, references,
            model_type="bert-base-uncased",
            lang="en", verbose=False,
        )
    return {
        "bertscore_p":  round(float(P.mean())  * 100, 2),
        "bertscore_r":  round(float(R.mean())  * 100, 2),
        "bertscore_f1": round(float(F1.mean()) * 100, 2),
    }


# ── aggregate ─────────────────────────────────────────────────────────────────

def compute_all_metrics(
    references: list[str],
    predictions: list[str],
    compute_bert: bool = True,
) -> dict:
    metrics = {}
    metrics.update(compute_bleu(references, predictions))
    metrics.update(compute_meteor(references, predictions))
    metrics.update(compute_rouge(references, predictions))
    if compute_bert:
        metrics.update(compute_bertscore(references, predictions))
    metrics["word_count_pred"] = word_count_stats(predictions)
    metrics["word_count_ref"]  = word_count_stats(references)
    metrics["n_samples"]       = len(references)
    return metrics


# ── per-subtype breakdown ─────────────────────────────────────────────────────

def per_subtype_metrics(results: list[dict], compute_bert: bool = True) -> dict:
    grouped = defaultdict(lambda: {"refs": [], "preds": []})
    for r in results:
        st = r["visualization_subtype"]
        grouped[st]["refs"].append(r["reference"])
        grouped[st]["preds"].append(r["prediction"])

    out = {}
    for subtype, data in sorted(grouped.items()):
        out[subtype] = compute_all_metrics(
            data["refs"], data["preds"], compute_bert=compute_bert
        )
    return out


# ── pretty print ──────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, title: str = "Overall"):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    skip = {"word_count_pred", "word_count_ref", "n_samples"}
    for k, v in metrics.items():
        if k not in skip:
            print(f"  {k:<20}  {v:>7.2f}")
    wc = metrics.get("word_count_pred", {})
    print(f"\n  Pred word count — mean: {wc.get('mean')}  std: {wc.get('std')}  "
          f"in 20-30 range: {wc.get('in_range_20_30_pct')}%")
    print(f"  N samples: {metrics.get('n_samples')}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_json", required=True,
                        help="JSON file from zero_shot.py or run_finetuned.py")
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--no_bertscore",     action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = json.loads(Path(args.predictions_json).read_text())
    refs  = [r["reference"]  for r in results]
    preds = [r["prediction"] for r in results]

    compute_bert = not args.no_bertscore

    # overall
    overall = compute_all_metrics(refs, preds, compute_bert=compute_bert)
    print_metrics(overall, "Overall")

    # per subtype
    per_sub = per_subtype_metrics(results, compute_bert=compute_bert)
    print("\nPer-subtype summary (BLEU-4 | BERTScore-F1):")
    for st, m in per_sub.items():
        bert_str = f"  BERTScore-F1: {m.get('bertscore_f1', '-')}" if compute_bert else ""
        print(f"  {st:<30}  BLEU-4: {m['bleu4']:>6.2f}{bert_str}")

    # save
    report = {"overall": overall, "per_subtype": per_sub}
    mode_tag = Path(args.predictions_json).stem.replace("_predictions", "")
    (out / f"{mode_tag}_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nMetrics saved to {out}")


if __name__ == "__main__":
    main()
