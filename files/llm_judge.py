"""
evaluation/llm_judge.py
-----------------------
LLM-as-judge evaluation for subcaption quality.
Uses Claude claude-sonnet-4-20250514 to score each (image, reference, prediction) triple
on four criteria specific to materials science figure captions.

Scoring rubric (each 1-5):
  1. domain_accuracy   — correct materials science terminology, no hallucination
  2. visual_grounding  — describes what is actually visible in the image
  3. completeness      — covers the key observation the reference captures
  4. word_count_adherence — stays within 20-30 words (penalises padding/truncation)

Usage:
  python -m evaluation.llm_judge \
      --predictions_json ./results/zero_shot/zero_shot_predictions.json \
      --dataset_dir      /path/to/matfig_captioning \
      --output_dir       ./results/zero_shot \
      --n_sample         200        # evaluate a random sample (cost control)
      --split            test
"""

import argparse
import base64
import json
import random
import time
from io import BytesIO
from pathlib import Path

import anthropic
from datasets import load_from_disk
from tqdm import tqdm


JUDGE_SYSTEM = """You are an expert materials scientist and scientific writing evaluator.
You will assess AI-generated subcaptions for materials science figure panels.
Return ONLY a valid JSON object — no prose, no markdown fences."""

JUDGE_USER_TEMPLATE = """Below is a panel image from a materials science publication, 
the human-written reference subcaption, and an AI-generated prediction.

REFERENCE:  {reference}

PREDICTION: {prediction}

Score the prediction on each criterion (integer 1–5):
  domain_accuracy   — correct materials science terminology; no hallucination
  visual_grounding  — accurately describes what is visible in the image
  completeness      — captures the key observation in the reference
  word_count_adherence — 5 if 20-30 words, penalise proportionally outside range

Return exactly this JSON:
{{
  "domain_accuracy": <int>,
  "visual_grounding": <int>,
  "completeness": <int>,
  "word_count_adherence": <int>,
  "brief_reasoning": "<one sentence>"
}}"""


def image_to_base64(pil_image) -> str:
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def judge_one(
    client: anthropic.Anthropic,
    image,
    reference: str,
    prediction: str,
    retries: int = 3,
) -> dict | None:
    img_b64 = image_to_base64(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": JUDGE_USER_TEMPLATE.format(
                        reference=reference, prediction=prediction
                    ),
                },
            ],
        }
    ]

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                system=JUDGE_SYSTEM,
                messages=messages,
            )
            raw = response.content[0].text.strip()
            # strip accidental markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == retries - 1:
                return None
        except anthropic.RateLimitError:
            time.sleep(30)
        except Exception as e:
            print(f"[WARN] Judge API error: {e}")
            if attempt == retries - 1:
                return None
            time.sleep(5)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_json", required=True)
    parser.add_argument("--dataset_dir",      required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--split",            default="test")
    parser.add_argument("--n_sample",         type=int, default=200,
                        help="Number of examples to judge (0 = all)")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # load predictions
    preds = json.loads(Path(args.predictions_json).read_text())

    # load images (keyed by figure_id + panel_id)
    dd = load_from_disk(args.dataset_dir)
    img_lookup = {
        (row["figure_id"], row["panel_id"]): row["image"]
        for row in dd[args.split]
    }

    # sample
    if args.n_sample > 0 and args.n_sample < len(preds):
        random.seed(args.seed)
        preds = random.sample(preds, args.n_sample)

    client = anthropic.Anthropic()

    scored = []
    for item in tqdm(preds, desc="LLM judge"):
        key = (item["figure_id"], item["panel_id"])
        image = img_lookup.get(key)
        if image is None:
            continue

        scores = judge_one(
            client, image,
            reference=item["reference"],
            prediction=item["prediction"],
        )
        if scores is None:
            continue

        scored.append({
            **item,
            "judge_scores": scores,
        })

        # gentle rate limiting
        time.sleep(0.5)

    # aggregate
    criteria = ["domain_accuracy", "visual_grounding", "completeness", "word_count_adherence"]
    aggregated = {}
    for c in criteria:
        vals = [s["judge_scores"][c] for s in scored if c in s["judge_scores"]]
        if vals:
            aggregated[c] = {
                "mean": round(sum(vals) / len(vals), 3),
                "min":  min(vals),
                "max":  max(vals),
            }

    print("\nLLM-as-Judge aggregated scores (1–5):")
    for c, stats in aggregated.items():
        print(f"  {c:<25}  mean: {stats['mean']:.3f}")

    # save
    mode_tag = Path(args.predictions_json).stem.replace("_predictions", "")
    result = {"n_judged": len(scored), "aggregated": aggregated, "detailed": scored}
    (out / f"{mode_tag}_judge_scores.json").write_text(json.dumps(result, indent=2))
    print(f"\nJudge scores saved to {out}")


if __name__ == "__main__":
    main()
