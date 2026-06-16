"""
baselines/zero_shot.py
----------------------
Runs the unmodified Qwen2-VL-2B-Instruct model on the test split
as a zero-shot and 3-shot baseline.

Zero-shot: just the task prompt + image, no examples.
Few-shot:  3 (image, subcaption) exemplars prepended, then the test image.
           Exemplars are drawn from the training set, one per unique subtype
           closest to the test sample's subtype.

Usage:
  python -m baselines.zero_shot \
      --dataset_dir  /path/to/matfig_captioning \
      --model_name   Qwen/Qwen2-VL-2B-Instruct \
      --output_dir   ./results/zero_shot \
      --mode         zero_shot        # or few_shot
      --batch_size   4
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from data.dataset import build_inference_messages, SYSTEM_PROMPT, USER_PROMPT


def load_model(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def generate_caption(model, processor, messages: list[dict], max_new_tokens=80) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # collect all images from message content
    images = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    images.append(item["image"])

    inputs = processor(
        text=[text], images=images if images else None,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    # decode only the newly generated tokens
    generated = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_few_shot_messages(
    test_image, exemplars: list[dict], n_shots: int = 3
) -> list[dict]:
    """Build a multi-turn few-shot prompt."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in exemplars[:n_shots]:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": ex["image"]},
                {"type": "text",  "text":  USER_PROMPT},
            ],
        })
        messages.append({
            "role":    "assistant",
            "content": ex["subcaption"],
        })

    # actual test query
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text",  "text":  USER_PROMPT},
        ],
    })
    return messages


def build_exemplar_pool(train_ds, n_per_subtype: int = 3) -> dict:
    """Build a pool of exemplars indexed by visualization_subtype."""
    pool = defaultdict(list)
    for row in train_ds:
        pool[row["visualization_subtype"]].append({
            "image":      row["image"],
            "subcaption": row["subcaption"],
        })
    # cap per subtype
    return {k: v[:n_per_subtype] for k, v in pool.items()}


def run_evaluation(
    model, processor, test_ds, mode: str,
    exemplar_pool: dict | None = None
) -> list[dict]:
    results = []
    for row in tqdm(test_ds, desc=f"Generating [{mode}]"):
        image   = row["image"]
        subtype = row["visualization_subtype"]

        if mode == "zero_shot":
            messages = build_inference_messages(image)
        else:  # few_shot
            exemplars = exemplar_pool.get(subtype, [])
            if not exemplars:
                # fall back to any subtype
                for v in exemplar_pool.values():
                    if v:
                        exemplars = v
                        break
            messages = build_few_shot_messages(image, exemplars, n_shots=3)

        prediction = generate_caption(model, processor, messages)
        results.append({
            "figure_id":              row["figure_id"],
            "panel_id":               row["panel_id"],
            "visualization_subtype":  subtype,
            "reference":              row["subcaption"],
            "prediction":             prediction,
            "mode":                   mode,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_name",  default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "both"],
                        default="both")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dd = load_from_disk(args.dataset_dir)
    model, processor = load_model(args.model_name)

    exemplar_pool = None
    if args.mode in ("few_shot", "both"):
        exemplar_pool = build_exemplar_pool(dd["train"])

    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]
    for mode in modes:
        results = run_evaluation(
            model, processor, dd["test"], mode, exemplar_pool
        )
        out_file = out / f"{mode}_predictions.json"
        out_file.write_text(json.dumps(results, indent=2))
        print(f"Saved {len(results)} predictions → {out_file}")


if __name__ == "__main__":
    main()
