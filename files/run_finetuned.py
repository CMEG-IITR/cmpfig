"""
evaluation/run_finetuned.py
---------------------------
Run the fine-tuned LoRA model on the test set and save predictions
in the same format as zero_shot.py, so metrics.py works on both.

Usage:
  python -m evaluation.run_finetuned \
      --adapter_dir  ./checkpoints/qwen2vl_subcap/final_adapter \
      --base_model   Qwen/Qwen2-VL-2B-Instruct \
      --dataset_dir  /path/to/matfig_captioning \
      --output_dir   ./results/finetuned
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

from data.dataset import build_inference_messages


def load_finetuned(adapter_dir: str, base_model: str):
    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, processor


@torch.no_grad()
def generate_caption(model, processor, image, max_new_tokens: int = 80) -> str:
    messages = build_inference_messages(image)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir",  required=True)
    parser.add_argument("--base_model",   default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset_dir",  required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dd    = load_from_disk(args.dataset_dir)
    model, processor = load_finetuned(args.adapter_dir, args.base_model)

    results = []
    for row in tqdm(dd["test"], desc="Fine-tuned inference"):
        prediction = generate_caption(
            model, processor, row["image"], args.max_new_tokens
        )
        results.append({
            "figure_id":             row["figure_id"],
            "panel_id":              row["panel_id"],
            "visualization_subtype": row["visualization_subtype"],
            "reference":             row["subcaption"],
            "prediction":            prediction,
            "mode":                  "finetuned",
        })

    out_file = out / "finetuned_predictions.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} predictions → {out_file}")


if __name__ == "__main__":
    main()
