"""
dataset.py
----------
PyTorch Dataset + collator for fine-tuning Qwen2-VL-2B-Instruct
on the panel → subcaption generation task.

The prompt format follows the Qwen2-VL chat template:
  <|im_start|>user
  <|vision_start|><|image_pad|><|vision_end|>
  Generate a concise 20-30 word subcaption for this materials science figure panel.
  <|im_end|>
  <|im_start|>assistant
  {subcaption}<|im_end|>

Loss is computed ONLY on the assistant tokens (labels = -100 elsewhere).
"""

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk, DatasetDict
from transformers import AutoProcessor
from PIL import Image
import re


SYSTEM_PROMPT = (
    "You are an expert materials scientist. "
    "Given a panel image from a scientific publication, "
    "write a precise 20-30 word subcaption describing what is shown."
)

USER_PROMPT = (
    "Generate a concise 20-30 word subcaption for this materials science "
    "figure panel. Focus on what is visualised, key observations, and any "
    "notable features."
)


class PanelCaptionDataset(Dataset):
    def __init__(
        self,
        hf_dataset,          # a HuggingFace Dataset split
        processor,           # Qwen2-VL AutoProcessor
        max_length: int = 512,
        is_train:   bool = True,
    ):
        self.data       = hf_dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_train   = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        image     = row["image"]
        subcap    = row["subcaption"]
        subtype   = row["visualization_subtype"]

        # ── build conversation ───────────────────────────────────────────────
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image",  "image": image},
                    {"type": "text",   "text":  USER_PROMPT},
                ],
            },
            {
                "role":    "assistant",
                "content": subcap,          # target during training
            },
        ]

        # apply_chat_template returns the full prompt string
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # process image + text together
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids      = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values   = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # ── compute labels: mask everything before <|im_start|>assistant ────
        labels = input_ids.clone()
        # find the last assistant marker and mask up to it
        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|im_start|>"
        )
        # locate last occurrence of im_start (the assistant turn)
        im_start_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]
        if len(im_start_positions) > 0:
            last_im_start = im_start_positions[-1].item()
            # +2 skips "<|im_start|>" and "assistant\n"
            labels[:last_im_start + 2] = -100
        labels[attention_mask == 0] = -100   # mask padding too

        sample = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        if pixel_values is not None:
            sample["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                sample["image_grid_thw"] = image_grid_thw

        # keep metadata for evaluation
        sample["_subcaption"]  = subcap
        sample["_subtype"]     = subtype
        sample["_figure_id"]   = row["figure_id"]
        sample["_panel_id"]    = row["panel_id"]

        return sample


class PanelCaptionCollator:
    """
    Collates variable-length sequences into padded batches.
    Handles the special pixel_values / image_grid_thw tensors from Qwen2-VL.
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[dict]) -> dict:
        batch = {}

        # standard tensors — stack directly (all padded to max_length already)
        for key in ("input_ids", "attention_mask", "labels"):
            batch[key] = torch.stack([s[key] for s in samples])

        # pixel_values might have different spatial sizes per image
        if "pixel_values" in samples[0]:
            batch["pixel_values"] = torch.cat(
                [s["pixel_values"] for s in samples], dim=0
            )
        if "image_grid_thw" in samples[0]:
            batch["image_grid_thw"] = torch.cat(
                [s["image_grid_thw"] for s in samples], dim=0
            )

        # metadata — keep as lists
        for key in ("_subcaption", "_subtype", "_figure_id", "_panel_id"):
            if key in samples[0]:
                batch[key] = [s[key] for s in samples]

        return batch


# ── inference-only prompt (no target appended) ────────────────────────────────

def build_inference_messages(image: Image.Image) -> list[dict]:
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  USER_PROMPT},
            ],
        },
    ]
