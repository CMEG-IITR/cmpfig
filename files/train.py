"""
train.py
--------
Fine-tune Qwen2-VL-2B-Instruct on panel → subcaption generation
using QLoRA (4-bit base + LoRA adapters).

Designed to run on a single A100 40GB or two A40s.
Swap --model_name for any HF-hosted VLM with a compatible chat template.

Usage:
  python train.py \
      --dataset_dir /path/to/output/matfig_captioning \
      --output_dir  ./checkpoints/qwen2vl_subcap \
      --model_name  Qwen/Qwen2-VL-2B-Instruct \
      --epochs      5 \
      --batch_size  8 \
      --lr          2e-4

Alternatives for --model_name:
  Qwen/Qwen2-VL-7B-Instruct   (needs ~40 GB)
  llava-hf/llava-1.5-7b-hf
  google/paligemma2-3b-pt-224
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from data.dataset import PanelCaptionDataset, PanelCaptionCollator


# ── LoRA config ───────────────────────────────────────────────────────────────

def get_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                    # rank — increase to 32/64 for better quality
        lora_alpha=32,           # scaling = lora_alpha / r
        target_modules=[         # Qwen2-VL attention + MLP projection layers
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )


# ── quantisation config (QLoRA) ───────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(model_name: str, use_qlora: bool = True):
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    # Qwen2-VL pads on the left for generation; ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bnb_cfg = get_bnb_config() if use_qlora else None

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    return model, processor


# ── training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",  required=True,
                        help="Path to matfig_captioning dataset (from prepare_data.py)")
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--model_name",   default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=8,
                        help="Per-device batch size; effective = batch_size × grad_accum")
    parser.add_argument("--grad_accum",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max_length",   type=int,   default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--no_qlora",     action="store_true",
                        help="Disable 4-bit quantisation (needs more VRAM)")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. load dataset
    print("Loading dataset …")
    dd = load_from_disk(args.dataset_dir)
    print(f"  train: {len(dd['train'])}  val: {len(dd['val'])}  test: {len(dd['test'])}")

    # 2. load model
    print(f"Loading model {args.model_name} …")
    model, processor = load_model_and_processor(
        args.model_name, use_qlora=not args.no_qlora
    )

    # 3. datasets
    train_ds = PanelCaptionDataset(
        dd["train"], processor, max_length=args.max_length, is_train=True
    )
    val_ds = PanelCaptionDataset(
        dd["val"],   processor, max_length=args.max_length, is_train=False
    )
    collator = PanelCaptionCollator(processor.tokenizer.pad_token_id)

    # 4. training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,   # needed: we have custom columns
        report_to="tensorboard",
        seed=args.seed,
        run_name="matfig_subcap_qwen2vl",
    )

    # 5. trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. train
    print("Starting training …")
    trainer.train()

    # 7. save final adapter weights
    final_path = output_dir / "final_adapter"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"Adapter saved to {final_path}")


if __name__ == "__main__":
    main()
