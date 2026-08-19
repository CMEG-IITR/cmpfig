"""
Step 1 (run once per checkpoint): encode the full test split and cache the
embeddings + metadata to disk. This is the slow GPU pass.

Usage:
    python -m retrieval.build_embeddings --output_dir outputs_no_hardneg_20ep

Then run examples instantly, as many times as you like, without re-encoding:
    python -m retrieval.run_examples --output_dir outputs_no_hardneg_20ep --n_examples 5
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import Config
from .demo_common import build_test_data, get_device_dtype, load_model, resolve_paths


@torch.no_grad()
def encode_all(model, config, test_data, device, dtype, batch_size):
    from transformers import AutoTokenizer, CLIPImageProcessor
    image_processor = CLIPImageProcessor.from_pretrained(config.image_encoder)
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)

    img_embs, txt_embs = [], []
    starts = range(0, len(test_data), batch_size)
    for start in tqdm(starts, desc="Encoding test split", total=len(starts)):
        chunk = test_data[start:start + batch_size]

        imgs = []
        for item in chunk:
            path = os.path.join(config.data_root, item["image_path"])
            try:
                imgs.append(Image.open(path).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224), color=0))
        pv = image_processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)

        texts = [item["summary"] or item["subcaption"] or "" for item in chunk]
        enc = tokenizer(texts, max_length=config.max_text_len, padding="max_length",
                         truncation=True, return_tensors="pt")
        ii, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype):
            ie = model.encode_image(pv)
            te = model.encode_text(ii, am)
        img_embs.append(ie.cpu().float().numpy())
        txt_embs.append(te.cpu().float().numpy())

    return np.concatenate(img_embs, axis=0), np.concatenate(txt_embs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="outputs_no_hardneg_20ep",
                    help="dir containing checkpoints/best.pt")
    p.add_argument("--ckpt", type=str, default=None, help="override checkpoint path")
    p.add_argument("--cache", type=str, default=None, help="override cache output path")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    output_dir, ckpt_path, cache_path = resolve_paths(args.output_dir, args.ckpt, args.cache)

    config = Config()
    device, dtype = get_device_dtype(config)
    print(f"Device: {device}  dtype: {dtype}  (cuda available: {torch.cuda.is_available()})")

    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(config, ckpt_path, device)

    print("Reproducing test split …")
    test_data = build_test_data(config)
    print(f"  test split: {len(test_data):,} panels")

    print(f"Encoding test split, batch_size={args.batch_size} …")
    img_embs, txt_embs = encode_all(model, config, test_data, device, dtype, args.batch_size)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        img_embs=img_embs,
        txt_embs=txt_embs,
        image_id=np.array([d["image_id"] for d in test_data], dtype=object),
        image_path=np.array([d["image_path"] for d in test_data], dtype=object),
        summary=np.array([d["summary"] for d in test_data], dtype=object),
        subcaption=np.array([d["subcaption"] for d in test_data], dtype=object),
        subtype=np.array([d["visualization_subtype"] for d in test_data], dtype=object),
        category=np.array([d["visualization_category"] for d in test_data], dtype=object),
    )
    print(f"Saved -> {cache_path}")
    print(f"Now run: python -m retrieval.run_examples --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
