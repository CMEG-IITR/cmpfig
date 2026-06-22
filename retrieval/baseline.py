"""
Baselines for cross-modal panel retrieval.

Baseline 1 — Zero-shot CLIP
    CLIP ViT-B/32 with its own text encoder, no fine-tuning.
    Uses CLIPModel.get_image_features() / get_text_features() → 512-d, L2-normalised.

Baseline 2 — Random retrieval
    Analytical: R@K = K / N  (uniform random ranking over gallery of size N).

Usage:
    python -m retrieval.baseline
    python -m retrieval.baseline --output_dir retrieval/outputs_proper
"""
import argparse
import json
import os
import sys

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.config import Config
from retrieval.data import (
    build_dataloaders,
    compute_split,
    load_index,
    load_split,
    save_split,
    split_exists,
)
from retrieval.eval import _build_index, compute_metrics, category_breakdown, print_results


# ── Zero-shot CLIP dataset ────────────────────────────────────────────────────

class CLIPPanelDataset(Dataset):
    """Wraps panels for CLIP's own processor (image + text together)."""

    def __init__(self, data, data_root, processor, max_text_len=77):
        self.data       = data
        self.data_root  = data_root
        self.processor  = processor
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = os.path.join(self.data_root, item["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=0)

        text = item["summary"] or item["subcaption"] or ""

        img_enc = self.processor(images=img, return_tensors="pt", padding=False)
        txt_enc = self.processor(
            text=text,
            return_tensors="pt",
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
        )

        return {
            "pixel_values":  img_enc["pixel_values"].squeeze(0),
            "input_ids":     txt_enc["input_ids"].squeeze(0),
            "attention_mask":txt_enc["attention_mask"].squeeze(0),
            "subtype":       item["visualization_subtype"],
            "category":      item["visualization_category"],
        }


def _clip_collate(batch):
    return {
        "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "subtype":        [b["subtype"]   for b in batch],
        "category":       [b["category"] for b in batch],
    }


# ── Embedding extraction (zero-shot CLIP) ─────────────────────────────────────

@torch.no_grad()
def encode_zeroshot_clip(model, loader, device):
    model.eval()
    img_list, txt_list, sub_list, cat_list = [], [], [], []

    for batch in tqdm(loader, desc="Zero-shot CLIP encoding", leave=False):
        pv = batch["pixel_values"].to(device)
        ii = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)

        img_out  = model.vision_model(pixel_values=pv)
        img_feat = model.visual_projection(img_out.pooler_output)

        txt_out  = model.text_model(input_ids=ii, attention_mask=am)
        txt_feat = model.text_projection(txt_out.pooler_output)

        img_feat = F.normalize(img_feat.float(), dim=-1)
        txt_feat = F.normalize(txt_feat.float(), dim=-1)

        img_list.append(img_feat.cpu().numpy())
        txt_list.append(txt_feat.cpu().numpy())
        sub_list.extend(batch["subtype"])
        cat_list.extend(batch["category"])

    return (
        np.concatenate(img_list, axis=0),
        np.concatenate(txt_list, axis=0),
        sub_list,
        cat_list,
    )


# ── Baseline 2: random retrieval (analytical) ─────────────────────────────────

def random_baseline_metrics(N, k_values, mrr_cutoff=1000):
    """
    For a uniform random ranker over gallery of size N:
      R@K  = K / N
      MRR  = (1/N) * sum_{r=1}^{min(N, cutoff)} 1/r  ≈  H(min(N,cutoff)) / N
      mAP@100 = 100 / (2 * N)   [expected rank of correct item = (N+1)/2]
    """
    results = {}
    for k in k_values:
        results[f"R@{k}"] = min(k / N, 1.0)

    cutoff = min(N, mrr_cutoff)
    mrr = sum(1.0 / r for r in range(1, cutoff + 1)) / N
    results["MRR"] = mrr

    # mAP@100: AP = 1/rank if rank<=100 else 0; E[rank] = (N+1)/2
    # P(rank <= 100) = 100/N; E[1/rank | rank<=100] ≈ H(100)/100
    h100 = sum(1.0 / r for r in range(1, 101))
    results["mAP@100"] = (100 / N) * (h100 / 100) if N > 0 else 0.0

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str,
                   default="/mnt/d/Subham/Compoun_img_01/retrieval/outputs_proper")
    p.add_argument("--data_root",  type=str,
                   default="/mnt/d/Subham/Compoun_img_01/main_data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = Config()
    config.output_dir = args.output_dir
    config.data_root  = args.data_root

    # ── Reconstruct test split ────────────────────────────────────────────────
    all_data = load_index(config)
    print(f"Loaded {len(all_data):,} panels")

    if split_exists(config.output_dir):
        saved = load_split(config.output_dir)
        train_ids, val_ids, test_ids = (
            saved["train_ids"], saved["val_ids"], saved["test_ids"]
        )
    else:
        train_ids, val_ids, test_ids = compute_split(
            all_data, config.train_frac, config.val_frac, config.test_frac,
            config.rare_subtype_threshold, config.split_seed,
        )
        save_split(config.output_dir, train_ids, val_ids, test_ids)

    test_set = set(test_ids)
    test_data = [d for d in all_data if d["image_id"] in test_set]
    print(f"Test panels: {len(test_data):,}")

    k_values   = [1, 5, 10, 50, 100]
    mrr_cutoff = 1000
    eval_dir   = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    all_results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE 1 — Zero-shot CLIP
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("BASELINE 1 — Zero-shot CLIP ViT-B/32")
    print("="*60)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_ds = CLIPPanelDataset(test_data, args.data_root, clip_proc, max_text_len=77)
    clip_loader = DataLoader(
        clip_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_clip_collate,
        persistent_workers=args.num_workers > 0,
    )

    img_embs, txt_embs, subtypes, categories = encode_zeroshot_clip(
        clip_model, clip_loader, device
    )
    N = len(subtypes)
    gt = np.arange(N)

    idx_txt = _build_index(txt_embs)
    idx_img = _build_index(img_embs)

    m_i2t = compute_metrics(img_embs, idx_txt, gt, k_values, mrr_cutoff)
    m_t2i = compute_metrics(txt_embs, idx_img, gt, k_values, mrr_cutoff)

    zs_overall = {
        **{f"i2t_{k}": v for k, v in m_i2t.items()},
        **{f"t2i_{k}": v for k, v in m_t2i.items()},
        "n_samples": N,
    }
    zs_by_cat = category_breakdown(
        img_embs, txt_embs, categories, k_values, mrr_cutoff, min_samples=50
    )
    zs_results = {"overall": zs_overall, "by_category": zs_by_cat}
    all_results["zero_shot_clip"] = zs_results

    print(f"\nZero-shot CLIP  ({N:,} test panels)")
    print(f"  {'Metric':<12}  {'i2t':>8}  {'t2i':>8}")
    print(f"  {'-'*30}")
    for k in k_values:
        print(f"  R@{k:<9}  {m_i2t[f'R@{k}']:>8.4f}  {m_t2i[f'R@{k}']:>8.4f}")
    print(f"  {'MRR':<12}  {m_i2t['MRR']:>8.4f}  {m_t2i['MRR']:>8.4f}")
    print(f"  {'mAP@100':<12}  {m_i2t['mAP@100']:>8.4f}  {m_t2i['mAP@100']:>8.4f}")

    del clip_model, clip_loader, img_embs, txt_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE 2 — Random retrieval (analytical)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("BASELINE 2 — Random Retrieval  (analytical)")
    print("="*60)

    rnd = random_baseline_metrics(N, k_values, mrr_cutoff)
    rnd_overall = {
        **{f"i2t_{k}": v for k, v in rnd.items()},
        **{f"t2i_{k}": v for k, v in rnd.items()},
        "n_samples": N,
    }
    all_results["random"] = {"overall": rnd_overall, "by_category": {}}

    print(f"\nRandom  ({N:,} panels,  gallery size = {N:,})")
    print(f"  {'Metric':<12}  {'value':>8}")
    print(f"  {'-'*22}")
    for k in k_values:
        print(f"  R@{k:<9}  {rnd[f'R@{k}']:>8.6f}")
    print(f"  {'MRR':<12}  {rnd['MRR']:>8.6f}")
    print(f"  {'mAP@100':<12}  {rnd['mAP@100']:>8.6f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════════════════
    out_path = os.path.join(eval_dir, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ── Comparison summary ────────────────────────────────────────────────────
    # Load our fine-tuned results for side-by-side
    ft_path = os.path.join(eval_dir, "test_results.json")
    if os.path.exists(ft_path):
        ft = json.load(open(ft_path))["overall"]
        print("\n" + "="*70)
        print("COMPARISON SUMMARY  (i2t)")
        print(f"  {'Metric':<10}  {'Random':>10}  {'Zero-shot CLIP':>16}  {'Ours (FT)':>12}")
        print(f"  {'-'*52}")
        for k in k_values:
            key = f"R@{k}"
            print(
                f"  {key:<10}  {rnd[key]:>10.4f}  "
                f"{zs_overall[f'i2t_{key}']:>16.4f}  "
                f"{ft[f'i2t_{key}']:>12.4f}"
            )
        print(
            f"  {'MRR':<10}  {rnd['MRR']:>10.4f}  "
            f"{zs_overall['i2t_MRR']:>16.4f}  "
            f"{ft['i2t_MRR']:>12.4f}"
        )
        print(
            f"  {'mAP@100':<10}  {rnd['mAP@100']:>10.4f}  "
            f"{zs_overall['i2t_mAP@100']:>16.4f}  "
            f"{ft['i2t_mAP@100']:>12.4f}"
        )
        print("="*70)


if __name__ == "__main__":
    main()
