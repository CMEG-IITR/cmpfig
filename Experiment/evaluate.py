"""
Retrieval evaluation on val or test split.

Task 1.1  Image → Caption  : Recall@1/5/10, Mean Rank, Median Rank
Task 1.2  Caption → Image  : same metrics

Loads any checkpoint saved by train.py (or the base CLIP for zero-shot baseline).

Usage
-----
    python evaluate.py --split test                         # best checkpoint
    python evaluate.py --split val                          # val split
    python evaluate.py --split test --ckpt checkpoints/final_model
    python evaluate.py --split test --ckpt openai/clip-vit-base-patch32  # zero-shot
    python evaluate.py --split test --ckpt checkpoints/best_model --full-pool
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from dataset import PanelDataset

DEFAULT_CKPT       = "checkpoints/best_model"
DEFAULT_SPLITS_DIR = "splits"
DEFAULT_IMAGES_DIR = "crops_allloys_img_0.55"
DEFAULT_BATCH      = 128
RESULTS_DIR        = "results"


# ── encode ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_split(model, dataloader, device):
    """Returns (image_embs, text_embs) as np arrays, L2-normalised."""
    img_list, txt_list = [], []
    model.eval()
    for batch in tqdm(dataloader, desc="  encoding"):
        pv   = batch["pixel_values"].to(device)
        ids  = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ie   = F.normalize(model.visual_projection(model.vision_model(pixel_values=pv).pooler_output), dim=-1)
        te   = F.normalize(model.text_projection(model.text_model(input_ids=ids, attention_mask=attn).pooler_output), dim=-1)
        img_list.append(ie.cpu().numpy())
        txt_list.append(te.cpu().numpy())
    return np.vstack(img_list), np.vstack(txt_list)


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_retrieval_metrics(sim: np.ndarray, task_name: str) -> dict:
    """
    sim[i, j] = similarity between query i and candidate j.
    Correct match for query i is candidate i.
    """
    N = sim.shape[0]
    ranks = np.zeros(N, dtype=np.int64)
    for i in range(N):
        order = np.argsort(-sim[i])
        ranks[i] = int(np.where(order == i)[0][0]) + 1

    r1  = float((ranks <= 1).mean())  * 100
    r5  = float((ranks <= 5).mean())  * 100
    r10 = float((ranks <= 10).mean()) * 100
    mr  = float(ranks.mean())
    mdr = int(np.median(ranks))

    print(f"\n  {task_name}")
    print(f"    R@1  = {r1:.2f}%")
    print(f"    R@5  = {r5:.2f}%")
    print(f"    R@10 = {r10:.2f}%")
    print(f"    Mean Rank   = {mr:.1f}")
    print(f"    Median Rank = {mdr}")
    print(f"    Pool size   = {N}")

    return {
        "R@1":         round(r1, 2),
        "R@5":         round(r5, 2),
        "R@10":        round(r10, 2),
        "mean_rank":   round(mr, 2),
        "median_rank": mdr,
        "pool_size":   N,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        default=DEFAULT_CKPT,
                        help="Path to fine-tuned checkpoint or HF model name")
    parser.add_argument("--split",       default="test", choices=["val", "test"])
    parser.add_argument("--splits-dir",  default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--images-dir",  default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--full-pool",   action="store_true",
                        help="Use every pair in the split as candidates (default: True, "
                             "flag kept for explicitness)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== Retrieval Evaluation ===")
    print(f"  checkpoint : {args.ckpt}")
    print(f"  split      : {args.split}")
    print(f"  device     : {device}")

    # ── load model ────────────────────────────────────────────────────────────
    processor = CLIPProcessor.from_pretrained(args.ckpt)
    model     = CLIPModel.from_pretrained(args.ckpt).to(device)

    # ── dataset ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.splits_dir, f"{args.split}.csv")
    dataset  = PanelDataset(csv_path, args.images_dir, processor)
    loader   = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"  pairs      : {len(dataset)}\n")

    # ── encode ────────────────────────────────────────────────────────────────
    t0 = time.time()
    img_emb, txt_emb = encode_split(model, loader, device)
    encode_time = time.time() - t0
    print(f"\n  Encoded in {encode_time:.1f}s")

    # ── similarity ────────────────────────────────────────────────────────────
    sim = img_emb @ txt_emb.T   # [N, N]

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    i2t = compute_retrieval_metrics(sim,   "Task 1.1  Image → Caption")
    t2i = compute_retrieval_metrics(sim.T, "Task 1.2  Caption → Image")
    print("=" * 50)

    # ── save results ──────────────────────────────────────────────────────────
    ckpt_name = Path(args.ckpt).name
    out_name  = f"{ckpt_name}_{args.split}_{len(dataset)}pairs.json"
    out_path  = os.path.join(RESULTS_DIR, out_name)

    results = {
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint":  args.ckpt,
        "split":       args.split,
        "num_pairs":   len(dataset),
        "encode_time_s": round(encode_time, 2),
        "task_1_1_image_to_caption": i2t,
        "task_1_2_caption_to_image": t2i,
    }
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results saved -> {out_path}\n")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
