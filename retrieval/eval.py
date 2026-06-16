"""
FAISS-based evaluation.
Reports per-direction: R@1, R@5, R@10, R@50, R@100, MRR, mAP@100
Plus category-level breakdown.
"""
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import DualEncoder


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def encode_dataset(
    model: DualEncoder,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Returns img_embs [N,D], txt_embs [N,D], subtypes [N], categories [N].
    """
    model.eval()
    img_list, txt_list, sub_list, cat_list = [], [], [], []

    for batch in tqdm(loader, desc="Encoding", leave=False):
        pv = batch["pixel_values"].to(device)
        ii = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype):
            ie = model.encode_image(pv)
            te = model.encode_text(ii, am)

        img_list.append(ie.cpu().float().numpy())
        txt_list.append(te.cpu().float().numpy())
        sub_list.extend(batch["subtype"])
        cat_list.extend(batch["category"])

    return (
        np.concatenate(img_list, axis=0),
        np.concatenate(txt_list, axis=0),
        sub_list,
        cat_list,
    )


# ── FAISS index ───────────────────────────────────────────────────────────────

def _build_index(embs: np.ndarray) -> faiss.Index:
    d = embs.shape[1]
    cpu_idx = faiss.IndexFlatIP(d)
    cpu_idx.add(embs.astype(np.float32))
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu_idx)
    return cpu_idx


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(
    queries: np.ndarray,
    index: faiss.Index,
    gt_indices: np.ndarray,
    k_values: List[int],
    mrr_cutoff: int = 1000,
) -> Dict[str, float]:
    """
    Computes R@K, MRR, and mAP@100 for a set of queries.

    R@K    : fraction of queries where correct answer is in top-K.
    MRR    : mean reciprocal rank (searched up to mrr_cutoff).
    mAP@100: mean average precision at 100 (for single-relevant retrieval,
             AP = 1/rank if rank ≤ 100, else 0).
    """
    max_search = max(max(k_values), min(mrr_cutoff, index.ntotal))
    _, retrieved = index.search(queries.astype(np.float32), max_search)  # [N, max_search]
    N = len(queries)

    results = {}

    # R@K
    for k in k_values:
        hits = np.any(retrieved[:, :k] == gt_indices[:, None], axis=1)
        results[f"R@{k}"] = float(hits.mean())

    # MRR (up to mrr_cutoff)
    mrr_depth = min(mrr_cutoff, max_search)
    rr = np.zeros(N)
    for i in range(N):
        pos = np.where(retrieved[i, :mrr_depth] == gt_indices[i])[0]
        if len(pos) > 0:
            rr[i] = 1.0 / (pos[0] + 1)   # pos is 0-indexed, rank is 1-indexed
    results["MRR"] = float(rr.mean())

    # mAP@100 (single relevant item: AP = 1/rank if rank ≤ 100, else 0)
    ap = np.zeros(N)
    for i in range(N):
        pos = np.where(retrieved[i, :100] == gt_indices[i])[0]
        if len(pos) > 0:
            ap[i] = 1.0 / (pos[0] + 1)
    results["mAP@100"] = float(ap.mean())

    return results


# ── Category breakdown ────────────────────────────────────────────────────────

def category_breakdown(
    img_embs: np.ndarray,
    txt_embs: np.ndarray,
    categories: List[str],
    k_values: List[int],
    mrr_cutoff: int = 1000,
    min_samples: int = 10,
) -> Dict[str, Dict[str, float]]:
    cats_arr = np.array(categories)
    results  = {}

    for cat in sorted(set(categories)):
        mask = cats_arr == cat
        if mask.sum() < min_samples:
            continue

        sub_img = img_embs[mask]
        sub_txt = txt_embs[mask]
        sub_gt  = np.arange(mask.sum())

        idx_txt = _build_index(sub_txt)
        idx_img = _build_index(sub_img)

        m_i2t = compute_metrics(sub_img, idx_txt, sub_gt, k_values, mrr_cutoff)
        m_t2i = compute_metrics(sub_txt, idx_img, sub_gt, k_values, mrr_cutoff)

        results[cat] = {
            **{f"i2t_{k}": v for k, v in m_i2t.items()},
            **{f"t2i_{k}": v for k, v in m_t2i.items()},
            "n": int(mask.sum()),
        }

    return results


# ── Main evaluation entry point ───────────────────────────────────────────────

def evaluate(
    model: DualEncoder,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    k_values: List[int],
    mrr_cutoff: int = 1000,
    min_category_samples: int = 10,
    output_path: Optional[str] = None,
) -> Dict:
    img_embs, txt_embs, subtypes, categories = encode_dataset(
        model, loader, device, dtype
    )
    N  = len(subtypes)
    gt = np.arange(N)

    idx_txt = _build_index(txt_embs)
    idx_img = _build_index(img_embs)

    m_i2t = compute_metrics(img_embs, idx_txt, gt, k_values, mrr_cutoff)
    m_t2i = compute_metrics(txt_embs, idx_img, gt, k_values, mrr_cutoff)

    overall = {
        **{f"i2t_{k}": v for k, v in m_i2t.items()},
        **{f"t2i_{k}": v for k, v in m_t2i.items()},
        "n_samples": N,
    }

    by_category = category_breakdown(
        img_embs, txt_embs, categories,
        k_values, mrr_cutoff, min_category_samples,
    )

    results = {"overall": overall, "by_category": by_category}

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


# ── Console printer ───────────────────────────────────────────────────────────

def print_results(results: Dict, label: str = ""):
    prefix = f"[{label}] " if label else ""
    ov = results["overall"]
    N  = ov["n_samples"]

    print(f"\n{prefix}Overall ({N:,} samples)")
    print(f"  {'Metric':<12}  {'i2t':>8}  {'t2i':>8}")
    print(f"  {'-'*30}")

    for k in sorted({int(key.split('R@')[1]) for key in ov if '_R@' in key}):
        i2t = ov.get(f"i2t_R@{k}", float("nan"))
        t2i = ov.get(f"t2i_R@{k}", float("nan"))
        print(f"  R@{k:<9}  {i2t:>8.4f}  {t2i:>8.4f}")

    print(f"  {'MRR':<12}  {ov.get('i2t_MRR', float('nan')):>8.4f}  {ov.get('t2i_MRR', float('nan')):>8.4f}")
    print(f"  {'mAP@100':<12}  {ov.get('i2t_mAP@100', float('nan')):>8.4f}  {ov.get('t2i_mAP@100', float('nan')):>8.4f}")

    by_cat = results.get("by_category", {})
    if by_cat:
        print(f"\n{prefix}Category breakdown:")
        print(f"  {'Category':<30}  {'n':>6}  {'i2t R@10':>9}  {'i2t R@100':>10}  {'i2t MRR':>8}  {'i2t mAP@100':>12}")
        print(f"  {'-'*80}")
        for cat, m in sorted(by_cat.items()):
            print(
                f"  {cat:<30}  {m['n']:>6}  "
                f"{m.get('i2t_R@10',0):>9.4f}  "
                f"{m.get('i2t_R@100',0):>10.4f}  "
                f"{m.get('i2t_MRR',0):>8.4f}  "
                f"{m.get('i2t_mAP@100',0):>12.4f}"
            )
