"""
Data loading, figure-level 80/10/10 panel split, and two-level hard-negative
batch sampling (same subtype → same category fallback).
"""
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, CLIPImageProcessor

from .config import Config


# ── Split ─────────────────────────────────────────────────────────────────────

def load_index(config: Config) -> List[dict]:
    path = os.path.join(config.data_root, config.dataset_index)
    with open(path) as f:
        return json.load(f)


def _dominant_subtype(subtypes: List[str]) -> str:
    return Counter(subtypes).most_common(1)[0][0]


def compute_split(
    all_data: List[dict],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rare_threshold: int,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Panel-aware figure-level stratified split.

    Targets val_frac / test_frac of PANELS (not figures) while preserving
    figure integrity — all panels of a figure stay in the same split.
    Within each visualization_subtype group, figures are shuffled then
    greedily assigned (val first, then test, rest to train) until panel
    count targets are met. Figures with < rare_threshold total panels
    in their subtype go entirely to train.
    """
    rng = np.random.default_rng(seed)

    # One pass: count panels per figure, record dominant subtype
    fig_panel_count: Dict[str, int] = {}
    fig_subtype_lists: Dict[str, List[str]] = defaultdict(list)
    for item in all_data:
        fid = item["image_id"]
        fig_subtype_lists[fid].append(item["visualization_subtype"])
        fig_panel_count[fid] = fig_panel_count.get(fid, 0) + 1

    # Group figures by dominant subtype, carry panel count
    subtype_groups: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for fid, subtypes in fig_subtype_lists.items():
        dom = _dominant_subtype(subtypes)
        subtype_groups[dom].append((fid, fig_panel_count[fid]))

    train_ids, val_ids, test_ids = [], [], []

    for subtype, fig_list in subtype_groups.items():
        total_panels = sum(c for _, c in fig_list)

        if total_panels < rare_threshold:
            train_ids.extend(f for f, _ in fig_list)
            continue

        # Shuffle figures (not panels) to randomise which figures end up in val/test
        indices = rng.permutation(len(fig_list)).tolist()
        shuffled = [fig_list[i] for i in indices]

        n_val_target  = max(1, round(total_panels * val_frac))
        n_test_target = max(1, round(total_panels * test_frac))

        val_panels = test_panels = 0
        for fid, count in shuffled:
            if val_panels < n_val_target:
                val_ids.append(fid)
                val_panels += count
            elif test_panels < n_test_target:
                test_ids.append(fid)
                test_panels += count
            else:
                train_ids.append(fid)

    return train_ids, val_ids, test_ids


def save_split(output_dir: str, train_ids, val_ids, test_ids):
    # Save to /tmp (Linux fs, fast) to avoid WSL2 NTFS write penalty.
    # Also write a flag to output_dir so the split is detected on restart.
    tmp_path = "/tmp/retrieval_split.npz"
    np.savez(tmp_path,
             train_ids=np.array(train_ids, dtype=object),
             val_ids=np.array(val_ids, dtype=object),
             test_ids=np.array(test_ids, dtype=object))
    # Flag file on NTFS is tiny — just records the seed so we can validate later
    flag_dir = os.path.join(output_dir, "split")
    os.makedirs(flag_dir, exist_ok=True)
    with open(os.path.join(flag_dir, "split_done.txt"), "w") as f:
        f.write("split_ready\n")
    return tmp_path


def load_split(output_dir: str):
    # Try /tmp first (fast), fall back to recomputing
    tmp_path = "/tmp/retrieval_split.npz"
    if os.path.exists(tmp_path):
        d = np.load(tmp_path, allow_pickle=True)
        return {
            "train_ids": d["train_ids"].tolist(),
            "val_ids":   d["val_ids"].tolist(),
            "test_ids":  d["test_ids"].tolist(),
        }
    return None


def split_exists(output_dir: str) -> bool:
    flag = os.path.join(output_dir, "split", "split_done.txt")
    tmp  = "/tmp/retrieval_split.npz"
    return os.path.exists(flag) and os.path.exists(tmp)


def print_split_stats(train_data, val_data, test_data):
    total = len(train_data) + len(val_data) + len(test_data)
    print(f"Split statistics (total {total:,} panels):")
    print(f"  Train : {len(train_data):>7,}  ({100*len(train_data)/total:.1f}%)")
    print(f"  Val   : {len(val_data):>7,}  ({100*len(val_data)/total:.1f}%)")
    print(f"  Test  : {len(test_data):>7,}  ({100*len(test_data)/total:.1f}%)")
    for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        n_figs = len(set(d["image_id"] for d in data))
        n_subs = len(set(d["visualization_subtype"] for d in data))
        print(f"  {name}: {n_figs:,} figures, {n_subs} subtypes")


# ── Dataset ───────────────────────────────────────────────────────────────────

class PanelDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        data_root: str,
        image_processor: CLIPImageProcessor,
        tokenizer: AutoTokenizer,
        max_text_len: int = 256,
    ):
        self.data       = data
        self.data_root  = data_root
        self.image_processor = image_processor
        self.tokenizer  = tokenizer
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        img_path = os.path.join(self.data_root, item["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=0)

        pixel_values = self.image_processor(images=img, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        text = item["summary"] or item["subcaption"] or ""
        enc  = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values":  pixel_values,
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "subtype":       item["visualization_subtype"],
            "category":      item["visualization_category"],
            "image_id":      item["image_id"],
            "idx":           idx,
        }


# ── Two-Level Hard-Negative Batch Sampler ─────────────────────────────────────

class HardNegativeBatchSampler(Sampler):
    """
    Two-level hard negative sampling per batch:
      Level 1: same visualization_subtype, different figure.
               Used when subtype pool >= fallback_threshold panels.
      Level 2: same visualization_category (broader), different figure.
               Fallback when subtype pool is too small.
      Level 3: random panel. Final fallback.

    Batch layout (size B):
      n_anchor = B - n_hard  randomly sampled anchors
      n_hard   = round(B * hard_neg_ratio)  hard negatives (one per anchor)
    """

    def __init__(
        self,
        data: List[dict],
        batch_size: int,
        hard_neg_ratio: float = 0.5,
        fallback_threshold: int = 50,
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.data      = data
        self.batch_size = batch_size
        self.n_hard    = max(1, round(batch_size * hard_neg_ratio))
        self.n_anchor  = batch_size - self.n_hard
        self.fallback_threshold = fallback_threshold
        self.drop_last = drop_last
        self.rng       = np.random.default_rng(seed)
        self.n         = len(data)

        # Build lookup maps
        subtype_map:  Dict[str, List[int]] = defaultdict(list)
        category_map: Dict[str, List[int]] = defaultdict(list)
        for i, item in enumerate(data):
            subtype_map[item["visualization_subtype"]].append(i)
            category_map[item["visualization_category"]].append(i)

        self.subtype_to_idx:  Dict[str, np.ndarray] = {
            k: np.array(v) for k, v in subtype_map.items()
        }
        self.category_to_idx: Dict[str, np.ndarray] = {
            k: np.array(v) for k, v in category_map.items()
        }

    def __len__(self) -> int:
        return self.n // self.n_anchor

    def _sample_hard_neg(self, ai: int) -> int:
        item     = self.data[ai]
        fig_id   = item["image_id"]
        subtype  = item["visualization_subtype"]
        category = item["visualization_category"]

        sub_pool = self.subtype_to_idx[subtype]

        # Level 1: same subtype (if pool is large enough)
        if len(sub_pool) >= self.fallback_threshold:
            for _ in range(20):
                neg = int(self.rng.choice(sub_pool))
                if neg != ai and self.data[neg]["image_id"] != fig_id:
                    return neg

        # Level 2: same category (broader fallback)
        cat_pool = self.category_to_idx[category]
        if len(cat_pool) > 1:
            for _ in range(20):
                neg = int(self.rng.choice(cat_pool))
                if neg != ai and self.data[neg]["image_id"] != fig_id:
                    return neg

        # Level 3: random
        return int(self.rng.integers(self.n))

    def __iter__(self):
        perm = self.rng.permutation(self.n)

        for start in range(0, self.n - self.n_anchor + 1, self.n_anchor):
            anchor_idxs = perm[start : start + self.n_anchor].tolist()

            hard_idxs = [self._sample_hard_neg(ai) for ai in anchor_idxs][: self.n_hard]

            batch = anchor_idxs + hard_idxs
            while len(batch) < self.batch_size:
                batch.append(int(self.rng.integers(self.n)))

            yield batch[: self.batch_size]


# ── DataLoader builders ───────────────────────────────────────────────────────

def _make_processors(config: Config):
    image_processor = CLIPImageProcessor.from_pretrained(config.image_encoder)
    tokenizer       = AutoTokenizer.from_pretrained(config.text_encoder)
    return image_processor, tokenizer


def _collate(batch):
    return {
        "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "subtype":        [b["subtype"]   for b in batch],
        "category":       [b["category"]  for b in batch],
        "image_id":       [b["image_id"]  for b in batch],
        "idx":            torch.tensor([b["idx"] for b in batch]),
    }


def build_dataloaders(
    config: Config,
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
    epoch_seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    image_processor, tokenizer = _make_processors(config)

    train_ds = PanelDataset(train_data, config.data_root, image_processor, tokenizer, config.max_text_len)
    val_ds   = PanelDataset(val_data,   config.data_root, image_processor, tokenizer, config.max_text_len)
    test_ds  = PanelDataset(test_data,  config.data_root, image_processor, tokenizer, config.max_text_len)

    train_sampler = HardNegativeBatchSampler(
        train_data,
        batch_size=config.batch_size,
        hard_neg_ratio=config.hard_neg_ratio,
        fallback_threshold=config.hard_neg_fallback_threshold,
        seed=epoch_seed,
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        collate_fn=_collate,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        collate_fn=_collate,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        collate_fn=_collate,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
