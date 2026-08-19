"""Shared helpers for build_embeddings.py, run_examples.py, and paper_figures.py."""
import os
import shutil

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import Config
from .data import compute_split, load_index
from .model import DualEncoder
from .train import load_checkpoint

_CKPT_CACHE_DIR = "/tmp/retrieval_demo_cache/ckpt_cache"


def _copy_with_progress(src: str, dest: str, chunk_size: int = 8 * 1024 * 1024):
    total = os.path.getsize(src)
    with open(src, "rb") as fsrc, open(dest, "wb") as fdst, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Copying {os.path.basename(src)}"
    ) as pbar:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            pbar.update(len(chunk))
    shutil.copystat(src, dest)


def _fast_local_copy(path: str) -> str:
    """
    """
    if not path.startswith("/mnt/"):
        return path
    os.makedirs(_CKPT_CACHE_DIR, exist_ok=True)
    dest = os.path.join(_CKPT_CACHE_DIR, os.path.basename(path))
    src_size = os.path.getsize(path)
    if os.path.exists(dest) and os.path.getsize(dest) == src_size:
        return dest
    print(f"Staging checkpoint to local disk (one-time): {path} -> {dest}")
    _copy_with_progress(path, dest)
    return dest


def resolve_paths(output_dir: str, ckpt: str = None, cache: str = None):
    here = os.path.dirname(__file__)
    output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(here, output_dir)
    ckpt_path = ckpt or os.path.join(output_dir, "checkpoints", "best.pt")
    # Default cache lives on the Linux native fs (/tmp), not /mnt/* (NTFS via
    # WSL2's 9p bridge) — reading a ~200MB npz off /mnt/* can take minutes;
    # off /tmp it's near-instant. Same trick data.py uses for the split cache.
    cache_path = cache or os.path.join(
        "/tmp/retrieval_demo_cache", os.path.basename(output_dir.rstrip("/")) + "_test_embeddings.npz"
    )
    return output_dir, ckpt_path, cache_path


def get_device_dtype(config: Config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, config.dtype) if device.type == "cuda" else torch.float32
    return device, dtype


def load_model(config: Config, ckpt_path: str, device: torch.device):
    model = DualEncoder(config).to(device)
    load_checkpoint(_fast_local_copy(ckpt_path), model)
    model.eval()
    return model


def build_test_data(config: Config):
    all_data = load_index(config)
    _, _, test_ids = compute_split(
        all_data,
        train_frac=config.train_frac,
        val_frac=config.val_frac,
        test_frac=config.test_frac,
        rare_threshold=config.rare_subtype_threshold,
        seed=config.split_seed,
    )
    test_set = set(test_ids)
    return [d for d in all_data if d["image_id"] in test_set]


def build_index(embs: np.ndarray) -> faiss.Index:
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    return idx


@torch.no_grad()
def encode_text_query(model, config, text, device, dtype):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
    enc = tokenizer([text], max_length=config.max_text_len, padding="max_length",
                     truncation=True, return_tensors="pt")
    ii, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        emb = model.encode_text(ii, am)
    return emb.cpu().float().numpy()


def _load_image(config: Config, image_path: str) -> Image.Image:
    path = os.path.join(config.data_root, image_path)
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=0)


def plot_text_to_image(query_text, query_emb, config, img_index, test_data, top_k, out_path,
                        gt_image_path=None, title_prefix="text -> image"):
    """Query embedding search over the image index; saves a PDF figure with
    each retrieved panel labeled rank/similarity, correct match outlined green.

    gt_image_path (not gt_image_id) is used to mark correctness: image_id is
    per-figure, not per-panel — a figure can have multiple sub-panels sharing
    one image_id, so matching on image_id could mark a sibling panel "correct"."""
    scores, ids = img_index.search(query_emb.astype(np.float32), top_k)

    print(f"\nQuery (text): {query_text!r}")
    fig, axes = plt.subplots(1, top_k, figsize=(3 * top_k, 3.8))
    if top_k == 1:
        axes = [axes]
    for rank, (i, s) in enumerate(zip(ids[0], scores[0])):
        item = test_data[i]
        im = _load_image(config, item["image_path"])
        is_correct = gt_image_path is not None and item["image_path"] == gt_image_path
        axes[rank].imshow(im)
        axes[rank].axis("off")
        mark = "  ✓ correct" if is_correct else ("  ✗" if gt_image_path is not None else "")
        color = "green" if is_correct else ("red" if (gt_image_path is not None and rank == 0) else "black")
        axes[rank].set_title(f"#{rank+1}  sim={s:.3f}{mark}\n{item['visualization_subtype']}",
                              fontsize=9, color=color)
        print(f"  #{rank+1}  sim={s:.3f}  subtype={item['visualization_subtype']:<15} "
              f"image_id={item['image_id']}  path={item['image_path']}{mark}")
    plt.suptitle(f"{title_prefix}  |  query: {query_text}", fontsize=10, wrap=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_image_to_text(query_item, query_emb, config, txt_index, test_data, top_k, out_path,
                        gt_index=None, title_prefix="image -> text"):
    """Query embedding search over the text index; saves a PDF figure with
    the query image on the left and ranked captions on the right, correct one in green."""
    scores, ids = txt_index.search(query_emb.astype(np.float32), top_k)

    print(f"\nQuery (image): {query_item['image_id']} ({query_item['image_path']})")
    print(f"  ground-truth caption: {str(query_item['summary'])[:150]}...")

    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})
    im = _load_image(config, query_item["image_path"])
    ax_img.imshow(im)
    ax_img.axis("off")
    ax_img.set_title(f"query image\n{query_item['image_id']}  ({query_item['visualization_subtype']})",
                      fontsize=9)

    ax_txt.axis("off")
    y = 1.0
    line_h = 1.0 / (top_k + 1)
    for rank, (i, s) in enumerate(zip(ids[0], scores[0])):
        item = test_data[i]
        is_correct = gt_index is not None and i == gt_index
        color = "green" if is_correct else "black"
        mark = " ✓" if is_correct else ""
        caption = str(item["summary"])[:120]
        ax_txt.text(0, y, f"#{rank+1}  sim={s:.3f}{mark}  [{item['visualization_subtype']}]",
                    fontsize=8.5, weight="bold", color=color, va="top", transform=ax_txt.transAxes)
        ax_txt.text(0, y - 0.4 * line_h, caption + "...", fontsize=7.5, color=color, va="top",
                    wrap=True, transform=ax_txt.transAxes)
        y -= line_h
        mark_print = " <-- correct" if is_correct else ""
        print(f"  #{rank+1}  sim={s:.3f}  subtype={item['visualization_subtype']:<15} "
              f"caption={caption}...{mark_print}")

    plt.suptitle(title_prefix, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  saved -> {out_path}")
