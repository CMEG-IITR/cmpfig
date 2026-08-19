"""
Qualitative i->t retrieval success figure for the paper.

Loads the trained dual-encoder checkpoint, reproduces the deterministic test
split, filters to the Phase/Equilibrium Diagram category (the strongest
category, R@1 = 23.1%), runs REAL inference (CLIP ViT-B/32 image encoder +
MatSciBERT text encoder -> linear projection -> LayerNorm -> L2-normalize) to
encode every image and text in that category slice, builds a FAISS
IndexFlatIP over the text embeddings (cosine similarity via inner product on
L2-normalized vectors), and for each query image retrieves its top-5 nearest
captions. It picks the first query whose own ground-truth caption is ranked
#1 (a clean success case), falling back to the best-ranked candidate seen if
no rank-1 hit turns up in the first N candidates checked.

No cached embeddings are read here -- every embedding is produced by a live
forward pass through the loaded checkpoint, so the figure reflects a real,
reproducible retrieval run rather than a precomputed/replayed result.

Usage:
    python -m retrieval.retrieval_success_example
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import faiss
from PIL import Image

from .config import Config
from .demo_common import build_test_data, get_device_dtype, load_model

# ── Configuration ──────────────────────────────────────────────────────────
CKPT_PATH = "/mnt/d/Subham/Compoun_img_01/retrieval/outputs_no_hardneg_20ep/checkpoints/best.pt"
DATA_ROOT = "/mnt/d/Subham/Compoun_img_01/main_data"          # panel crops live under here
OUTPUT_PDF = "/mnt/d/Subham/Compoun_img_01/retrieval/paper_figures/retrieval_success_example.pdf"

TARGET_CATEGORY = "Phase/Equilibrium Diagram"
TOP_K = 5
MAX_QUERY_CANDIDATES = 260   # cap on how many queries to probe before falling back
BATCH_SIZE = 64


# ── Inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_category(model, config, items, device, dtype, batch_size):
    """Real forward pass: image encoder + text encoder -> projected, L2-normalized embeddings."""
    from transformers import AutoTokenizer, CLIPImageProcessor
    image_processor = CLIPImageProcessor.from_pretrained(config.image_encoder)
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)

    img_embs, txt_embs = [], []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]

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


def build_faiss_text_index(txt_embs: np.ndarray) -> faiss.Index:
    """Exact inner-product search; embeddings are already L2-normalized by the
    model's forward pass, so inner product == cosine similarity."""
    index = faiss.IndexFlatIP(txt_embs.shape[1])
    index.add(txt_embs.astype(np.float32))
    return index


# ── Figure ─────────────────────────────────────────────────────────────────

def load_panel_image(config, image_path):
    try:
        return Image.open(os.path.join(config.data_root, image_path)).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=(30, 30, 30))


def wrap_words(text, max_words=18):
    words = str(text).split()
    truncated = " ".join(words[:max_words])
    if len(words) > max_words:
        truncated += "..."
    return "\n".join(textwrap.wrap(truncated, width=26))


def build_figure(config, query_item, retrieved_items, ranks_correct_idx, out_path):
    fig, axes = plt.subplots(1, TOP_K + 1, figsize=(3.0 * (TOP_K + 1), 4.2))

    # panel 0: query image
    ax_img = axes[0]
    ax_img.imshow(load_panel_image(config, query_item["image_path"]))
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax_img.set_title("Query (Image)", fontsize=10.5, weight="bold")
    ax_img.set_xlabel(f"[{query_item['visualization_subtype']}]", fontsize=8.5)

    # panels 1..K: retrieved text cards
    for rank, item in enumerate(retrieved_items):
        ax = axes[rank + 1]
        ax.set_xticks([]); ax.set_yticks([])
        is_correct = (rank == ranks_correct_idx)
        text = item["summary"] or item["subcaption"] or ""
        ax.text(0.5, 0.55, wrap_words(text), fontsize=8.3, ha="center", va="center",
                 transform=ax.transAxes, wrap=True)
        title_color = "green" if is_correct else "black"
        ax.set_title(f"Rank {rank + 1}", fontsize=10.5, weight="bold", color=title_color)

        border_color = "green" if is_correct else "lightgray"
        border_width = 3.0 if is_correct else 1.0
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(border_width)

        if is_correct:
            ax.set_xlabel("✓ Correct Match", fontsize=9, color="green", weight="bold")

    fig.suptitle(f"Image → Text retrieval  |  category: {TARGET_CATEGORY}", fontsize=12, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.savefig(os.path.splitext(out_path)[0] + ".png", format="png", dpi=200)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    config = Config()
    config.data_root = DATA_ROOT
    device, dtype = get_device_dtype(config)
    print(f"Device: {device}  dtype: {dtype}  (cuda available: {torch.cuda.is_available()})")

    print(f"Loading checkpoint: {CKPT_PATH}")
    model = load_model(config, CKPT_PATH, device)

    print("Reproducing deterministic test split ...")
    test_data = build_test_data(config)
    print(f"  full test split: {len(test_data):,} panels")

    category_items = [d for d in test_data if d["visualization_category"] == TARGET_CATEGORY]
    print(f"  {TARGET_CATEGORY}: {len(category_items):,} panels")
    if len(category_items) < 2:
        raise RuntimeError(f"Not enough panels in category {TARGET_CATEGORY!r} to run retrieval.")

    print(f"Encoding {len(category_items)} panels (real forward pass, batch_size={BATCH_SIZE}) ...")
    img_embs, txt_embs = encode_category(model, config, category_items, device, dtype, BATCH_SIZE)

    print("Building FAISS IndexFlatIP over text embeddings ...")
    text_index = build_faiss_text_index(txt_embs)

    n_probe = min(MAX_QUERY_CANDIDATES, len(category_items))
    best_rank, best_i, best_retrieved_ids = None, None, None
    success_i, success_retrieved_ids = None, None

    print(f"Probing up to {n_probe} query candidates for a rank-1 success case ...")
    for i in range(n_probe):
        q = img_embs[i:i + 1].astype(np.float32)
        scores, ids = text_index.search(q, TOP_K)
        ids = ids[0]

        if ids[0] == i:
            success_i, success_retrieved_ids = i, ids
            break

        pos = np.where(ids == i)[0]
        rank = int(pos[0]) + 1 if len(pos) > 0 else None  # None => outside top-K
        if rank is not None and (best_rank is None or rank < best_rank):
            best_rank, best_i, best_retrieved_ids = rank, i, ids

    if success_i is not None:
        qi, retrieved_ids, gt_rank_idx = success_i, success_retrieved_ids, 0
        print(f"Found clean rank-1 success: query index {qi} "
              f"(image_id={category_items[qi]['image_id']})")
    else:
        if best_i is None:
            raise RuntimeError(
                f"No candidate among the first {n_probe} had its ground-truth caption "
                f"within top-{TOP_K}; widen MAX_QUERY_CANDIDATES or TOP_K."
            )
        qi, retrieved_ids, gt_rank_idx = best_i, best_retrieved_ids, best_rank - 1
        print(f"No rank-1 hit in first {n_probe} candidates; using best available: "
              f"rank {best_rank} (query index {qi}, image_id={category_items[qi]['image_id']})")

    query_item = category_items[qi]
    retrieved_items = [category_items[j] for j in retrieved_ids]

    print(f"Query image: {query_item['image_path']}")
    for r, item in enumerate(retrieved_items):
        mark = "  <-- ground truth" if r == gt_rank_idx else ""
        cap = (item["summary"] or item["subcaption"] or "")[:100]
        print(f"  Rank {r+1}: {cap}{mark}")

    build_figure(config, query_item, retrieved_items, gt_rank_idx, OUTPUT_PDF)
    print(f"Saved -> {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
