"""
Step 2: run qualitative retrieval examples against embeddings cached by
build_embeddings.py — instant, no re-encoding of the test split. Figures are
saved as PDF (vector, publication-ready) with the ground-truth match marked
green when known.

image -> text queries use only the cached embeddings (no model needed).
text -> image queries (free-form text, or --n_examples) need the checkpoint
loaded, to encode the query text itself.

Usage:
    # N random test panels, both directions, ground truth marked
    python -m retrieval.run_examples --output_dir outputs_no_hardneg_20ep --n_examples 5

    # free-form text query -> top images (no ground truth to mark)
    python -m retrieval.run_examples --output_dir outputs_no_hardneg_20ep \
        --text "dark field TEM image showing dislocations"

    # a specific test panel's image -> top captions (no model load needed)
    python -m retrieval.run_examples --output_dir outputs_no_hardneg_20ep \
        --image_id ni_alloy_img0

    # keep typing queries in one session
    python -m retrieval.run_examples --output_dir outputs_no_hardneg_20ep --interactive
"""
import argparse
import os
import random

import numpy as np
from tqdm import tqdm

from .config import Config
from .demo_common import (
    build_index, encode_text_query, get_device_dtype, load_model,
    plot_image_to_text, plot_text_to_image, resolve_paths,
)


def load_cache(cache_path):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cached embeddings at {cache_path}.\n"
            f"Run this first: python -m retrieval.build_embeddings --output_dir <output_dir>"
        )
    d = np.load(cache_path, allow_pickle=True)
    # Materialize each array once — repeatedly indexing d["field"] inside a loop
    # re-reads and re-decompresses that array from the npz zip archive every
    # single call, turning an O(n) loop into an O(n * fields) disk operation.
    image_id = d["image_id"]
    image_path = d["image_path"]
    summary = d["summary"]
    subcaption = d["subcaption"]
    subtype = d["subtype"]
    category = d["category"]
    img_embs = d["img_embs"]
    txt_embs = d["txt_embs"]

    test_data = [
        {
            "image_id": image_id[i],
            "image_path": image_path[i],
            "summary": summary[i],
            "subcaption": subcaption[i],
            "visualization_subtype": subtype[i],
            "visualization_category": category[i],
        }
        for i in tqdm(range(len(image_id)), desc="Loading cached metadata")
    ]
    return test_data, img_embs, txt_embs


def find_index(test_data, image_id):
    idx = next((i for i, d in enumerate(test_data) if d["image_id"] == image_id), None)
    if idx is None:
        raise ValueError(f"image_id {image_id!r} not found in test split")
    return idx


def find_index_by_path(test_data, image_path):
    """image_id is per-figure, not per-panel — a figure can have multiple
    sub-panels (_A, _B, _C…) sharing one image_id. image_path is the unique key."""
    idx = next((i for i, d in enumerate(test_data) if d["image_path"] == image_path), None)
    if idx is None:
        raise ValueError(f"image_path {image_path!r} not found in test split")
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="outputs_no_hardneg_20ep")
    p.add_argument("--ckpt", type=str, default=None, help="override checkpoint path")
    p.add_argument("--cache", type=str, default=None, help="override cache path")
    p.add_argument("--text", type=str, default=None, help="free-form text query -> top images")
    p.add_argument("--image_id", type=str, default=None, help="test panel image_id -> top texts")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_examples", type=int, default=0,
                    help="sample N random test panels, run both directions for each")
    p.add_argument("--out_dir", type=str, default="demo_outputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interactive", action="store_true",
                    help="prompt for repeated queries; 'img:<image_id>' for image->text, "
                         "anything else for text->image, 'quit' to stop")
    args = p.parse_args()

    output_dir, ckpt_path, cache_path = resolve_paths(args.output_dir, args.ckpt, args.cache)
    here = os.path.dirname(__file__)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(here, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading cached embeddings: {cache_path}")
    test_data, img_embs, txt_embs = load_cache(cache_path)
    print(f"  {len(test_data):,} panels")
    img_index = build_index(img_embs)
    txt_index = build_index(txt_embs)

    config = Config()
    needs_model = bool(args.text) or args.n_examples > 0 or args.interactive
    model = device = dtype = None
    if needs_model:
        device, dtype = get_device_dtype(config)
        print(f"Loading checkpoint (needed to encode text queries): {ckpt_path}")
        model = load_model(config, ckpt_path, device)

    random.seed(args.seed)

    if args.text:
        q = encode_text_query(model, config, args.text, device, dtype)
        plot_text_to_image(args.text, q, config, img_index, test_data, args.top_k,
                            os.path.join(out_dir, "text_to_image.pdf"))

    if args.image_id:
        idx = find_index(test_data, args.image_id)
        q = img_embs[idx:idx + 1]
        plot_image_to_text(test_data[idx], q, config, txt_index, test_data, args.top_k,
                            os.path.join(out_dir, f"{args.image_id}_image_to_text.pdf"), gt_index=idx)

    if args.n_examples > 0 and not args.text and not args.image_id:
        sample = random.sample(test_data, args.n_examples)
        for i, item in enumerate(sample):
            idx = find_index(test_data, item["image_id"])
            query_text = item["summary"] or item["subcaption"]
            q_txt = encode_text_query(model, config, query_text, device, dtype)
            plot_text_to_image(query_text, q_txt, config, img_index, test_data, args.top_k,
                                os.path.join(out_dir, f"example{i}_text_to_image.pdf"),
                                gt_image_path=item["image_path"])

            q_img = img_embs[idx:idx + 1]
            plot_image_to_text(item, q_img, config, txt_index, test_data, args.top_k,
                                os.path.join(out_dir, f"example{i}_image_to_text.pdf"), gt_index=idx)

    if args.interactive:
        print("\nInteractive mode. Enter a text query for text->image, "
              "'img:<image_id>' for image->text, or 'quit'.")
        counter = 0
        while True:
            try:
                query = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("quit", "exit"):
                break
            if query.startswith("img:"):
                image_id = query[len("img:"):].strip()
                try:
                    idx = find_index(test_data, image_id)
                    q = img_embs[idx:idx + 1]
                    plot_image_to_text(test_data[idx], q, config, txt_index, test_data, args.top_k,
                                        os.path.join(out_dir, f"{image_id}_image_to_text.pdf"), gt_index=idx)
                except ValueError as e:
                    print(f"  {e}")
            else:
                counter += 1
                q = encode_text_query(model, config, query, device, dtype)
                plot_text_to_image(query, q, config, img_index, test_data, args.top_k,
                                    os.path.join(out_dir, f"query{counter}_text_to_image.pdf"))

    if not args.text and not args.image_id and args.n_examples == 0 and not args.interactive:
        print("\nNothing to do — pass --text, --image_id, --n_examples N, or --interactive. See --help.")


if __name__ == "__main__":
    main()
