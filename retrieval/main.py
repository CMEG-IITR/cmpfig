"""
Entry point for training the cross-modal retrieval model.

Usage:
    python -m retrieval.main [options]

Key options (override Config defaults):
    --output_dir PATH
    --batch_size N
    --num_epochs N
    --lr_encoder FLOAT
    --lr_proj FLOAT
    --resume PATH          # path to a checkpoint to resume from
    --eval_only            # skip training, evaluate best checkpoint
    --train_size N         # override train_target panel count
"""
import argparse
import logging
import os
import random
import sys
import time as _time

import numpy as np
import torch

# Allow running as `python -m retrieval.main` or `python retrieval/main.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.config import Config
from retrieval.data import (
    build_dataloaders,
    compute_split,
    load_index,
    load_split,
    print_split_stats,
    save_split,
    split_exists,
)
from retrieval.eval import evaluate, print_results
from retrieval.model import DualEncoder
from retrieval.train import (
    build_optimiser_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str):
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "logs", "train.log")),
        ],
    )


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str)
    p.add_argument("--data_root", type=str)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--lr_vision", type=float)
    p.add_argument("--lr_text", type=float)
    p.add_argument("--lr_proj", type=float)
    p.add_argument("--embed_dim", type=int)
    p.add_argument("--train_size", type=int, help="Override train_target panel count")
    p.add_argument("--image_encoder", type=str)
    p.add_argument("--text_encoder", type=str)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--hard_neg_ratio", type=float)
    p.add_argument("--grad_accum_steps", type=int)
    p.add_argument("--seed", type=int)
    return p.parse_args()


def apply_args(config: Config, args) -> Config:
    for key, val in vars(args).items():
        if val is not None and hasattr(config, key):
            setattr(config, key, val)
    if args.train_size is not None:
        config.train_target = args.train_size
    return config


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = Config()
    config = apply_args(config, args)

    set_seed(config.seed)
    setup_logging(config.output_dir)
    log = logging.getLogger(__name__)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, config.dtype)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log.info(f"Device: {device} | dtype: {dtype}")
    log.info(f"Output dir: {config.output_dir}")

    # ── Data split ──────────────────────────────────────────────────────────
    all_data = load_index(config)
    log.info(f"Loaded {len(all_data)} panels")

    if split_exists(config.output_dir):
        log.info("Loading existing split from /tmp …")
        t0 = _time.perf_counter()
        saved = load_split(config.output_dir)
        train_ids, val_ids, test_ids = saved["train_ids"], saved["val_ids"], saved["test_ids"]
        log.info(f"  loaded in {_time.perf_counter()-t0:.1f}s")
    else:
        log.info("Computing 80/10/10 figure-level stratified split …")
        t0 = _time.perf_counter()
        train_ids, val_ids, test_ids = compute_split(
            all_data,
            train_frac=config.train_frac,
            val_frac=config.val_frac,
            test_frac=config.test_frac,
            rare_threshold=config.rare_subtype_threshold,
            seed=config.split_seed,
        )
        log.info(f"  compute_split done in {_time.perf_counter()-t0:.1f}s  "
                 f"(train={len(train_ids):,}  val={len(val_ids):,}  test={len(test_ids):,} figures)")
        t0 = _time.perf_counter()
        save_split(config.output_dir, train_ids, val_ids, test_ids)
        log.info(f"  save_split done in {_time.perf_counter()-t0:.1f}s")

    # Single O(N) pass — 3× faster than 3 separate list comprehensions
    log.info("Partitioning panels …")
    t0 = _time.perf_counter()
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)
    train_data, val_data, test_data = [], [], []
    for d in all_data:
        fid = d["image_id"]
        if fid in train_set:
            train_data.append(d)
        elif fid in val_set:
            val_data.append(d)
        elif fid in test_set:
            test_data.append(d)
    log.info(f"  partition done in {_time.perf_counter()-t0:.1f}s")

    print_split_stats(train_data, val_data, test_data)

    # ── Model ───────────────────────────────────────────────────────────────
    log.info("Building model …")
    model = DualEncoder(config).to(device)
    log.info(
        f"  image_encoder: {config.image_encoder}\n"
        f"  text_encoder:  {config.text_encoder}\n"
        f"  embed_dim:     {config.embed_dim}"
    )

    # ── DataLoaders ─────────────────────────────────────────────────────────
    # Build epoch-0 train_loader first so its __len__ (from HardNegativeBatchSampler)
    # gives the true steps_per_epoch — len(train_data)//batch_size undercounts by 2×
    # because the sampler iterates over anchors (batch_size//2), not full batches.
    train_loader, val_loader, test_loader = build_dataloaders(
        config, train_data, val_data, test_data, epoch_seed=0
    )

    ckpt_dir = os.path.join(config.output_dir, "checkpoints")
    eval_dir = os.path.join(config.output_dir, "eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    if args.eval_only:
        best_ckpt = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(best_ckpt):
            load_checkpoint(best_ckpt, model)
            log.info(f"Loaded checkpoint: {best_ckpt}")
        test_results = evaluate(
            model, test_loader, device, dtype,
            k_values=config.recall_k,
            mrr_cutoff=config.mrr_cutoff,
            min_category_samples=config.min_category_test_samples,
            output_path=os.path.join(eval_dir, "test_results.json"),
        )
        print_results(test_results, label="Test")
        return

    # ── Optimiser ───────────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps     = (steps_per_epoch * config.num_epochs) // config.grad_accum_steps

    optimizer, scheduler = build_optimiser_and_scheduler(model, config, total_steps)

    start_epoch     = 0
    global_step     = 0
    best_val_recall = 0.0

    if args.resume and os.path.exists(args.resume):
        start_epoch, global_step, best_val_recall = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        log.info(f"Resumed from {args.resume} (epoch={start_epoch}, step={global_step})")

    # ── Training loop ────────────────────────────────────────────────────────
    log.info(f"Training for {config.num_epochs} epochs (~{total_steps} total steps)")
    for epoch in range(start_epoch, config.num_epochs):
        if epoch > 0:
            train_loader, _, _ = build_dataloaders(
                config, train_data, val_data, test_data, epoch_seed=epoch
            )

        epoch_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
            config=config,
            epoch=epoch,
            start_step=global_step,
            val_loader=val_loader,
            best_val_recall=best_val_recall,
            ckpt_dir=ckpt_dir,
            eval_results_dir=eval_dir,
        )
        global_step     = epoch_stats["step"]
        best_val_recall = epoch_stats["best_val_recall"]
        log.info(
            f"Epoch {epoch} done — avg_loss={epoch_stats['avg_loss']:.4f}, "
            f"best_val_R@100_i2t={best_val_recall:.4f}"
        )

        save_checkpoint(
            os.path.join(ckpt_dir, f"epoch_{epoch}.pt"),
            model, optimizer, scheduler,
            epoch + 1, global_step, best_val_recall, config,
        )

    # ── Final evaluation on test set ─────────────────────────────────────────
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, model)
        log.info("Loaded best checkpoint for test evaluation")

    test_results = evaluate(
        model, test_loader, device, dtype,
        k_values=config.recall_k,
        mrr_cutoff=config.mrr_cutoff,
        min_category_samples=config.min_category_test_samples,
        output_path=os.path.join(eval_dir, "test_results.json"),
    )
    print_results(test_results, label="Test (best ckpt)")
    log.info("Done.")


if __name__ == "__main__":
    main()
