"""
Training loop: one epoch, validation step, checkpoint management.
"""
import logging
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from .config import Config
from .eval import evaluate, print_results
from .loss import infonce_loss
from .model import DualEncoder, build_param_groups

log = logging.getLogger(__name__)


# ── Optimiser + Scheduler ─────────────────────────────────────────────────────

def build_optimiser_and_scheduler(
    model: DualEncoder,
    config: Config,
    total_steps: int,
):
    param_groups = build_param_groups(model, config)
    optimizer = AdamW(param_groups, eps=config.adam_eps)

    warmup_steps = max(1, round(config.warmup_fraction * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: DualEncoder,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    best_val_recall: float,
    config: Config,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_recall": best_val_recall,
        },
        path,
    )


def load_checkpoint(path: str, model: DualEncoder, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("best_val_recall", 0.0)


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: DualEncoder,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
    config: Config,
    epoch: int,
    start_step: int,
    val_loader: Optional[DataLoader],
    best_val_recall: float,
    ckpt_dir: str,
    eval_results_dir: str,
) -> Dict:
    model.train()
    total_loss = 0.0
    n_batches = 0
    step = start_step

    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        pv = batch["pixel_values"].to(device, non_blocking=True)
        ii = batch["input_ids"].to(device, non_blocking=True)
        am = batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=dtype):
            img_emb, txt_emb, tau = model(pv, ii, am)
            loss = infonce_loss(img_emb, txt_emb, tau)
            if config.grad_accum_steps > 1:
                loss = loss / config.grad_accum_steps

        loss.backward()
        total_loss += loss.item() * (config.grad_accum_steps if config.grad_accum_steps > 1 else 1)
        n_batches += 1

        if (batch_idx + 1) % config.grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                avg_loss = total_loss / n_batches
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    tau=f"{model.tau.item():.4f}",
                    lr=f"{lr:.2e}",
                )
                log.info(
                    f"epoch={epoch} step={step} loss={avg_loss:.4f} "
                    f"tau={model.tau.item():.4f} lr={lr:.2e} elapsed={elapsed:.0f}s"
                )

            # Periodic eval on val set
            if val_loader is not None and step % config.eval_every == 0:
                val_results = evaluate(
                    model, val_loader, device, dtype,
                    k_values=config.recall_k,
                    min_category_samples=config.min_category_test_samples,
                    output_path=os.path.join(eval_results_dir, f"val_step{step}.json"),
                )
                print_results(val_results, label=f"Val step {step}")
                ov = val_results["overall"]
                val_rsum = sum(ov.get(k, 0.0) for k in (
                    "i2t_R@1", "i2t_R@5", "i2t_R@10",
                    "t2i_R@1", "t2i_R@5", "t2i_R@10",
                ))

                if val_rsum > best_val_recall:
                    best_val_recall = val_rsum
                    save_checkpoint(
                        os.path.join(ckpt_dir, "best.pt"),
                        model, optimizer, scheduler,
                        epoch, step, best_val_recall, config,
                    )
                    log.info(f"New best val RSUM = {best_val_recall:.4f} → saved best.pt")

                model.train()

            # Periodic checkpoint
            if step % config.save_every == 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"step_{step}.pt"),
                    model, optimizer, scheduler,
                    epoch, step, best_val_recall, config,
                )

    return {
        "epoch": epoch,
        "step": step,
        "avg_loss": total_loss / max(n_batches, 1),
        "best_val_recall": best_val_recall,
    }
