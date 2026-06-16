"""
Fine-tune CLIP on compound scientific figure image-caption pairs.

Training strategy
-----------------
- Symmetric InfoNCE (contrastive) loss over in-batch negatives
- Full model fine-tuning with low LR
- Linear warmup + cosine LR decay
- Mixed precision (bf16 if available, else fp16)
- Gradient accumulation for large effective batch size
- Val R@1 (image→caption) used for best-checkpoint selection
- Training log saved to checkpoints/train_log.csv

Usage
-----
    python train.py                          # defaults
    python train.py --epochs 10 --batch-size 128 --accum-steps 2
    python train.py --wandb                  # enable W&B logging
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup

from dataset import PanelDataset

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL       = "openai/clip-vit-base-patch32"
DEFAULT_SPLITS_DIR  = "splits"
DEFAULT_IMAGES_DIR  = "crops_allloys_img_0.55"
DEFAULT_CKPT_DIR    = "checkpoints"
DEFAULT_EPOCHS      = 10
DEFAULT_BATCH       = 64          # per-GPU batch size
DEFAULT_ACCUM       = 2           # gradient accumulation → effective BS = 128
DEFAULT_LR          = 2e-5
DEFAULT_WEIGHT_DECAY= 1e-2
DEFAULT_WARMUP_RATIO= 0.05        # 5 % of total steps
DEFAULT_VAL_EVERY   = 1           # validate every N epochs
DEFAULT_SEED        = 42


# ── loss ──────────────────────────────────────────────────────────────────────
def clip_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor):
    """Symmetric InfoNCE loss. image_emb and text_emb must be L2-normalised."""
    logit_scale = logit_scale.exp().clamp(max=100)
    logits_per_image = logit_scale * image_emb @ text_emb.T   # [B, B]
    logits_per_text  = logits_per_image.T

    targets = torch.arange(image_emb.shape[0], device=image_emb.device)
    loss_i  = F.cross_entropy(logits_per_image, targets)
    loss_t  = F.cross_entropy(logits_per_text,  targets)
    return (loss_i + loss_t) / 2


# ── retrieval metrics (on val) ────────────────────────────────────────────────
@torch.no_grad()
def retrieval_recall(model, dataloader, device, direction="i2t"):
    """Compute R@1/5/10 for image→caption (i2t) or caption→image (t2i)."""
    all_img_emb, all_txt_emb = [], []
    model.eval()
    for batch in tqdm(dataloader, desc="  val encode", leave=False):
        pv  = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        attn= batch["attention_mask"].to(device)
        ie  = F.normalize(model.visual_projection(model.vision_model(pixel_values=pv).pooler_output), dim=-1)
        te  = F.normalize(model.text_projection(model.text_model(input_ids=ids, attention_mask=attn).pooler_output), dim=-1)
        all_img_emb.append(ie.cpu())
        all_txt_emb.append(te.cpu())

    img_emb = torch.cat(all_img_emb)   # [N, D]
    txt_emb = torch.cat(all_txt_emb)

    sim = img_emb @ txt_emb.T          # [N, N]
    if direction == "t2i":
        sim = sim.T

    N = sim.shape[0]
    ranks = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        order = sim[i].argsort(descending=True)
        ranks[i] = (order == i).nonzero(as_tuple=True)[0][0] + 1

    r1  = (ranks <= 1).float().mean().item()  * 100
    r5  = (ranks <= 5).float().mean().item()  * 100
    r10 = (ranks <= 10).float().mean().item() * 100
    mr  = ranks.float().mean().item()
    return {"R@1": round(r1,2), "R@5": round(r5,2), "R@10": round(r10,2), "mean_rank": round(mr,2)}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--splits-dir",   default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--images-dir",   default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--ckpt-dir",     default=DEFAULT_CKPT_DIR)
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",   type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--accum-steps",  type=int,   default=DEFAULT_ACCUM)
    parser.add_argument("--lr",           type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--val-every",    type=int,   default=DEFAULT_VAL_EVERY)
    parser.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    parser.add_argument("--wandb",        action="store_true")
    parser.add_argument("--num-workers",  type=int,   default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"\n=== CLIP Fine-tuning ===")
    print(f"  device  : {device}  |  amp dtype: {dtype}")
    print(f"  model   : {args.model}")
    print(f"  epochs  : {args.epochs}  |  batch: {args.batch_size}  |  accum: {args.accum_steps}")
    print(f"  eff. BS : {args.batch_size * args.accum_steps}")

    # ── W&B (optional) ────────────────────────────────────────────────────────
    if args.wandb:
        import wandb
        wandb.init(project="compound-fig-clip", config=vars(args))

    # ── model + processor ─────────────────────────────────────────────────────
    processor = CLIPProcessor.from_pretrained(args.model)
    model     = CLIPModel.from_pretrained(args.model).to(device)

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = PanelDataset(
        os.path.join(args.splits_dir, "train.csv"), args.images_dir, processor
    )
    val_ds = PanelDataset(
        os.path.join(args.splits_dir, "val.csv"), args.images_dir, processor
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"  train   : {len(train_ds)} pairs  |  val: {len(val_ds)} pairs\n")

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps   = (len(train_loader) // args.accum_steps) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = torch.cuda.amp.GradScaler(enabled=True)

    # ── log file ──────────────────────────────────────────────────────────────
    log_path = os.path.join(args.ckpt_dir, "train_log.csv")
    log_fields = ["epoch", "train_loss", "val_r1_i2t", "val_r5_i2t", "val_r10_i2t",
                  "val_r1_t2i", "val_r5_t2i", "val_r10_t2i", "val_mean_rank_i2t", "lr", "epoch_time_s"]
    with open(log_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=log_fields).writeheader()

    best_val_r1  = -1.0
    best_ckpt    = os.path.join(args.ckpt_dir, "best_model")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in pbar:
            pv   = batch["pixel_values"].to(device)
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast(dtype=dtype):
                img_emb = F.normalize(
                    model.visual_projection(model.vision_model(pixel_values=pv).pooler_output), dim=-1
                )
                txt_emb = F.normalize(
                    model.text_projection(model.text_model(input_ids=ids, attention_mask=attn).pooler_output), dim=-1
                )
                loss = clip_loss(img_emb, txt_emb, model.logit_scale) / args.accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.accum_steps

            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{running_loss/(step+1):.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - t0

        # ── validation ────────────────────────────────────────────────────────
        log_row = {
            "epoch": epoch, "train_loss": round(avg_loss, 4),
            "val_r1_i2t": "", "val_r5_i2t": "", "val_r10_i2t": "",
            "val_r1_t2i": "", "val_r5_t2i": "", "val_r10_t2i": "",
            "val_mean_rank_i2t": "",
            "lr": round(scheduler.get_last_lr()[0], 8),
            "epoch_time_s": round(epoch_time, 1),
        }

        if epoch % args.val_every == 0:
            print(f"\n  [Val] epoch {epoch}")
            i2t = retrieval_recall(model, val_loader, device, "i2t")
            t2i = retrieval_recall(model, val_loader, device, "t2i")
            print(f"  i2t  R@1={i2t['R@1']}%  R@5={i2t['R@5']}%  R@10={i2t['R@10']}%  MR={i2t['mean_rank']}")
            print(f"  t2i  R@1={t2i['R@1']}%  R@5={t2i['R@5']}%  R@10={t2i['R@10']}%  MR={t2i['mean_rank']}")

            log_row.update({
                "val_r1_i2t":   i2t["R@1"],  "val_r5_i2t":  i2t["R@5"],
                "val_r10_i2t":  i2t["R@10"], "val_r1_t2i":  t2i["R@1"],
                "val_r5_t2i":   t2i["R@5"],  "val_r10_t2i": t2i["R@10"],
                "val_mean_rank_i2t": i2t["mean_rank"],
            })

            if args.wandb:
                import wandb
                wandb.log({"epoch": epoch, "train_loss": avg_loss,
                           "val/i2t_R1": i2t["R@1"], "val/i2t_R5": i2t["R@5"],
                           "val/t2i_R1": t2i["R@1"]})

            # ── save best checkpoint by val i2t R@1 ───────────────────────────
            if i2t["R@1"] > best_val_r1:
                best_val_r1 = i2t["R@1"]
                model.save_pretrained(best_ckpt)
                processor.save_pretrained(best_ckpt)
                with open(os.path.join(best_ckpt, "best_info.json"), "w") as fh:
                    json.dump({"epoch": epoch, "val_i2t_R1": best_val_r1,
                               "val_t2i_R1": t2i["R@1"]}, fh, indent=2)
                print(f"  [Saved] best checkpoint  val_i2t_R@1={best_val_r1}%")

            model.train()

        # ── append log row ─────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=log_fields).writerow(log_row)

        print(f"Epoch {epoch}  loss={avg_loss:.4f}  time={epoch_time:.0f}s")

    # ── save final checkpoint ─────────────────────────────────────────────────
    final_ckpt = os.path.join(args.ckpt_dir, "final_model")
    model.save_pretrained(final_ckpt)
    processor.save_pretrained(final_ckpt)
    print(f"\nTraining complete.")
    print(f"  Best val i2t R@1 : {best_val_r1}%  ->  {best_ckpt}")
    print(f"  Final checkpoint  ->  {final_ckpt}")
    print(f"  Train log         ->  {log_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
