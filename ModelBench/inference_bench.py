#!/usr/bin/env python3
"""
Inference-speed benchmark — measures latency, FPS, peak VRAM, params and
model size for every trained model.  Run on your own GPU to find the best
accuracy / speed tradeoff for your hardware.

Outputs:
  inference_results.csv   — full table with all metrics
  (also prints a sorted table to stdout)

Usage:
    python inference_bench.py
    python inference_bench.py --only yolov9e yolo12m dabdetr
    python inference_bench.py --n-images 100 --warmup 10
    python inference_bench.py --device 0
"""

import os
import csv
import time
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

NUM_CLASSES = 22
IMAGE_EXTS  = (".jpg", ".jpeg", ".png")

CSV_FIELDS = ["Model", "Variant", "Params(M)", "Size(MB)", "Latency(ms)", "FPS", "VRAM(MB)"]


# ── helpers ────────────────────────────────────────────────────────────────────

def collect_images(test_dir: str, n: int):
    images_dir = os.path.join(test_dir, "images")
    paths = []
    for f in sorted(os.listdir(images_dir)):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
            paths.append(os.path.join(images_dir, f))
        if len(paths) >= n:
            break
    return paths


def model_file_size_mb(path: str) -> float:
    """Sum of all weight files under a path (handles .pt, .safetensors, .bin)."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1e6
    total = 0
    for ext in ("*.pt", "*.safetensors", "*.bin"):
        for f in p.rglob(ext):
            total += f.stat().st_size
    return total / 1e6


def count_params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6



def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ── YOLO benchmark ─────────────────────────────────────────────────────────────

def bench_yolo(ckpt: str, images: list, device_id: str, warmup: int):
    from ultralytics import YOLO
    model = YOLO(ckpt)
    params_m  = count_params_m(model.model)
    size_mb   = model_file_size_mb(ckpt)
    device    = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # warmup
    for p in images[:warmup]:
        model.predict(p, device=device_id, verbose=False)

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    sync(device)
    t0 = time.perf_counter()
    for p in images:
        model.predict(p, device=device_id, verbose=False)
    sync(device)
    elapsed = time.perf_counter() - t0

    vram_mb  = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else -1.0
    latency  = elapsed / len(images) * 1000
    fps      = len(images) / elapsed
    return params_m, size_mb, latency, fps, vram_mb


# ── HF benchmark ───────────────────────────────────────────────────────────────

def bench_hf(ckpt: str, images: list, device_id: str, warmup: int, conf: float = 0.3, iou: float = 0.5):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    device    = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(ckpt)
    model     = AutoModelForObjectDetection.from_pretrained(ckpt).to(device)
    model.eval()

    params_m = count_params_m(model)
    size_mb  = model_file_size_mb(ckpt)

    def run_one(img_path):
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        enc  = processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            outs = model(pixel_values=enc["pixel_values"].to(device))
        res = processor.post_process_object_detection(outs, threshold=conf, target_sizes=[(H, W)])[0]
        pb = res["boxes"].cpu(); ps = res["scores"].cpu(); pl = res["labels"].cpu()
        if pb.numel() > 0:
            from torchvision.ops import batched_nms
            batched_nms(pb, ps, pl, iou)

    # warmup
    for p in images[:warmup]:
        run_one(p)

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    sync(device)
    t0 = time.perf_counter()
    for p in images:
        run_one(p)
    sync(device)
    elapsed = time.perf_counter() - t0

    vram_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else -1.0
    latency = elapsed / len(images) * 1000
    fps     = len(images) / elapsed
    return params_m, size_mb, latency, fps, vram_mb


# ── table print ────────────────────────────────────────────────────────────────

def print_table(rows):
    rows = sorted(rows, key=lambda r: r["FPS"], reverse=True)
    hdr = (f"{'Model':<14} {'Variant':<12} {'Params':>8} {'Size':>8} "
           f"{'Latency':>10} {'FPS':>8} {'VRAM':>9}")
    sub = (f"{'':14} {'':12} {'(M)':>8} {'(MB)':>8} "
           f"{'(ms/img)':>10} {'':>8} {'(MB)':>9}")
    sep = "-" * len(hdr)
    print(f"\n{'─'*len(hdr)}")
    print(f"  INFERENCE BENCHMARK  —  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'─'*len(hdr)}")
    print(hdr); print(sub); print(sep)
    for r in rows:
        def f(k, fmt=".1f"):
            v = r.get(k, -1)
            return f"{v:{fmt}}" if isinstance(v, (int, float)) and v >= 0 else "  n/a"
        print(f"{r['Model']:<14} {r['Variant']:<12} "
              f"{f('Params(M)'):>8} {f('Size(MB)'):>8} "
              f"{f('Latency(ms)'):>10} {f('FPS','.1f'):>8} "
              f"{f('VRAM(MB)','.0f'):>9}")
    print(sep)
    print("  Sorted by FPS (highest = fastest)\n")


# ── main ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("Inference speed benchmark")
    p.add_argument("--test-dir",      default="../MatDetect/mydata_all/test")
    p.add_argument("--runs-dir",      default="./runs/detect/runs_mydata")
    p.add_argument("--ckpt-dir",      default="./checkpoints_mydata")
    p.add_argument("--out",           default="./inference_results.csv")
    p.add_argument("--device",        default="0")
    p.add_argument("--n-images",      type=int, default=200,
                   help="Number of test images to time (default: 200)")
    p.add_argument("--warmup",        type=int, default=10,
                   help="Warmup passes before timing (default: 10)")
    p.add_argument("--conf",          type=float, default=0.3,
                   help="Confidence threshold for HF post-process (default: 0.3)")
    p.add_argument("--iou",           type=float, default=0.5,
                   help="NMS IoU threshold for HF models (default: 0.5)")
    p.add_argument("--only",          nargs="+", default=None)
    return p.parse_args()


def main():
    args   = get_args()
    images = collect_images(args.test_dir, args.n_images)
    if not images:
        print(f"No images found in {args.test_dir}/images"); return
    print(f"Benchmarking on {len(images)} images  (warmup={args.warmup})\n")

    from models_registry_mydata import MODELS_MYDATA
    queue = MODELS_MYDATA if not args.only else \
            [m for m in MODELS_MYDATA if m["name"] in args.only]

    rows   = []
    failed = []

    for m in queue:
        name   = m["name"]
        family = m["family"]
        print(f"[bench]  {name}  ({m['variant']})", flush=True)

        # resolve checkpoint
        if m.get("eval_only"):
            ckpt = m["model_id"]
        elif family == "yolo":
            ckpt = os.path.join(args.runs_dir, name, "weights", "best.pt")
        else:
            best_txt = os.path.join(args.ckpt_dir, name, "best_path.txt")
            ckpt = None
            if os.path.exists(best_txt):
                with open(best_txt) as f:
                    ckpt = f.read().strip()

        if ckpt is None or not os.path.exists(ckpt):
            print(f"  SKIP — checkpoint not found: {ckpt}")
            failed.append(name); continue

        try:
            if family == "yolo":
                params_m, size_mb, lat, fps, vram = bench_yolo(
                    ckpt, images, args.device, args.warmup)
            else:
                params_m, size_mb, lat, fps, vram = bench_hf(
                    ckpt, images, args.device, args.warmup, args.conf, args.iou)

            print(f"  {lat:.1f} ms/img  {fps:.1f} FPS  {vram:.0f} MB VRAM  {params_m:.1f}M params")

            rows.append({
                "Model":       name,
                "Variant":     m["variant"],
                "Params(M)":   round(params_m, 2),
                "Size(MB)":    round(size_mb,  1),
                "Latency(ms)": round(lat,      2),
                "FPS":         round(fps,       1),
                "VRAM(MB)":    round(vram,      0),
            })

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED — {e}")
            failed.append(name)

    print_table(rows)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["FPS"], reverse=True):
            w.writerow(r)
    print(f"Results saved → {args.out}")

    if failed:
        print(f"Failed / skipped: {failed}")


if __name__ == "__main__":
    main()
