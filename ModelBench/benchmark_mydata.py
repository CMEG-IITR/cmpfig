#!/usr/bin/env python3
"""
Retrain all registry models on mydata_all (train+val) and save metrics
to separate output directories — does not touch original benchmark outputs.

Usage:
    python benchmark_mydata.py
    python benchmark_mydata.py --only yolov9e yolo11m yolo12m
    python benchmark_mydata.py --skip-done
    python benchmark_mydata.py --dry-run
"""

import os
import sys
import argparse
import subprocess

TRAIN_DIR = "../MatDetect/all_mydata/train"
VAL_DIR   = "../MatDetect/all_mydata/val"
DATA_YAML = "./data_mydata.yaml"


def get_args():
    p = argparse.ArgumentParser("ModelBench — mydata_all retraining")
    p.add_argument("--only",        nargs="+", default=None,
                   help="Run only these model names")
    p.add_argument("--skip-done",   action="store_true",
                   help="Skip models whose results JSON already exists")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch-yolo",  type=int, default=16)
    p.add_argument("--batch-hf",    type=int, default=4)
    p.add_argument("--device",      default="0")
    p.add_argument("--results-dir", default="./results_mydata")
    p.add_argument("--runs-dir",    default="./runs_mydata")
    p.add_argument("--ckpt-dir",    default="./checkpoints_mydata")
    p.add_argument("--dry-run",     action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.runs_dir,    exist_ok=True)
    os.makedirs(args.ckpt_dir,    exist_ok=True)

    from models_registry_mydata import MODELS_MYDATA as MODELS

    queue = list(MODELS)
    if args.only:
        queue = [m for m in queue if m["name"] in args.only]
    if args.skip_done:
        queue = [m for m in queue
                 if not os.path.exists(os.path.join(args.results_dir, f"{m['name']}.json"))]

    print(f"\nModels to fine-tune on mydata_all (from best checkpoints): {len(queue)}")
    for m in queue:
        print(f"  [{m['family'].upper():4s}]  {m['variant']:25s}  ({m['model_id']})")
    print()

    if args.dry_run:
        print("[dry-run] nothing executed.")
        return

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    failed    = []

    for m in queue:
        print(f"\n{'#'*60}")
        print(f"  Starting: {m['variant']}  ({m['name']})")
        print(f"{'#'*60}")

        if m["family"] == "yolo":
            cmd = [
                sys.executable, os.path.join(bench_dir, "trainers", "train_yolo.py"),
                "--model",       m["model_id"],
                "--name",        m["name"],
                "--data",        os.path.join(bench_dir, DATA_YAML),
                "--epochs",      str(args.epochs),
                "--batch",       str(args.batch_yolo),
                "--device",      args.device,
                "--results-dir", args.results_dir,
                "--runs-dir",    args.runs_dir,
            ]
        else:
            cmd = [
                sys.executable, os.path.join(bench_dir, "trainers", "train_hf.py"),
                "--model",       m["model_id"],
                "--name",        m["name"],
                "--train-dir",   os.path.join(bench_dir, TRAIN_DIR),
                "--val-dir",     os.path.join(bench_dir, VAL_DIR),
                "--epochs",      str(args.epochs),
                "--batch-size",  str(args.batch_hf),
                "--cuda-device", f"cuda:{args.device}",
                "--results-dir", args.results_dir,
                "--ckpt-dir",    args.ckpt_dir,
            ]

        ret = subprocess.run(cmd, cwd=bench_dir)
        if ret.returncode != 0:
            print(f"[FAILED] {m['name']} exited with code {ret.returncode}")
            failed.append(m["name"])

    print(f"\n{'='*60}")
    done = [m["name"] for m in queue if m["name"] not in failed]
    print(f"  Completed   : {len(done)}")
    print(f"  Failed      : {len(failed)}  {failed if failed else ''}")
    print(f"  Results     : {args.results_dir}/")
    print(f"  YOLO runs   : {args.runs_dir}/")
    print(f"  HF checkpts : {args.ckpt_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
