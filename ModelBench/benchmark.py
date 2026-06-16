#!/usr/bin/env python3
"""
Orchestrator — trains every model in models_registry.py sequentially
and saves per-model metrics to ./results/.

Usage:
    # train all models
    python benchmark.py

    # train only specific models by name
    python benchmark.py --only yolo11n yolo11s dabdetr

    # skip already-completed models (results/<name>.json exists)
    python benchmark.py --skip-done

    # dry-run: print what would run without training
    python benchmark.py --dry-run
"""

import os
import sys
import json
import argparse
import subprocess


def get_args():
    p = argparse.ArgumentParser("ModelBench orchestrator")
    p.add_argument("--only",       nargs="+", default=None,
                   help="Run only these model names (from models_registry)")
    p.add_argument("--skip-done",  action="store_true",
                   help="Skip models whose results JSON already exists")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-yolo", type=int, default=16,  help="Batch size for YOLO models")
    p.add_argument("--batch-hf",   type=int, default=4,   help="Batch size for HF models")
    p.add_argument("--device",     default="0")
    p.add_argument("--results-dir",default="./results")
    p.add_argument("--dry-run",    action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)

    from models_registry import MODELS

    queue = MODELS
    if args.only:
        queue = [m for m in queue if m["name"] in args.only]
    if args.skip_done:
        queue = [m for m in queue
                 if not os.path.exists(os.path.join(args.results_dir, f"{m['name']}.json"))]

    print(f"\nModels to train: {len(queue)}")
    for m in queue:
        print(f"  [{m['family'].upper():4s}]  {m['variant']:25s}  ({m['model_id']})")
    print()

    if args.dry_run:
        print("[dry-run] nothing executed.")
        return

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    failed = []

    for m in queue:
        print(f"\n{'#'*60}")
        print(f"  Starting: {m['variant']}  ({m['name']})")
        print(f"{'#'*60}")

        if m["family"] == "yolo":
            cmd = [
                sys.executable, os.path.join(bench_dir, "trainers", "train_yolo.py"),
                "--model",        m["model_id"],
                "--name",         m["name"],
                "--epochs",       str(args.epochs),
                "--batch",        str(args.batch_yolo),
                "--device",       args.device,
                "--results-dir",  args.results_dir,
            ]
        elif m["family"] == "gdino":
            cmd = [
                sys.executable, os.path.join(bench_dir, "trainers", "train_gdino.py"),
                "--model",        m["model_id"],
                "--name",         m["name"],
                "--epochs",       str(args.epochs),
                "--batch-size",   str(args.batch_hf),
                "--cuda-device",  f"cuda:{args.device}",
                "--results-dir",  args.results_dir,
            ]
        else:
            cmd = [
                sys.executable, os.path.join(bench_dir, "trainers", "train_hf.py"),
                "--model",        m["model_id"],
                "--name",         m["name"],
                "--cuda-device",  f"cuda:{args.device}",
                "--results-dir",  args.results_dir,
            ]
            if m.get("eval_only"):
                cmd.append("--eval-only")
            else:
                cmd += ["--epochs", str(args.epochs), "--batch-size", str(args.batch_hf)]

        ret = subprocess.run(cmd, cwd=bench_dir)
        if ret.returncode != 0:
            print(f"[FAILED] {m['name']} exited with code {ret.returncode}")
            failed.append(m["name"])

    print(f"\n{'='*60}")
    done = [m["name"] for m in queue if m["name"] not in failed]
    print(f"  Completed : {len(done)}")
    print(f"  Failed    : {len(failed)}  {failed if failed else ''}")
    print(f"  Results   : {args.results_dir}/")
    print(f"  Run 'python compare.py' to see the benchmark table.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
