"""
Creates train / val / test splits from linked_dataset.csv.

Split is done by image_id so sub-panels of the same figure
never appear across splits (prevents data leakage).

Output
------
splits/train.csv   ~80 % of image_ids
splits/val.csv     ~10 %
splits/test.csv    ~10 %
splits/split_stats.json
"""

import json
import os
import random

import pandas as pd

CSV_PATH   = "linked_dataset.csv"
SPLITS_DIR = "splits"
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 1 - TRAIN - VAL  (remainder)
SEED = 42


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # ── keep only clean matched pairs ─────────────────────────────────────────
    df = df[df["matched"] == True].copy()
    df = df[df["variant"].isna() | (df["variant"] == "")]
    df = df[df["subcaption"].notna() & (df["subcaption"].str.strip() != "")]
    df = df.reset_index(drop=True)
    print(f"Clean matched pairs : {len(df)}")

    # ── split by image_id ─────────────────────────────────────────────────────
    image_ids = sorted(df["image_id"].unique().tolist())
    random.seed(SEED)
    random.shuffle(image_ids)

    n       = len(image_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_ids = set(image_ids[:n_train])
    val_ids   = set(image_ids[n_train : n_train + n_val])
    test_ids  = set(image_ids[n_train + n_val :])

    train_df = df[df["image_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["image_id"].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df["image_id"].isin(test_ids)].reset_index(drop=True)

    # ── save splits ───────────────────────────────────────────────────────────
    train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(SPLITS_DIR, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(SPLITS_DIR, "test.csv"),  index=False)

    stats = {
        "seed": SEED,
        "total_pairs": len(df),
        "total_image_ids": n,
        "train": {"image_ids": len(train_ids), "pairs": len(train_df)},
        "val":   {"image_ids": len(val_ids),   "pairs": len(val_df)},
        "test":  {"image_ids": len(test_ids),  "pairs": len(test_df)},
    }
    with open(os.path.join(SPLITS_DIR, "split_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nSplit summary (seed={SEED})")
    print(f"  train : {len(train_ids):>5} image_ids  |  {len(train_df):>6} pairs")
    print(f"  val   : {len(val_ids):>5} image_ids  |  {len(val_df):>6} pairs")
    print(f"  test  : {len(test_ids):>5} image_ids  |  {len(test_df):>6} pairs")
    print(f"\nSaved to {SPLITS_DIR}/")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
