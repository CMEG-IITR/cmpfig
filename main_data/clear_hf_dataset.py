#!/usr/bin/env python3
"""
Delete all Parquet shards from HuggingFace repo so upload can be restarted clean.
Also clears the local upload log.

Usage:
    python clear_hf_dataset.py
"""

import os
from huggingface_hub import HfApi

REPO_ID  = "subham2507/MatSciFig"
LOG_PATH = "./upload_log.txt"


def main():
    api = HfApi()

    print(f"Fetching file list from: {REPO_ID}")
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]

    if not parquet_files:
        print("No Parquet files found — nothing to delete.")
        return

    print(f"Found {len(parquet_files)} Parquet file(s) to delete.")
    confirm = input("Type YES to confirm deletion: ").strip()
    if confirm != "YES":
        print("Aborted.")
        return

    for path in parquet_files:
        api.delete_file(
            path_in_repo=path,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Remove {path}",
        )
        print(f"  deleted: {path}")

    # clear local log so upload restarts from shard 0
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
        print(f"\nCleared local log: {LOG_PATH}")

    print("\nDone — re-run upload_dataset.py to start fresh.")


if __name__ == "__main__":
    main()
