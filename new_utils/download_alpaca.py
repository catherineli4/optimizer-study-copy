import os
import sys
import subprocess
from pathlib import Path

from datasets import load_dataset


LOCAL_TRAIN_DIR = "/data/user_data/catheri4/datasets/alpaca"
GCS_TRAIN_DIR = "gs://cmu-gpucloud-catheri4/datasets/alpaca/"
VAL_SPLIT_RATIO = 0.05  # 5% for validation


def ensure_local_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def gsutil_ls(path: str) -> bool:
    result = subprocess.run(
        ["gsutil", "ls", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def ensure_gcs_dir(path: str) -> None:
    if not path.endswith("/"):
        raise ValueError("GCS directory path must end with '/'")
    _exists = gsutil_ls(path)
    # Directory-like prefixes in GCS are virtual; nothing to create.


def upload_to_gcs(local_path: str, gcs_dir: str) -> None:
    subprocess.check_call(["gsutil", "-m", "cp", local_path, gcs_dir])


def save_split(dataset, split_name: str, output_dir: str) -> str:
    """Save a dataset split as JSONL and return its path."""
    path = os.path.join(output_dir, f"{split_name}.jsonl")
    dataset.to_json(path, orient="records", lines=True)
    return path


def main() -> int:
    print("📥 Loading Alpaca dataset from Hugging Face...")
    ds = load_dataset("tatsu-lab/alpaca")

    print(f"ℹ️ Splitting dataset: {100 - VAL_SPLIT_RATIO*100:.1f}% train / {VAL_SPLIT_RATIO*100:.1f}% val")
    ds_split = ds["train"].train_test_split(test_size=VAL_SPLIT_RATIO, seed=42)

    ensure_local_dir(LOCAL_TRAIN_DIR)
    ensure_gcs_dir(GCS_TRAIN_DIR)

    print("💾 Saving dataset splits locally...")
    saved_files = []
    for split_name, dsplit in ds_split.items():
        out_path = save_split(dsplit, split_name, LOCAL_TRAIN_DIR)
        saved_files.append(out_path)
        print(f"  ✅ Saved {split_name} -> {out_path}")

    print("☁️ Uploading files to GCS...")
    for path in saved_files:
        upload_to_gcs(path, GCS_TRAIN_DIR)
        print(f"  ☁️ Uploaded {path} -> {GCS_TRAIN_DIR}")
        os.remove(path)
        print(f"  🧹 Deleted local file: {path}")

    print("🎉 All Alpaca splits uploaded to GCS successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
