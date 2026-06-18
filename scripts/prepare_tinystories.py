"""Pre-tokenize or fetch pretraining data, pack as .npy memmaps, and upload to GCS.

Replaces the live HF streaming path in deep_memorization.py, which stalls at
startup (5-10 min per job on tokenizer + shard warmup).

Supported datasets:
  - tinystories : HF roneneldan/TinyStories tokenized with GPT-2 BPE (uint16)
  - dclm        : existing pre-tokenized Dolma2 shards mirrored from the shared
                  bucket (uint32). No re-tokenization.

Default upload destination: gs://cmu-gpucloud-catheri4/datasets/<dataset>/

Usage:
    # TinyStories (tokenize from HF, upload)
    python scripts/prepare_tinystories.py /scratch/catheri4/cache/tinystories_gpt2 \\
        --dataset tinystories

    # DCLM (copy shards from shared bucket to local + your bucket)
    python scripts/prepare_tinystories.py /scratch/catheri4/cache/dclm \\
        --dataset dclm
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np


log = logging.getLogger("prepare_data")


# ---------------------------------------------------------------------------
# DCLM source (matches launch_jolmo/pretraining_matrix.py:27,133-137)
# ---------------------------------------------------------------------------

DCLM_SOURCE_PREFIX = (
    "gs://cmu-gpucloud-jspringe/shared/datasets/OLMo/dclm/train/"
    "preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train"
    "_allenai_dolma2-tokenizer_part-000"
)
DCLM_SHARD_COUNT = 5
DCLM_TOKENIZER = "allenai/OLMo-2-0425-1B-Instruct"
DCLM_VOCAB_SIZE = 100278
DCLM_EOS = 100257
DCLM_DTYPE = "uint32"


# ---------------------------------------------------------------------------
# Generic GCS upload (matches new_utils/tokenize_data.py:25-36)
# ---------------------------------------------------------------------------

def upload_to_gcs(local_dir: Path, gcs_path: str, dataset_name: str) -> None:
    if not gcs_path.startswith("gs://"):
        gcs_path = f"gs://{gcs_path}"
    gcs_path = f"{gcs_path.rstrip('/')}/{dataset_name}"
    cmd = ["gsutil", "-m", "cp", "-r", f"{str(local_dir)}/*", gcs_path]
    log.info(f"Uploading to {gcs_path}  ({' '.join(cmd)})")
    subprocess.run(cmd, check=True)
    log.info(f"Upload successful to: {gcs_path}")


def gsutil_cp(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gsutil", "-m", "cp", src, str(dst)]
    log.info(" ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# TinyStories: tokenize from HF
# ---------------------------------------------------------------------------

def prepare_tinystories(output_dir: Path, tokenizer_id: str,
                        max_train_tokens: int | None,
                        max_val_tokens: int | None,
                        skip_if_exists: bool) -> dict:
    from datasets import load_dataset
    from rich.progress import track
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_id)
    eot = tokenizer.eos_token_id

    def tokenize_split(split: str, out_path: Path, max_tokens: int | None) -> int:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if skip_if_exists and out_path.exists():
            n = np.memmap(str(out_path), dtype=np.uint16, mode="r").shape[0]
            log.info(f"[{split}] exists → {n:,} tokens, skipping")
            return n
        ds = load_dataset("roneneldan/TinyStories", split=split)
        log.info(f"[{split}] {len(ds):,} documents")
        chunks = []
        total = 0
        for row in track(ds, description=f"tokenize [{split}]"):
            ids = tokenizer.encode(row["text"])
            ids.append(eot)
            chunks.append(np.asarray(ids, dtype=np.uint16))
            total += len(ids)
            if max_tokens is not None and total >= max_tokens:
                break
        arr = np.concatenate(chunks)
        if max_tokens is not None and arr.shape[0] > max_tokens:
            arr = arr[:max_tokens]
        log.info(f"[{split}] writing {arr.shape[0]:,} tokens → {out_path}")
        mm = np.memmap(str(out_path), dtype=np.uint16, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        return int(arr.shape[0])

    n_train = tokenize_split("train", output_dir / "train" / "input_ids.npy", max_train_tokens)
    n_val = tokenize_split("validation", output_dir / "val" / "input_ids.npy", max_val_tokens)

    return {
        "dataset": "tinystories",
        "source": "roneneldan/TinyStories",
        "tokenizer": tokenizer_id,
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "dtype": "uint16",
        "train_tokens": n_train,
        "val_tokens": n_val,
        "layout": "train/input_ids.npy  val/input_ids.npy  (contiguous uint16, EOS between docs)",
    }


# ---------------------------------------------------------------------------
# DCLM: mirror existing pre-tokenized shards from shared bucket
# ---------------------------------------------------------------------------

def prepare_dclm(output_dir: Path, skip_if_exists: bool) -> dict:
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    shard_names = [f"{i:05d}.npy" for i in range(DCLM_SHARD_COUNT)]
    total_tokens = 0
    for name in shard_names:
        local = train_dir / name
        if skip_if_exists and local.exists():
            log.info(f"[dclm] {local} exists, skipping download")
        else:
            gsutil_cp(f"{DCLM_SOURCE_PREFIX}/{name}", local)
        # Introspect shape via memmap header (uint32 packed ids, 1-D array).
        n = np.memmap(str(local), dtype=np.uint32, mode="r").shape[0]
        log.info(f"[dclm] {name}: {n:,} tokens")
        total_tokens += n

    return {
        "dataset": "dclm",
        "source": DCLM_SOURCE_PREFIX,
        "tokenizer": DCLM_TOKENIZER,
        "vocab_size": DCLM_VOCAB_SIZE,
        "eos_token_id": DCLM_EOS,
        "dtype": DCLM_DTYPE,
        "train_tokens": total_tokens,
        "val_tokens": 0,
        "shard_count": DCLM_SHARD_COUNT,
        "layout": "train/{00000..%05d}.npy (uint32 packed ids, dolma2-tokenizer)" % (DCLM_SHARD_COUNT - 1),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=str,
                    help="Local directory to write dataset into.")
    ap.add_argument("--dataset", choices=["tinystories", "dclm"], default="tinystories",
                    help="Which dataset to prepare.")
    ap.add_argument("--tokenizer", default="gpt2",
                    help="HF tokenizer id for tinystories (ignored for dclm).")
    ap.add_argument("--max-train-tokens", type=int, default=None,
                    help="Cap train tokens (tinystories only).")
    ap.add_argument("--max-val-tokens", type=int, default=20_000_000,
                    help="Cap val tokens (tinystories only).")
    ap.add_argument("-g", "--gcs-bucket", default="gs://cmu-gpucloud-catheri4/datasets",
                    help="GCS destination prefix. Pass empty string to skip upload.")
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="Skip work if output files already exist.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "tinystories":
        meta = prepare_tinystories(
            output_dir, args.tokenizer,
            args.max_train_tokens, args.max_val_tokens,
            args.skip_if_exists,
        )
    elif args.dataset == "dclm":
        meta = prepare_dclm(output_dir, args.skip_if_exists)
    else:
        raise ValueError(args.dataset)

    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"meta.json: {meta}")

    if args.gcs_bucket:
        upload_to_gcs(output_dir, args.gcs_bucket, args.dataset)


if __name__ == "__main__":
    main()
