#!/usr/bin/env python3
"""Generate associative-memory fact data and (optionally) upload to GCS.

Each example is the fixed 256-token sequence::

    <bos>  v[0:126]  r  u[0:126]  <eos>  <pad>

where ``v`` and ``u`` are independently sampled 126-token strings from a large
entity pool. A parallel ``label_mask`` stream marks **only the 126-token ``u``
string as supervised** so CE / NLL is computed solely on predicting ``u``.

On-disk format (matches DCLM / NumpyFSLDataset). Layout uses
``{split}/`` for ids and ``{split}_label_mask/`` for masks so the
launch_jolmo cache resolver (last two GCS path components) stays unique::

    train/00000.npy              raw uint32 memmap (no .npy header)
    train_label_mask/00000.npy   raw bool memmap, same length
    val/...
    val_label_mask/...
    metadata.json

Special IDs (dolma2 / OLMo-2 Instruct; taken out of the entity pool)::

    bos = 100256
    rel = 100255
    eos = 100257   (official dolma2 EOS)
    pad = 100277

Example (generate ~1.5B tokens = 6M facts, then upload)::

    python scripts/generate_associative_facts.py \\
        --out-dir /scratch/catheri4/associative_facts_v3 \\
        --num-facts 6000000 \\
        --entity-vocab-size 50000 \\
        --upload gs://cmu-gpucloud-catheri4/datasets/associative_facts_v3
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# Dolma2 / OLMo-2 Instruct specials (see prepare_tinystories.py / TokenizerConfig.dolma2).
EOS_ID = 100257
PAD_ID = 100277
BOS_ID = 100256
REL_ID = 100255

ENTITY_LEN = 126
# <bos> v[126] r u[126] <eos> = 255 content tokens; +1 pad → 256.
CONTENT_LEN = 1 + ENTITY_LEN + 1 + ENTITY_LEN + 1  # 255
FACT_LEN = 256
PAD_LEN = FACT_LEN - CONTENT_LEN  # 1
assert PAD_LEN >= 0
# Positions within each fact.
BOS_POS = 0
V_START, V_END = 1, 1 + ENTITY_LEN
REL_POS = V_END
U_START, U_END = REL_POS + 1, REL_POS + 1 + ENTITY_LEN
EOS_POS = U_END
PAD_START = EOS_POS + 1
TOKENS_PER_FACT = FACT_LEN


def _upload(local_dir: Path, gcs_uri: str) -> None:
    gcs_uri = gcs_uri.rstrip("/")
    if not gcs_uri.startswith("gs://"):
        gcs_uri = f"gs://{gcs_uri}"
    cmd = ["gsutil", "-m", "rsync", "-r", str(local_dir), gcs_uri]
    print(f"[upload] {' '.join(cmd)}", file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True)
    print(f"[upload] done → {gcs_uri}", file=sys.stderr)


def _write_split(
    out_dir: Path,
    split: str,
    num_facts: int,
    entity_vocab_size: int,
    shard_facts: int,
    seed: int,
) -> dict:
    """Write packed facts + label masks for one split. Returns shard metadata."""
    assert entity_vocab_size > 0
    specials = {BOS_ID, REL_ID, EOS_ID, PAD_ID}
    if entity_vocab_size > min(specials):
        raise ValueError(
            f"entity_vocab_size={entity_vocab_size} overlaps reserved specials "
            f"{sorted(specials)}; keep entity_vocab_size <= {min(specials)}"
        )

    ids_dir = out_dir / split
    mask_dir = out_dir / f"{split}_label_mask"
    ids_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    n_shards = max(1, math.ceil(num_facts / shard_facts))
    rng = np.random.default_rng(seed)
    shards = []
    facts_remaining = num_facts

    # Template mask: only u[126] supervised.
    base_mask = np.zeros(FACT_LEN, dtype=np.bool_)
    base_mask[U_START:U_END] = True

    for shard_i in range(n_shards):
        n = min(shard_facts, facts_remaining)
        facts_remaining -= n
        n_tokens = n * FACT_LEN

        ids_path = ids_dir / f"{shard_i:05d}.npy"
        mask_path = mask_dir / f"{shard_i:05d}.npy"
        print(
            f"[{split}] shard {shard_i:05d}: {n:,} facts ({n_tokens:,} tokens) → {ids_path}",
            file=sys.stderr, flush=True,
        )

        ids_mm = np.memmap(ids_path, mode="w+", dtype=np.uint32, shape=(n_tokens,))
        mask_mm = np.memmap(mask_path, mode="w+", dtype=np.bool_, shape=(n_tokens,))

        block = min(n, max(1, 4_000_000 // FACT_LEN))
        written = 0
        while written < n:
            b = min(block, n - written)
            docs = np.empty((b, FACT_LEN), dtype=np.uint32)
            docs[:, BOS_POS] = BOS_ID
            docs[:, V_START:V_END] = rng.integers(
                0, entity_vocab_size, size=(b, ENTITY_LEN), dtype=np.uint32
            )
            docs[:, REL_POS] = REL_ID
            docs[:, U_START:U_END] = rng.integers(
                0, entity_vocab_size, size=(b, ENTITY_LEN), dtype=np.uint32
            )
            docs[:, EOS_POS] = EOS_ID
            if PAD_LEN:
                docs[:, PAD_START:] = PAD_ID
            masks = np.broadcast_to(base_mask, (b, FACT_LEN)).copy()

            start = written * FACT_LEN
            end = start + b * FACT_LEN
            ids_mm[start:end] = docs.ravel()
            mask_mm[start:end] = masks.ravel()
            written += b

        ids_mm.flush()
        mask_mm.flush()
        del ids_mm, mask_mm

        shards.append({
            "shard": shard_i,
            "num_facts": n,
            "num_tokens": n_tokens,
            "input_ids": str(ids_path.relative_to(out_dir)),
            "label_mask": str(mask_path.relative_to(out_dir)),
        })

    return {
        "split": split,
        "num_facts": num_facts,
        "num_tokens": num_facts * FACT_LEN,
        "num_shards": n_shards,
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--num-facts", type=int, default=6_000_000,
        help="Number of training facts (default 6M ≈ 1.536B tokens)",
    )
    parser.add_argument("--num-val-facts", type=int, default=8_192)
    parser.add_argument(
        "--entity-vocab-size", type=int, default=50_000,
        help="Size of the entity pool for u and v (exclusive of specials)",
    )
    parser.add_argument(
        "--shard-facts", type=int, default=1_000_000,
        help="Facts per shard (default 1M → 256M tokens/shard)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--upload", type=str, default=None,
        help="If set, gsutil rsync --out-dir to this gs:// prefix after writing",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating associative facts → {out_dir}\n"
        f"  train={args.num_facts:,} facts  val={args.num_val_facts:,}\n"
        f"  entity_vocab={args.entity_vocab_size:,}  entity_len={ENTITY_LEN}\n"
        f"  specials bos={BOS_ID} rel={REL_ID} eos={EOS_ID} pad={PAD_ID}\n"
        f"  layout: <bos> v[{ENTITY_LEN}] r u[{ENTITY_LEN}] <eos> "
        f"+ {PAD_LEN} pad  (loss only on u)",
        file=sys.stderr,
    )

    train_meta = _write_split(
        out_dir, "train", args.num_facts, args.entity_vocab_size,
        args.shard_facts, args.seed,
    )
    val_meta = _write_split(
        out_dir, "val", args.num_val_facts, args.entity_vocab_size,
        max(args.num_val_facts, 1), args.seed + 1,
    )

    metadata = {
        "format": "associative_facts_v3",
        "sequence": f"<bos> v[0:{ENTITY_LEN}] r u[0:{ENTITY_LEN}] <eos> + {PAD_LEN} pad",
        "fact_len": FACT_LEN,
        "entity_len": ENTITY_LEN,
        "content_len": CONTENT_LEN,
        "pad_len": PAD_LEN,
        "dtype_ids": "uint32",
        "dtype_mask": "bool",
        "bos_id": BOS_ID,
        "rel_id": REL_ID,
        "eos_id": EOS_ID,
        "pad_id": PAD_ID,
        "entity_vocab_size": args.entity_vocab_size,
        "label_mask": f"True only on u positions [{U_START}:{U_END})",
        "tokenizer": "allenai/OLMo-2-0425-1B-Instruct (dolma2 ids)",
        "seed": args.seed,
        "train": train_meta,
        "val": val_meta,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {meta_path}", file=sys.stderr)

    if args.upload:
        _upload(out_dir, args.upload)
        metadata["gcs_uri"] = args.upload.rstrip("/")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(
        f"Done. train_tokens={train_meta['num_tokens']:,}  "
        f"val_tokens={val_meta['num_tokens']:,}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
