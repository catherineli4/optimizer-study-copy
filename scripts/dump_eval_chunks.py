#!/usr/bin/env python3
"""Write the first N fixed-size eval chunks as decoded text.

Example:
    python3 scripts/dump_eval_chunks.py \\
        --dataset DCLM_heldout \\
        --dataset-cache /scratch/catheri4/cache/datasets \\
        --out /tmp/dclm_first20.txt

    python3 scripts/dump_eval_chunks.py \\
        --dataset C4_val \\
        --dataset-cache /scratch/catheri4/cache/datasets \\
        --out /tmp/c4_first20.txt
"""

from __future__ import annotations

import argparse
import os
import sys

from transformers import AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.divergence_datasets import (  # noqa: E402
    DEFAULT_DATASET_CACHE,
    DEFAULT_TOKENIZER,
    resolve_dataset_spec,
    read_first_chunks,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["DCLM_heldout", "C4_val"],
        help="Validation dataset to dump.",
    )
    ap.add_argument(
        "--dataset-cache",
        default=DEFAULT_DATASET_CACHE,
        help="Local dataset cache dir.",
    )
    ap.add_argument("--num-chunks", type=int, default=20)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", required=True, help="Output .txt path.")
    ap.add_argument("--memmap-path", default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--max-instances", type=int, default=None)
    args = ap.parse_args()

    spec = resolve_dataset_spec(
        args.dataset,
        args.dataset_cache,
        chunk_size=args.chunk_size,
        max_instances=args.max_instances,
        memmap_path=args.memmap_path,
    )
    for path in spec.paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Memmap not found: {path}")

    chunks = read_first_chunks(spec, args.num_chunks)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"dataset: {spec.name}\n")
        f.write(f"chunk_size: {spec.chunk_size}\n")
        f.write(f"num_chunks: {len(chunks)}\n")
        f.write(f"memmap: {spec.paths[0]}\n\n")
        for i, ids in enumerate(chunks):
            f.write(f"===== chunk {i} ({len(ids)} tokens) =====\n")
            f.write(tokenizer.decode(ids, skip_special_tokens=False))
            f.write("\n\n")

    print(f"wrote {args.out} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
