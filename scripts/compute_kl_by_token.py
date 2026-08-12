#!/usr/bin/env python3
"""Aggregate per-token-type mean KL from a divergence .npz + memmap walk.

Output .npz per model with arrays sorted by descending token frequency:
  token_id, count, mean_kl, std_kl

Example:
    python3 scripts/compute_kl_by_token.py \\
        --npz ../DivergenceEvaluation/foo-divergence.npz \\
        --dataset-cache /scratch/catheri4/cache/datasets \\
        --out results/kl_by_token/foo.npz
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.divergence_datasets import (  # noqa: E402
    DEFAULT_DATASET_CACHE,
    METRIC_KEYS,
    DatasetSpec,
    resolve_dataset_spec,
)
from scripts.divergence_eval import iter_batches_memmap  # noqa: E402

_REF_TAG_RE = re.compile(r"-vs-(?P<ref>.+)-divergence\.npz$")
_DIVERGENCE_NPZ_RE = re.compile(r"^(?P<run>.+?)(?:-vs-[^/]+)?-divergence\.npz$")


def _dataset_label_from_npz(npz_path: str) -> str:
    with np.load(npz_path) as data:
        labels = [k for k in data.files if k not in METRIC_KEYS]
    if len(labels) != 1:
        raise ValueError(f"{npz_path}: expected one dataset label, got {labels}")
    return labels[0]


def _load_metric(npz_path: str, key: str, dataset: str) -> np.ndarray:
    with np.load(npz_path) as data:
        arr = data[dataset] if dataset in data.files else data[key]
    return np.asarray(arr, dtype=np.float32).ravel()


def aggregate_kl_by_token(
    kl: np.ndarray,
    spec: DatasetSpec,
    *,
    batch_size: int = 8,
    max_token_id: int = 100_352,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (token_id, count, mean_kl, std_kl) sorted by descending count."""
    kl_sum = np.zeros(max_token_id, dtype=np.float64)
    kl_sum_sq = np.zeros(max_token_id, dtype=np.float64)
    counts = np.zeros(max_token_id, dtype=np.int64)
    offset = 0

    for batch in iter_batches_memmap(
        spec.paths, spec.chunk_size, batch_size, spec.max_instances
    ):
        bsz, seq_len = batch.shape
        n_positions = bsz * (seq_len - 1)
        if offset + n_positions > kl.size:
            raise ValueError(
                f"KL shorter than memmap walk: need {offset + n_positions}, have {kl.size}"
            )
        kl_batch = kl[offset : offset + n_positions].reshape(bsz, seq_len - 1)
        offset += n_positions

        for row in range(bsz):
            token_ids = batch[row, 1:].astype(np.int64)
            kl_row = kl_batch[row].astype(np.float64)
            np.add.at(kl_sum, token_ids, kl_row)
            np.add.at(kl_sum_sq, token_ids, kl_row * kl_row)
            np.add.at(counts, token_ids, 1)

    if offset != kl.size:
        raise ValueError(f"KL longer than memmap walk: consumed {offset}, have {kl.size}")

    present = counts > 0
    token_ids = np.flatnonzero(present)
    order = np.argsort(-counts[token_ids], kind="stable")
    token_ids = token_ids[order]
    tok_counts = counts[token_ids]
    mean_kl = kl_sum[token_ids] / tok_counts
    var = np.maximum(kl_sum_sq[token_ids] / tok_counts - mean_kl * mean_kl, 0.0)
    std_kl = np.sqrt(var)
    return (
        token_ids.astype(np.int32),
        tok_counts.astype(np.int64),
        mean_kl.astype(np.float32),
        std_kl.astype(np.float32),
    )


def process_npz(
    npz_path: str,
    out_path: str,
    *,
    dataset_cache: str,
    metric_key: str = "kl_forward",
    chunk_size: Optional[int] = None,
    max_instances: Optional[int] = None,
    memmap_path: Optional[str] = None,
    batch_size: int = 8,
) -> str:
    dataset_label = _dataset_label_from_npz(npz_path)
    kl = _load_metric(npz_path, metric_key, dataset_label)
    spec = resolve_dataset_spec(
        dataset_label,
        dataset_cache,
        chunk_size=chunk_size,
        max_instances=max_instances,
        memmap_path=memmap_path,
    )
    for path in spec.paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Memmap not found: {path}")

    run_name = _DIVERGENCE_NPZ_RE.match(os.path.basename(npz_path)).group("run")
    print(f"[kl-by-token] {run_name}: aggregating {kl.size:,} positions", flush=True)
    token_ids, counts, mean_kl, std_kl = aggregate_kl_by_token(
        kl, spec, batch_size=batch_size
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        token_id=token_ids,
        count=counts,
        mean_kl=mean_kl,
        std_kl=std_kl,
        metric_key=np.array(metric_key),
        dataset=np.array(dataset_label),
        run_name=np.array(run_name),
    )
    summary = {
        "run_name": run_name,
        "dataset": dataset_label,
        "metric_key": metric_key,
        "n_token_types": int(token_ids.size),
        "n_positions": int(kl.size),
    }
    with open(os.path.splitext(out_path)[0] + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[kl-by-token] wrote {out_path} ({token_ids.size} token types)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-cache", default=DEFAULT_DATASET_CACHE)
    ap.add_argument("--key", default="kl_forward", choices=["kl_forward", "kl_reverse"])
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--memmap-path", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    process_npz(
        args.npz,
        args.out,
        dataset_cache=args.dataset_cache,
        metric_key=args.key,
        chunk_size=args.chunk_size,
        max_instances=args.max_instances,
        memmap_path=args.memmap_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
