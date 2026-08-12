"""Shared validation-dataset paths/settings for divergence analysis scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_DCLM_NPY_GS = (
    "gs://cmu-gpucloud-jspringe/shared/datasets/OLMo/dclm/train/"
    "preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer"
)
_DIVERSITY_V2_GS = "gs://cmu-gpucloud-jspringe/outputs/diversity.pretraining.v2"
DCLM_HELDOUT_INSTANCES = 8192
DEFAULT_DATASET_CACHE = "/scratch/catheri4/cache/datasets"
DEFAULT_TOKENIZER = "allenai/OLMo-2-0425-1B-Instruct"

DATASET_URIS: Dict[str, str] = {
    "DCLM_heldout": f"{_DCLM_NPY_GS}_part-059/00004.npy",
    "C4_val": f"{_DIVERSITY_V2_GS}/ValidationDataset/Tokenized/C4_val.bin",
    "Reddit_val": f"{_DIVERSITY_V2_GS}/ValidationDataset/Tokenized/Reddit_val.bin",
    "Wiki_val": f"{_DIVERSITY_V2_GS}/ValidationDataset/Tokenized/Wiki_val.bin",
    "Books_val": f"{_DIVERSITY_V2_GS}/ValidationDataset/Tokenized/Books_val.bin",
}

DATASET_DEFAULTS: Dict[str, Tuple[int, Optional[int]]] = {
    "DCLM_heldout": (4096, DCLM_HELDOUT_INSTANCES),
    "C4_val": (1024, None),
    "Reddit_val": (1024, None),
    "Wiki_val": (1024, None),
    "Books_val": (1024, None),
}

METRIC_KEYS = ("kl_forward", "kl_reverse", "jsd")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    paths: List[str]
    chunk_size: int
    max_instances: Optional[int] = None


def resolve_global_chunk_path(uri: str, dataset_cache: str) -> str:
    """Local cache path for a gs:// chunk (matches launch_jolmo/training.py)."""
    gs_parent = uri.rsplit("/", 1)[0]
    components = gs_parent.replace("gs://", "").split("/")
    parent_name = os.path.join(*components[-2:]) if len(components) >= 2 else components[-1]
    return os.path.join(dataset_cache, parent_name, os.path.basename(uri))


def resolve_dataset_spec(
    label: str,
    dataset_cache: str,
    *,
    chunk_size: Optional[int] = None,
    max_instances: Optional[int] = None,
    memmap_path: Optional[str] = None,
) -> DatasetSpec:
    if memmap_path is not None:
        if label not in DATASET_DEFAULTS:
            default_cs, default_max = 4096, None
        else:
            default_cs, default_max = DATASET_DEFAULTS[label]
        return DatasetSpec(
            name=label,
            paths=[memmap_path],
            chunk_size=chunk_size or default_cs,
            max_instances=max_instances if max_instances is not None else default_max,
        )

    if label not in DATASET_URIS:
        raise ValueError(
            f"Unknown dataset label {label!r}; known: {sorted(DATASET_URIS)}. "
            "Pass --memmap-path and --chunk-size to override."
        )
    default_cs, default_max = DATASET_DEFAULTS[label]
    return DatasetSpec(
        name=label,
        paths=[resolve_global_chunk_path(DATASET_URIS[label], dataset_cache)],
        chunk_size=chunk_size or default_cs,
        max_instances=max_instances if max_instances is not None else default_max,
    )


def read_first_chunks(
    spec: DatasetSpec,
    n_chunks: int,
) -> List[List[int]]:
    """Return the first ``n_chunks`` fixed-size token sequences from ``spec``."""
    out: List[List[int]] = []
    for path in spec.paths:
        arr = np.memmap(path, mode="r", dtype=np.uint32)
        n_full = (arr.shape[0] // spec.chunk_size) * spec.chunk_size
        limit = n_chunks if spec.max_instances is None else min(n_chunks, spec.max_instances)
        for start in range(0, n_full, spec.chunk_size):
            if len(out) >= limit:
                return out
            out.append(arr[start : start + spec.chunk_size].astype(int).tolist())
    return out
