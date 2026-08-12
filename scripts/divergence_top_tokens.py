#!/usr/bin/env python3
"""Find the lowest-KL next-token positions for each divergence .npz.

Reads precomputed ``kl_forward`` / ``kl_reverse`` arrays from divergence eval
output and walks the same memmap validation data in lockstep to recover the
predicted token and its preceding context. Writes one text file per model.

Example (single model):
    python3 scripts/divergence_top_tokens.py \\
        --npz ../DivergenceEvaluation/MuonExpt3-...-divergence.npz \\
        --dataset-cache /scratch/catheri4/cache/datasets \\
        --out-dir results/top_kl_tokens

Example (all models in processed results):
    python3 scripts/divergence_top_tokens.py \\
        --from-results colm-moss-latex/results/60M/divergence_results.json \\
        --npz-dir ../DivergenceEvaluation \\
        --dataset-cache /scratch/catheri4/cache/datasets \\
        --reference OLMo-2-1124-7B \\
        --out-dir colm-moss-latex/results/60M/top_kl_tokens
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from transformers import AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def iter_batches_memmap(
    paths: List[str],
    chunk_size: int,
    batch_size: int,
    max_instances: Optional[int] = None,
):
    """Iterate fixed-size chunks from memmap files (same walk as divergence_eval)."""
    batch = []
    emitted = 0
    for p in paths:
        arr = np.memmap(p, mode="r", dtype=np.uint32)
        n_full = (arr.shape[0] // chunk_size) * chunk_size
        for start in range(0, n_full, chunk_size):
            if max_instances is not None and emitted >= max_instances:
                if batch:
                    yield np.stack(batch, axis=0)
                return
            batch.append(np.asarray(arr[start : start + chunk_size]))
            emitted += 1
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0)
                batch = []
    if batch:
        yield np.stack(batch, axis=0)

_METRIC_KEYS = ("kl_forward", "kl_reverse", "jsd")
_REF_TAG_RE = re.compile(r"-vs-(?P<ref>.+)-divergence\.npz$")
_DIVERGENCE_NPZ_RE = re.compile(r"^(?P<run>.+?)(?:-vs-[^/]+)?-divergence\.npz$")
_DEFAULT_TOKENIZER = "allenai/OLMo-2-0425-1B-Instruct"

# Mirror launch_jolmo/pretraining_matrix.py validation chunks (avoid importing it:
# that module touches Project.config at import time and fails off-cluster).
_DCLM_NPY_GS = (
    "gs://cmu-gpucloud-jspringe/shared/datasets/OLMo/dclm/train/"
    "preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer"
)
_DIVERSITY_V2_GS = "gs://cmu-gpucloud-jspringe/outputs/diversity.pretraining.v2"
_DCLM_HELDOUT_INSTANCES = 8192
_DATASET_URIS: Dict[str, str] = {
    "DCLM_heldout": f"{_DCLM_NPY_GS}_part-059/00004.npy",
    "C4_val": f"{_DIVERSITY_V2_GS}/ValidationDataset/Tokenized/C4_val.bin",
    "Reddit_val": f"{_DIVERSITY_V2_GS}/ValidationDataset/Tokenized/Reddit_val.bin",
    "Wiki_val": f"{_DIVERSITY_V2_GS}/ValidationDataset/Tokenized/Wiki_val.bin",
    "Books_val": f"{_DIVERSITY_V2_GS}/ValidationDataset/Tokenized/Books_val.bin",
}
_DATASET_DEFAULTS: Dict[str, Tuple[int, Optional[int]]] = {
    # label -> (chunk_size, max_instances)
    "DCLM_heldout": (4096, _DCLM_HELDOUT_INSTANCES),
    "C4_val": (1024, None),
    "Reddit_val": (1024, None),
    "Wiki_val": (1024, None),
    "Books_val": (1024, None),
}


def _resolve_global_chunk_path(uri: str, dataset_cache: str) -> str:
    """Local cache path for a gs:// chunk (matches launch_jolmo/training.py)."""
    gs_parent = uri.rsplit("/", 1)[0]
    components = gs_parent.replace("gs://", "").split("/")
    parent_name = os.path.join(*components[-2:]) if len(components) >= 2 else components[-1]
    return os.path.join(dataset_cache, parent_name, os.path.basename(uri))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    paths: List[str]
    chunk_size: int
    max_instances: Optional[int] = None


@dataclass(frozen=True)
class TopToken:
    rank: int
    kl: float
    token_id: int
    token_text: str
    context_text: str
    seq_index: int
    token_position: int


def _parse_reference_tag(filename: str, fallback_reference: str) -> str:
    match = _REF_TAG_RE.search(filename)
    return match.group("ref") if match else fallback_reference


def _parse_run_name(filename: str) -> str:
    match = _DIVERGENCE_NPZ_RE.match(filename)
    if not match:
        raise ValueError(f"Could not parse run name from {filename!r}")
    return match.group("run")


def _dataset_label_from_npz(npz_path: str) -> str:
    with np.load(npz_path) as data:
        labels = [k for k in data.files if k not in _METRIC_KEYS]
    if len(labels) != 1:
        raise ValueError(
            f"{npz_path}: expected exactly one dataset label in .npz, got {labels}"
        )
    return labels[0]


def _resolve_dataset_spec(
    label: str,
    dataset_cache: str,
    *,
    chunk_size: Optional[int] = None,
    max_instances: Optional[int] = None,
) -> DatasetSpec:
    if label not in _DATASET_URIS:
        raise ValueError(
            f"Unknown dataset label {label!r}; known: {sorted(_DATASET_URIS)}. "
            "Pass --memmap-path and --chunk-size to override."
        )
    default_cs, default_max = _DATASET_DEFAULTS[label]
    return DatasetSpec(
        name=label,
        paths=[_resolve_global_chunk_path(_DATASET_URIS[label], dataset_cache)],
        chunk_size=chunk_size or default_cs,
        max_instances=max_instances if max_instances is not None else default_max,
    )


def _infer_chunk_size(kl_len: int, max_instances: Optional[int]) -> Optional[int]:
    for chunk_size in (4096, 2048, 1024, 512):
        per_seq = chunk_size - 1
        if per_seq <= 0:
            continue
        if kl_len % per_seq != 0:
            continue
        n_seq = kl_len // per_seq
        if max_instances is not None and n_seq != max_instances:
            continue
        return chunk_size
    return None


def _count_positions(
    paths: Sequence[str],
    chunk_size: int,
    batch_size: int,
    max_instances: Optional[int],
) -> int:
    total = 0
    for batch in iter_batches_memmap(list(paths), chunk_size, batch_size, max_instances):
        total += batch.shape[0] * (batch.shape[1] - 1)
    return total


def _load_metric(npz_path: str, key: str, dataset: str) -> np.ndarray:
    with np.load(npz_path) as data:
        if dataset in data.files:
            arr = data[dataset]
        elif key in data.files:
            arr = data[key]
        else:
            raise KeyError(f"{npz_path}: neither {dataset!r} nor {key!r} found")
    return np.asarray(arr, dtype=np.float32).ravel()


def _iter_positions_with_kl(
    kl: np.ndarray,
    spec: DatasetSpec,
    *,
    batch_size: int = 1,
) -> Iterable[Tuple[float, np.ndarray, int, int, int]]:
    """Yield (kl, seq_ids, seq_index, token_position, flat_index)."""
    offset = 0
    seq_index = 0
    for batch in iter_batches_memmap(
        spec.paths, spec.chunk_size, batch_size, spec.max_instances
    ):
        batch_size_actual, seq_len = batch.shape
        n_positions = batch_size_actual * (seq_len - 1)
        if offset + n_positions > kl.size:
            raise ValueError(
                f"KL array shorter than memmap walk: need {offset + n_positions}, "
                f"have {kl.size} (dataset={spec.name}, chunk_size={spec.chunk_size})"
            )
        kl_batch = kl[offset : offset + n_positions].reshape(batch_size_actual, seq_len - 1)
        offset += n_positions

        for row in range(batch_size_actual):
            seq = batch[row].astype(np.int64)
            for pos in range(1, seq_len):
                yield (
                    float(kl_batch[row, pos - 1]),
                    seq,
                    seq_index,
                    pos,
                    offset - n_positions + row * (seq_len - 1) + (pos - 1),
                )
            seq_index += 1

    if offset != kl.size:
        raise ValueError(
            f"KL array longer than memmap walk: consumed {offset}, have {kl.size}"
        )


def _collect_below_kl(
    items: Iterable[Tuple[float, np.ndarray, int, int, int]],
    max_kl: float,
) -> List[Tuple[float, np.ndarray, int, int, int]]:
    matches: List[Tuple[float, np.ndarray, int, int, int]] = []
    for kl, seq, seq_index, token_position, flat_index in items:
        if kl <= max_kl:
            matches.append((kl, seq.copy(), seq_index, token_position, flat_index))
    matches.sort(key=lambda x: (x[0], x[2], x[3]))
    return matches


def _top_k_smallest(
    items: Iterable[Tuple[float, np.ndarray, int, int, int]],
    k: int,
) -> List[Tuple[float, np.ndarray, int, int, int]]:
    heap: List[Tuple[float, int, Tuple[float, np.ndarray, int, int, int]]] = []
    tie = 0
    for kl, seq, seq_index, token_position, flat_index in items:
        tie += 1
        entry = (kl, seq.copy(), seq_index, token_position, flat_index)
        if len(heap) < k:
            heapq.heappush(heap, (-kl, -tie, entry))
        elif kl < -heap[0][0]:
            heapq.heapreplace(heap, (-kl, -tie, entry))
    # heap entries store -kl; sort ascending by true KL.
    return [entry for _, _, entry in sorted(heap, key=lambda x: (-x[0], -x[1]))]


def _format_context(tokenizer, seq: np.ndarray, token_position: int, context_tokens: int) -> str:
    start = max(0, token_position - context_tokens)
    context_ids = seq[start:token_position].tolist()
    return tokenizer.decode(context_ids, skip_special_tokens=False)


def _write_report(
    out_path: str,
    *,
    run_name: str,
    reference: str,
    metric_key: str,
    dataset: str,
    top_tokens: List[TopToken],
    max_kl: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    metric_desc = {
        "kl_forward": "KL(reference || student)",
        "kl_reverse": "KL(student || reference)",
    }.get(metric_key, metric_key)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"run: {run_name}\n")
        f.write(f"reference: {reference}\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"metric: {metric_key} ({metric_desc})\n")
        if max_kl is not None:
            f.write(f"filter: kl_forward <= {max_kl}\n")
        f.write(f"count: {len(top_tokens)}\n")
        f.write("\n")

        for entry in top_tokens:
            f.write(f"rank: {entry.rank}\n")
            f.write(f"kl: {entry.kl:.8f}\n")
            f.write(f"token_id: {entry.token_id}\n")
            f.write(f"token: {entry.token_text!r}\n")
            f.write(f"seq_index: {entry.seq_index}\n")
            f.write(f"token_position: {entry.token_position}\n")
            f.write("context_preceding:\n")
            f.write(entry.context_text)
            if entry.context_text and not entry.context_text.endswith("\n"):
                f.write("\n")
            f.write("\n---\n\n")


def process_npz(
    npz_path: str,
    out_dir: str,
    *,
    dataset_cache: str,
    metric_key: str = "kl_forward",
    top_k: int = 50,
    context_tokens: int = 64,
    tokenizer_name: str = _DEFAULT_TOKENIZER,
    fallback_reference: str = "OLMo-2-1124-1B",
    chunk_size: Optional[int] = None,
    max_instances: Optional[int] = None,
    memmap_path: Optional[str] = None,
    batch_size: int = 1,
    max_kl: Optional[float] = None,
) -> str:
    filename = os.path.basename(npz_path)
    run_name = _parse_run_name(filename)
    reference = _parse_reference_tag(filename, fallback_reference)
    dataset_label = _dataset_label_from_npz(npz_path)
    kl = _load_metric(npz_path, metric_key, dataset_label)

    if memmap_path is not None:
        spec = DatasetSpec(
            name=dataset_label,
            paths=[memmap_path],
            chunk_size=chunk_size or _infer_chunk_size(kl.size, max_instances) or 4096,
            max_instances=max_instances,
        )
    else:
        spec = _resolve_dataset_spec(
            dataset_label,
            dataset_cache,
            chunk_size=chunk_size,
            max_instances=max_instances,
        )
        if chunk_size is None:
            inferred = _infer_chunk_size(kl.size, spec.max_instances)
            if inferred is not None and inferred != spec.chunk_size:
                spec = DatasetSpec(
                    name=spec.name,
                    paths=spec.paths,
                    chunk_size=inferred,
                    max_instances=spec.max_instances,
                )

    for path in spec.paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Memmap not found: {path}\n"
                f"Download validation data into --dataset-cache ({dataset_cache!r}) first."
            )

    expected = _count_positions(spec.paths, spec.chunk_size, batch_size, spec.max_instances)
    if expected != kl.size:
        raise ValueError(
            f"{filename}: KL length {kl.size} != walked positions {expected} "
            f"(dataset={spec.name}, chunk_size={spec.chunk_size}, "
            f"max_instances={spec.max_instances})"
        )

    print(
        f"[top-tokens] {run_name}: scanning {kl.size:,} positions "
        f"({spec.name}, chunk_size={spec.chunk_size})",
        file=sys.stderr,
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    positions = _iter_positions_with_kl(kl, spec, batch_size=batch_size)
    if max_kl is not None:
        raw_top = _collect_below_kl(positions, max_kl)
        print(
            f"[top-tokens] {run_name}: {len(raw_top):,} positions with KL <= {max_kl}",
            file=sys.stderr,
            flush=True,
        )
    else:
        raw_top = _top_k_smallest(positions, top_k)

    top_tokens: List[TopToken] = []
    for rank, (kl_value, seq, seq_index, token_position, _) in enumerate(raw_top, start=1):
        token_id = int(seq[token_position])
        top_tokens.append(
            TopToken(
                rank=rank,
                kl=kl_value,
                token_id=token_id,
                token_text=tokenizer.decode([token_id], skip_special_tokens=False),
                context_text=_format_context(tokenizer, seq, token_position, context_tokens),
                seq_index=seq_index,
                token_position=token_position,
            )
        )

    if max_kl is not None:
        kl_tag = f"{max_kl:g}".replace(".", "p")
        out_name = f"{run_name}-{metric_key}-le-{kl_tag}.txt"
    else:
        out_name = f"{run_name}-top{top_k}-{metric_key}.txt"
    out_path = os.path.join(out_dir, out_name)
    _write_report(
        out_path,
        run_name=run_name,
        reference=reference,
        metric_key=metric_key,
        dataset=spec.name,
        top_tokens=top_tokens,
        max_kl=max_kl,
    )
    print(f"[top-tokens] wrote {out_path}", file=sys.stderr)
    return out_path


def _records_from_results(
    results_path: str,
    *,
    reference: Optional[str],
    chinchilla: Optional[Union[int, float]] = None,
) -> List[dict]:
    with open(results_path, encoding="utf-8") as f:
        records = json.load(f)
    if reference is not None:
        records = [r for r in records if r.get("reference") == reference]
    if chinchilla is not None:
        records = [r for r in records if r.get("chinchilla") == chinchilla]
    return records


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--npz", help="Single divergence .npz to process")
    ap.add_argument(
        "--from-results",
        help="Processed divergence_results.json; process every record",
    )
    ap.add_argument(
        "--npz-dir",
        default=None,
        help="Directory containing .npz files when using --from-results",
    )
    ap.add_argument(
        "--dataset-cache",
        default=None,
        help="Local dataset cache dir (same layout as training jobs). "
        "Required unless --memmap-path is set.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory for per-model .txt reports",
    )
    ap.add_argument(
        "--key",
        default="kl_forward",
        choices=["kl_forward", "kl_reverse"],
        help="KL metric to rank by (default: kl_forward)",
    )
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument(
        "--max-kl",
        type=float,
        default=None,
        help="If set, list all positions with KL <= this threshold (instead of top-k)",
    )
    ap.add_argument(
        "--chinchilla",
        type=float,
        default=None,
        help="When using --from-results, keep only this chinchilla budget",
    )
    ap.add_argument(
        "--context-tokens",
        type=int,
        default=64,
        help="How many preceding tokens to include as context (default: 64)",
    )
    ap.add_argument("--tokenizer", default=_DEFAULT_TOKENIZER)
    ap.add_argument(
        "--reference",
        default=None,
        help="When using --from-results, keep only this reference tag",
    )
    ap.add_argument(
        "--fallback-reference",
        default="OLMo-2-1124-1B",
        help="Reference label for old .npz files without -vs-<ref> in the name",
    )
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument(
        "--memmap-path",
        default=None,
        help="Override memmap path instead of resolving from dataset label",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    if not args.npz and not args.from_results:
        raise SystemExit("Pass --npz or --from-results")
    if args.npz and args.from_results:
        raise SystemExit("Pass only one of --npz or --from-results")

    if not args.dataset_cache and not args.memmap_path:
        raise SystemExit("Pass --dataset-cache or --memmap-path")
    dataset_cache = args.dataset_cache or "."

    os.makedirs(args.out_dir, exist_ok=True)

    if args.npz:
        process_npz(
            args.npz,
            args.out_dir,
            dataset_cache=dataset_cache,
            metric_key=args.key,
            top_k=args.top_k,
            context_tokens=args.context_tokens,
            tokenizer_name=args.tokenizer,
            fallback_reference=args.fallback_reference,
            chunk_size=args.chunk_size,
            max_instances=args.max_instances,
            memmap_path=args.memmap_path,
            batch_size=args.batch_size,
            max_kl=args.max_kl,
        )
        return

    if not args.npz_dir:
        raise SystemExit("--npz-dir is required with --from-results")

    records = _records_from_results(
        args.from_results,
        reference=args.reference,
        chinchilla=args.chinchilla,
    )
    if not records:
        raise SystemExit("No records matched the reference filter.")

    for rec in records:
        npz_path = os.path.join(args.npz_dir, rec["file"])
        if not os.path.isfile(npz_path):
            print(f"[warn] missing {npz_path}, skipping", file=sys.stderr)
            continue
        process_npz(
            npz_path,
            args.out_dir,
            dataset_cache=dataset_cache,
            metric_key=args.key,
            top_k=args.top_k,
            context_tokens=args.context_tokens,
            tokenizer_name=args.tokenizer,
            fallback_reference=args.fallback_reference,
            chunk_size=args.chunk_size,
            max_instances=args.max_instances,
            memmap_path=args.memmap_path,
            batch_size=args.batch_size,
            max_kl=args.max_kl,
        )


if __name__ == "__main__":
    main()
