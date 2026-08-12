#!/usr/bin/env python3
"""Per-token cosine similarity of output logits: adamw vs muon (same chinchilla).

For each next-token position, with logit vectors ``ℓ_a, ℓ_m ∈ ℝ^V``::

    cos = ⟨ℓ_a, ℓ_m⟩ / (‖ℓ_a‖₂ ‖ℓ_m‖₂)

Config YAML (written by :class:`LogitCosineEvaluation`)::

    adamw_model: {type: olmo, path: ...}
    muon_model:  {type: olmo, path: ...}
    chunk_size: 1024
    batch_size: 4
    device: cuda
    max_instances: 512
    validation_datasets: [{name: C4_val, paths: [...], max_instances: 512}]

Output JSON::

    {
      "chinchilla": 8,
      "mean_cosine": float,
      "std_cosine": float,
      "median_cosine": float,
      "p10": float, "p90": float,
      "num_tokens": int,
      "histogram": {"bins": [...], "counts": [...]},
      ...
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml


def _load_eval_helpers():
    from scripts.divergence_eval import (
        detect_device,
        iter_batches_memmap,
        load_hf,
        load_olmo,
    )

    return detect_device, iter_batches_memmap, load_hf, load_olmo


def _logits(model: torch.nn.Module, model_type: str, ids: torch.Tensor) -> torch.Tensor:
    if model_type == "hf":
        return model(input_ids=ids, use_cache=False).logits
    return model(input_ids=ids)


def _load_model(cfg: dict, device: torch.device, load_hf, load_olmo):
    model_type = cfg["type"]
    path = cfg["path"]
    print(f"Loading {model_type} from {path} on {device}", file=sys.stderr)
    if model_type == "hf":
        return load_hf(path, device), model_type
    return load_olmo(path, device), model_type


@torch.inference_mode()
def collect_cosine(
    adamw_model,
    adamw_type: str,
    muon_model,
    muon_type: str,
    datasets,
    device: torch.device,
    batch_size: int,
    chunk_size: int,
    iter_batches_memmap,
) -> Dict[str, Any]:
    # Online stats + reservoir for percentiles / hist (all tokens; max_instances caps size).
    pieces: List[np.ndarray] = []
    n_tokens = 0

    for ds in datasets:
        max_inst = ds.get("max_instances")
        for batch_idx, np_batch in enumerate(
            iter_batches_memmap(ds["paths"], chunk_size, batch_size, max_inst)
        ):
            ids = torch.from_numpy(np_batch.astype(np.int64)).to(device)
            la = _logits(adamw_model, adamw_type, ids)[:, :-1, :].float()
            lm = _logits(muon_model, muon_type, ids)[:, :-1, :].float()
            # cos over vocab: flatten to [N, V]
            la_f = la.reshape(-1, la.shape[-1])
            lm_f = lm.reshape(-1, lm.shape[-1])
            na = la_f.norm(dim=-1).clamp_min(1e-12)
            nm = lm_f.norm(dim=-1).clamp_min(1e-12)
            cos = (la_f * lm_f).sum(dim=-1) / (na * nm)
            cos_np = cos.float().cpu().numpy()
            pieces.append(cos_np)
            n_tokens += int(cos_np.size)
            print(
                f"[eval] {ds['name']} batch {batch_idx:4d} "
                f"seqs={np_batch.shape[0]} tokens_so_far={n_tokens} "
                f"mean_cos={float(cos_np.mean()):.4f}",
                file=sys.stderr,
                flush=True,
            )
            del la, lm, la_f, lm_f, cos

    if n_tokens == 0:
        raise RuntimeError("No tokens scored — check validation_datasets / max_instances")

    all_cos = np.concatenate(pieces).astype(np.float64)
    hist_counts, hist_edges = np.histogram(all_cos, bins=50, range=(-1.0, 1.0))
    return {
        "mean_cosine": float(all_cos.mean()),
        "std_cosine": float(all_cos.std()),
        "median_cosine": float(np.median(all_cos)),
        "p10": float(np.percentile(all_cos, 10)),
        "p50": float(np.percentile(all_cos, 50)),
        "p90": float(np.percentile(all_cos, 90)),
        "min_cosine": float(all_cos.min()),
        "max_cosine": float(all_cos.max()),
        "num_tokens": int(n_tokens),
        "histogram": {
            "bin_edges": hist_edges.tolist(),
            "counts": hist_counts.astype(int).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    detect_device, iter_batches_memmap, load_hf, load_olmo = _load_eval_helpers()
    device = detect_device(cfg.get("device"))

    adamw_model, adamw_type = _load_model(cfg["adamw_model"], device, load_hf, load_olmo)
    muon_model, muon_type = _load_model(cfg["muon_model"], device, load_hf, load_olmo)

    chunk_size = int(cfg["chunk_size"])
    batch_size = int(cfg["batch_size"])
    datasets = cfg["validation_datasets"]
    global_max = cfg.get("max_instances")
    if global_max is not None:
        for ds in datasets:
            ds.setdefault("max_instances", int(global_max))

    results = collect_cosine(
        adamw_model,
        adamw_type,
        muon_model,
        muon_type,
        datasets,
        device,
        batch_size,
        chunk_size,
        iter_batches_memmap,
    )
    results["chinchilla"] = cfg.get("chinchilla")
    results["adamw_path"] = cfg["adamw_model"]["path"]
    results["muon_path"] = cfg["muon_model"]["path"]
    results["adamw_run"] = cfg.get("adamw_run")
    results["muon_run"] = cfg.get("muon_run")
    results["datasets"] = [ds.get("name") for ds in datasets]
    results["chunk_size"] = chunk_size
    results["batch_size"] = batch_size
    results["max_instances"] = global_max

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(
        f"Wrote {args.output}  mean_cos={results['mean_cosine']:.4f}  "
        f"tokens={results['num_tokens']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
