#!/usr/bin/env python3
"""Perturb per-token output logits and measure CE/NLL degradation.

LLM analogue of the associative-memory output-vector experiment
(``multi_min_output_perturbation``): for each next-token position with
logits ``ℓ ∈ ℝ^V``,

    ℓ ← ℓ + σ · ‖ℓ‖₂ · ξ,     ‖ξ‖₂ = 1

(relative ℓ₂ noise; ``σ`` is dimensionless). One forward pass per batch;
sigmas / directions are applied in-place on that batch's logits.

Config YAML (written by :class:`LogitPerturbEvaluation`)::

    model: {type: olmo|hf, path: ...}
    chunk_size: 1024
    batch_size: 4
    device: cuda
    max_instances: 512
    num_directions: 5
    seed: 0
    sigma_min: 1.0e-3
    sigma_max: 10.0
    sigma_ratio: 1.4142135623730951
    validation_datasets:
      - name: C4_val
        paths: [...]

Output JSON::

    {
      "sigmas": [...],                 # includes 0.0 first
      "mean_nll": [...],               # mean over tokens (& directions)
      "std_nll": [...],                # std over directions of mean-token NLL
      "baseline_mean_nll": float,
      "num_tokens": int,
      "num_directions": int,
      "normalize": "per_token_l2",
      ...
    }

Example::

    python3 scripts/logit_perturb_ce_eval.py config.yaml --output out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-token NLL. logits [B,L,V], targets [B,L]."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def build_sigmas(sigma_min: float, sigma_max: float, ratio: float) -> List[float]:
    if sigma_min <= 0 or sigma_max <= sigma_min or ratio <= 1.0:
        raise ValueError(
            f"need 0 < sigma_min < sigma_max and ratio > 1; got "
            f"{sigma_min}, {sigma_max}, {ratio}"
        )
    k_max = int(math.ceil(math.log(sigma_max / sigma_min, ratio)))
    sigmas = [sigma_min * (ratio ** k) for k in range(k_max + 1)]
    if sigmas[-1] > sigma_max:
        sigmas[-1] = sigma_max
    return sigmas


def _sample_unit_dirs(
    K: int, V: int, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return (K, V) unit-ℓ₂ Gaussian directions."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out = torch.empty(K, V, device=device, dtype=dtype)
    for k in range(K):
        xi = torch.randn(V, generator=gen, dtype=torch.float32)
        out[k] = (xi / xi.norm()).to(device=device, dtype=dtype)
    return out


@torch.inference_mode()
def collect_logit_perturb_nll(
    model,
    model_type: str,
    datasets,
    device: torch.device,
    batch_size: int,
    chunk_size: int,
    iter_batches_memmap,
    sigmas: List[float],
    num_directions: int,
    seed: int,
) -> Dict[str, Any]:
    """Accumulate mean NLL at σ=0 and each σ>0 (relative per-token ℓ₂)."""
    # Running sums: [1+len(sigmas)] mean-NLL over tokens, and per-direction
    # token sums so we can also report std across directions.
    n_sigma = len(sigmas)
    sum_nll = torch.zeros(1 + n_sigma, device=device, dtype=torch.float64)
    sum_nll_dir = torch.zeros(
        n_sigma, num_directions, device=device, dtype=torch.float64
    )
    n_tokens = 0
    vocab = None
    dirs = None  # (K, V), built on first batch once V is known

    for ds in datasets:
        max_inst = ds.get("max_instances")
        for np_batch in iter_batches_memmap(
            ds["paths"], chunk_size, batch_size, max_inst
        ):
            ids = torch.from_numpy(np_batch.astype(np.int64)).to(device)
            logits = _logits(model, model_type, ids)  # [B, L, V]
            logits_use = logits[:, :-1, :].float()
            targets = ids[:, 1:]
            B, L, V = logits_use.shape

            if vocab is None:
                vocab = V
                dirs = _sample_unit_dirs(
                    num_directions, V, seed, device, torch.float32
                )

            # Baseline (σ = 0).
            nll0 = nll_from_logits(logits_use, targets)  # [B, L]
            sum_nll[0] += nll0.double().sum()
            n_tok = nll0.numel()
            n_tokens += n_tok

            # Relative ℓ₂ scales per token position: ‖ℓ‖₂ of shape [B, L, 1].
            scales = logits_use.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            for s_idx, sigma in enumerate(sigmas):
                dir_means = []
                for k in range(num_directions):
                    # Broadcast (1,1,V) direction onto (B,L,V).
                    noise = (sigma * scales) * dirs[k].view(1, 1, V)
                    nll = nll_from_logits(logits_use + noise, targets)
                    nll_sum = nll.double().sum()
                    sum_nll_dir[s_idx, k] += nll_sum
                    dir_means.append(nll_sum)
                # Average over directions for the primary mean curve.
                sum_nll[1 + s_idx] += sum(dir_means) / max(num_directions, 1)

    if n_tokens == 0:
        raise RuntimeError("No tokens scored — check validation_datasets / max_instances")

    mean_nll = (sum_nll / n_tokens).cpu().numpy().tolist()
    # Std of (per-direction mean-token NLL) across directions, for each σ>0.
    dir_means = (sum_nll_dir / n_tokens).cpu().numpy()  # [S, K]
    std_nll = (
        dir_means.std(axis=1, ddof=1).tolist()
        if num_directions > 1
        else [0.0] * n_sigma
    )

    return {
        "sigmas": [0.0] + [float(s) for s in sigmas],
        "mean_nll": [float(x) for x in mean_nll],
        "std_nll_across_dirs": [0.0] + [float(x) for x in std_nll],
        "baseline_mean_nll": float(mean_nll[0]),
        "num_tokens": int(n_tokens),
        "num_directions": int(num_directions),
        "vocab_size": int(vocab) if vocab is not None else None,
        "normalize": "per_token_l2",
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    detect_device, iter_batches_memmap, load_hf, load_olmo = _load_eval_helpers()

    device_spec = cfg.get("device")
    device = detect_device(device_spec)
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    path = model_cfg["path"]
    print(f"Loading {model_type} model from {path} on {device}", file=sys.stderr)
    if model_type == "hf":
        model = load_hf(path, device)
    else:
        model = load_olmo(path, device)

    sigma_min = float(cfg.get("sigma_min", 1e-3))
    sigma_max = float(cfg.get("sigma_max", 10.0))
    sigma_ratio = float(cfg.get("sigma_ratio", math.sqrt(2.0)))
    if "sigmas" in cfg and cfg["sigmas"]:
        sigmas = [float(s) for s in cfg["sigmas"]]
    else:
        sigmas = build_sigmas(sigma_min, sigma_max, sigma_ratio)

    num_directions = int(cfg.get("num_directions", 5))
    seed = int(cfg.get("seed", 0))
    chunk_size = int(cfg["chunk_size"])
    batch_size = int(cfg["batch_size"])

    datasets = cfg["validation_datasets"]
    # Optional global max_instances applied to every dataset if per-ds unset.
    global_max = cfg.get("max_instances")
    if global_max is not None:
        for ds in datasets:
            ds.setdefault("max_instances", int(global_max))

    print(
        f"σ schedule ({len(sigmas)}): {sigmas[0]:.3g} → {sigmas[-1]:.3g}; "
        f"K={num_directions}; chunk={chunk_size}; batch={batch_size}",
        file=sys.stderr,
    )

    results = collect_logit_perturb_nll(
        model,
        model_type,
        datasets,
        device,
        batch_size,
        chunk_size,
        iter_batches_memmap,
        sigmas,
        num_directions,
        seed,
    )
    results["model_path"] = path
    results["model_type"] = model_type
    results["datasets"] = [ds.get("name") for ds in datasets]
    results["chunk_size"] = chunk_size
    results["batch_size"] = batch_size
    results["max_instances"] = global_max

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(
        f"Wrote {args.output}  baseline_nll={results['baseline_mean_nll']:.4f}  "
        f"tokens={results['num_tokens']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
