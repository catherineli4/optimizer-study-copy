#!/usr/bin/env python3
"""Split tokens by adamw↔muon logit angle, then run Gaussian logit noise per group.

Pipeline::

  1. Forward both models on C4_val; assign each next-token to a 10° bin
     ``θ = arccos(cos(ℓ_a, ℓ_m))``.
  2. For each angle bin independently, apply relative-ℓ₂ Gaussian noise to
     each model's logits and measure mean NLL (same protocol as
     ``logit_perturb_ce_eval``), only averaging over tokens in that bin.

So each bin is its own data group for the Gaussian perturbation experiment.

Output JSON::

    {
      "chinchilla": 16,
      "sigmas": [0.0, σ1, …],
      "bins": {
        "0-10": {
          "num_tokens": …,
          "adamw": {"mean_nll": [...], "baseline_mean_nll": …,
                    "std_nll_across_dirs": [...]},
          "muon":  {…}
        },
        "10-20": {…},
        …
      }
    }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

ANGLE_BIN_EDGES = list(range(0, 91, 10))


def _load_helpers():
    from scripts.divergence_eval import (
        detect_device,
        iter_batches_memmap,
        load_hf,
        load_olmo,
    )
    from scripts.logit_perturb_ce_eval import (
        _sample_unit_dirs,
        build_sigmas,
        nll_from_logits,
    )

    return (
        detect_device,
        iter_batches_memmap,
        load_hf,
        load_olmo,
        _sample_unit_dirs,
        build_sigmas,
        nll_from_logits,
    )


def _logits(model, model_type: str, ids: torch.Tensor) -> torch.Tensor:
    if model_type == "hf":
        return model(input_ids=ids, use_cache=False).logits
    return model(input_ids=ids)


def _load_model(cfg, device, load_hf, load_olmo):
    t, p = cfg["type"], cfg["path"]
    print(f"Loading {t} from {p} on {device}", file=sys.stderr)
    return (load_hf(p, device) if t == "hf" else load_olmo(p, device)), t


def bin_label(i: int) -> str:
    lo, hi = ANGLE_BIN_EDGES[i], ANGLE_BIN_EDGES[i + 1]
    return f"{lo}-{hi}"


def cosine_angle_deg(la: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
    la_f = la.float().reshape(-1, la.shape[-1])
    lm_f = lm.float().reshape(-1, lm.shape[-1])
    na = la_f.norm(dim=-1).clamp_min(1e-12)
    nm = lm_f.norm(dim=-1).clamp_min(1e-12)
    cos = ((la_f * lm_f).sum(dim=-1) / (na * nm)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos)).reshape(la.shape[:2])


def angle_bin_index(angle_deg: torch.Tensor) -> torch.Tensor:
    """Bin index for θ, same edges as ANGLE_BIN_EDGES (10°). Returns long [B,L]."""
    edges = torch.tensor(
        ANGLE_BIN_EDGES[1:-1], device=angle_deg.device, dtype=angle_deg.dtype
    )
    # digitize: values in [edges[i-1], edges[i]) → i; we want 0 for <10, …
    return torch.bucketize(angle_deg, edges, right=False)


@torch.inference_mode()
def collect_per_bin(
    adamw, adamw_type,
    muon, muon_type,
    datasets,
    device: torch.device,
    batch_size: int,
    chunk_size: int,
    iter_batches_memmap,
    sample_unit_dirs,
    nll_from_logits,
    sigmas: List[float],
    num_directions: int,
    seed: int,
) -> Dict[str, Any]:
    n_bins = len(ANGLE_BIN_EDGES) - 1
    n_sigma = len(sigmas)
    # sum_nll[opt, bin, 1+S]; sum_nll_dir[opt, bin, S, K]; n_tokens[bin]
    sum_nll = torch.zeros(2, n_bins, 1 + n_sigma, device=device, dtype=torch.float64)
    sum_nll_dir = torch.zeros(
        2, n_bins, n_sigma, num_directions, device=device, dtype=torch.float64
    )
    n_tokens = torch.zeros(n_bins, device=device, dtype=torch.float64)
    dirs = None

    models = [(0, adamw, adamw_type), (1, muon, muon_type)]

    for ds in datasets:
        max_inst = ds.get("max_instances")
        for batch_idx, np_batch in enumerate(
            iter_batches_memmap(ds["paths"], chunk_size, batch_size, max_inst)
        ):
            ids = torch.from_numpy(np_batch.astype(np.int64)).to(device)
            targets = ids[:, 1:]
            la = _logits(adamw, adamw_type, ids)[:, :-1, :].float()
            lm = _logits(muon, muon_type, ids)[:, :-1, :].float()
            B, L, V = la.shape
            if dirs is None:
                dirs = sample_unit_dirs(num_directions, V, seed, device, torch.float32)

            # --- split data into angle bins (from both logits, unperturbed) ---
            bins = angle_bin_index(cosine_angle_deg(la, lm))  # [B, L]
            logits_by_opt = {0: la, 1: lm}

            for bi in range(n_bins):
                mask = bins == bi  # [B, L]
                n = int(mask.sum().item())
                if n == 0:
                    continue
                n_tokens[bi] += n

                for opt_i, _model, _mtype in models:
                    logits = logits_by_opt[opt_i]
                    scales = logits.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    nll0 = nll_from_logits(logits, targets)
                    sum_nll[opt_i, bi, 0] += nll0[mask].double().sum()

                    for s_idx, sigma in enumerate(sigmas):
                        dir_sums = []
                        for k in range(num_directions):
                            noise = (sigma * scales) * dirs[k].view(1, 1, V)
                            nll = nll_from_logits(logits + noise, targets)
                            s = nll[mask].double().sum()
                            sum_nll_dir[opt_i, bi, s_idx, k] += s
                            dir_sums.append(s)
                        sum_nll[opt_i, bi, 1 + s_idx] += (
                            sum(dir_sums) / max(num_directions, 1)
                        )

            print(
                f"[eval] {ds['name']} batch {batch_idx:4d} seqs={B} "
                f"tokens_so_far={int(n_tokens.sum().item())}",
                file=sys.stderr, flush=True,
            )
            del la, lm

    # Build per-bin result dicts
    bins_out: Dict[str, Any] = {}
    for bi in range(n_bins):
        label = bin_label(bi)
        nt = int(n_tokens[bi].item())
        if nt == 0:
            bins_out[label] = {"num_tokens": 0}
            continue
        entry: Dict[str, Any] = {"num_tokens": nt}
        for opt_i, name in ((0, "adamw"), (1, "muon")):
            mean = (sum_nll[opt_i, bi] / nt).cpu().numpy()
            dir_means = (sum_nll_dir[opt_i, bi] / nt).cpu().numpy()  # [S, K]
            std = (
                dir_means.std(axis=1, ddof=1).tolist()
                if num_directions > 1
                else [0.0] * n_sigma
            )
            entry[name] = {
                "mean_nll": [float(x) for x in mean],
                "std_nll_across_dirs": [0.0] + [float(x) for x in std],
                "baseline_mean_nll": float(mean[0]),
            }
        bins_out[label] = entry

    return {
        "sigmas": [0.0] + [float(s) for s in sigmas],
        "bins": bins_out,
        "angle_bin_edges": ANGLE_BIN_EDGES,
        "num_directions": int(num_directions),
        "normalize": "per_token_l2",
        "seed": int(seed),
        "num_tokens_total": int(n_tokens.sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    (
        detect_device,
        iter_batches_memmap,
        load_hf,
        load_olmo,
        sample_unit_dirs,
        build_sigmas,
        nll_from_logits,
    ) = _load_helpers()

    device = detect_device(cfg.get("device"))
    adamw, adamw_type = _load_model(cfg["adamw_model"], device, load_hf, load_olmo)
    muon, muon_type = _load_model(cfg["muon_model"], device, load_hf, load_olmo)

    if "sigmas" in cfg and cfg["sigmas"]:
        sigmas = [float(s) for s in cfg["sigmas"]]
    else:
        sigmas = build_sigmas(
            float(cfg.get("sigma_min", 1e-3)),
            float(cfg.get("sigma_max", 10.0)),
            float(cfg.get("sigma_ratio", math.sqrt(2.0))),
        )

    datasets = cfg["validation_datasets"]
    global_max = cfg.get("max_instances")
    if global_max is not None:
        for ds in datasets:
            ds.setdefault("max_instances", int(global_max))

    print(
        f"σ schedule ({len(sigmas)}): {sigmas[0]:.3g} → {sigmas[-1]:.3g}; "
        f"K={cfg.get('num_directions', 5)}; angle bins every 10°",
        file=sys.stderr,
    )

    results = collect_per_bin(
        adamw, adamw_type, muon, muon_type,
        datasets, device,
        int(cfg["batch_size"]), int(cfg["chunk_size"]),
        iter_batches_memmap, sample_unit_dirs, nll_from_logits,
        sigmas,
        int(cfg.get("num_directions", 5)),
        int(cfg.get("seed", 0)),
    )
    results["chinchilla"] = cfg.get("chinchilla")
    results["adamw_run"] = cfg.get("adamw_run")
    results["muon_run"] = cfg.get("muon_run")
    results["chunk_size"] = int(cfg["chunk_size"])
    results["batch_size"] = int(cfg["batch_size"])
    results["max_instances"] = global_max

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    nonempty = {
        k: v["num_tokens"] for k, v in results["bins"].items() if v.get("num_tokens", 0)
    }
    print(
        f"Wrote {args.output}  total_tokens={results['num_tokens_total']}  "
        f"bin_counts={nonempty}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
