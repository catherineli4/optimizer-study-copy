#!/usr/bin/env python3
"""Gaussian **weight** perturbation; measure ΔNLL on adamw↔muon angle-bin groups.

1. Forward clean adamw + muon on C4_val → assign each next-token to a 10°
   logit-angle bin ``θ = arccos(cos(ℓ_a, ℓ_m))`` (fixed data groups).
2. For each γ in a geometric schedule, draw K independent Gaussian weight
   noises (same rule as ``perturb_weights.perturb_state_dict``::

       std = γ · ‖W‖_F / √numel

   ) on each model separately and measure mean NLL **restricted to tokens
   in each angle bin**.

Output JSON (mirrors ``logit_angle_bin_perturb.py``)::

    {
      "gammas": [0.0, γ1, …],
      "bins": {
        "0-10": {"num_tokens": N, "adamw": {...}, "muon": {...}},
        ...
      }
    }
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
import yaml

ANGLE_BIN_EDGES = list(range(0, 91, 10))


def _load_helpers():
    from scripts.divergence_eval import (
        detect_device,
        iter_batches_memmap,
        load_olmo,
    )
    from scripts.logit_perturb_ce_eval import build_sigmas, nll_from_logits
    from new_utils.perturb_weights import perturb_state_dict

    return (
        detect_device,
        iter_batches_memmap,
        load_olmo,
        build_sigmas,
        nll_from_logits,
        perturb_state_dict,
    )


def _logits(model, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids)


def bin_label(i: int) -> str:
    return f"{ANGLE_BIN_EDGES[i]}-{ANGLE_BIN_EDGES[i + 1]}"


def cosine_angle_deg(la: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
    la_f = la.float().reshape(-1, la.shape[-1])
    lm_f = lm.float().reshape(-1, lm.shape[-1])
    na = la_f.norm(dim=-1).clamp_min(1e-12)
    nm = lm_f.norm(dim=-1).clamp_min(1e-12)
    cos = ((la_f * lm_f).sum(dim=-1) / (na * nm)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos)).reshape(la.shape[:2])


def angle_bin_index(angle_deg: torch.Tensor) -> torch.Tensor:
    edges = torch.tensor(
        ANGLE_BIN_EDGES[1:-1], device=angle_deg.device, dtype=angle_deg.dtype
    )
    return torch.bucketize(angle_deg, edges, right=False)


def _cpu_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state(model: torch.nn.Module, state: dict, device: torch.device) -> None:
    model.load_state_dict(
        {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in state.items()},
        strict=True,
    )


@torch.inference_mode()
def collect(
    adamw, muon,
    datasets,
    device: torch.device,
    batch_size: int,
    chunk_size: int,
    iter_batches_memmap,
    nll_from_logits,
    perturb_state_dict,
    gammas: List[float],
    num_directions: int,
    seed: int,
) -> Dict[str, Any]:
    n_bins = len(ANGLE_BIN_EDGES) - 1
    n_g = len(gammas)
    # sum_nll[opt, bin, 1+G]; sum_nll_dir[opt, bin, G, K]
    sum_nll = torch.zeros(2, n_bins, 1 + n_g, device=device, dtype=torch.float64)
    sum_nll_dir = torch.zeros(
        2, n_bins, n_g, num_directions, device=device, dtype=torch.float64
    )
    n_tokens = torch.zeros(n_bins, device=device, dtype=torch.float64)

    # Cache batches + fixed angle bins (from clean logits).
    cached: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (ids_cpu, bins_cpu)

    print("[pass 0] clean forwards → angle bins + baseline NLL", file=sys.stderr)
    for ds in datasets:
        max_inst = ds.get("max_instances")
        for batch_idx, np_batch in enumerate(
            iter_batches_memmap(ds["paths"], chunk_size, batch_size, max_inst)
        ):
            ids = torch.from_numpy(np_batch.astype(np.int64)).to(device)
            targets = ids[:, 1:]
            la = _logits(adamw, ids)[:, :-1, :].float()
            lm = _logits(muon, ids)[:, :-1, :].float()
            bins = angle_bin_index(cosine_angle_deg(la, lm))
            cached.append((ids.cpu(), bins.cpu()))

            nll_a = nll_from_logits(la, targets)
            nll_m = nll_from_logits(lm, targets)
            for bi in range(n_bins):
                mask = bins == bi
                n = int(mask.sum().item())
                if n == 0:
                    continue
                n_tokens[bi] += n
                sum_nll[0, bi, 0] += nll_a[mask].double().sum()
                sum_nll[1, bi, 0] += nll_m[mask].double().sum()

            print(
                f"  batch {batch_idx:4d} seqs={ids.shape[0]} "
                f"tokens={int(n_tokens.sum().item())}",
                file=sys.stderr, flush=True,
            )
            del la, lm

    clean_states = [_cpu_state(adamw), _cpu_state(muon)]
    models = [adamw, muon]

    for opt_i, name in ((0, "adamw"), (1, "muon")):
        print(f"[pass {name}] weight perturbation over {n_g} γ × {num_directions} draws",
              file=sys.stderr)
        for g_idx, gamma in enumerate(gammas):
            for k in range(num_directions):
                draw_seed = int(seed) + 10007 * opt_i + 97 * g_idx + k
                pert = perturb_state_dict(
                    clean_states[opt_i], float(gamma), seed=draw_seed
                )
                _load_state(models[opt_i], pert, device)

                for ids_cpu, bins_cpu in cached:
                    ids = ids_cpu.to(device)
                    bins = bins_cpu.to(device)
                    targets = ids[:, 1:]
                    logits = _logits(models[opt_i], ids)[:, :-1, :].float()
                    nll = nll_from_logits(logits, targets)
                    for bi in range(n_bins):
                        mask = bins == bi
                        if not mask.any():
                            continue
                        s = nll[mask].double().sum()
                        sum_nll_dir[opt_i, bi, g_idx, k] += s
                    del logits, nll

                # Average contrib for primary mean curve
                for bi in range(n_bins):
                    # filled after loop over k — do after k loop
                    pass

                print(
                    f"  {name} γ={gamma:.3g} draw {k+1}/{num_directions}",
                    file=sys.stderr, flush=True,
                )

            # Mean over directions into sum_nll[..., 1+g_idx]
            for bi in range(n_bins):
                sum_nll[opt_i, bi, 1 + g_idx] = (
                    sum_nll_dir[opt_i, bi, g_idx].sum() / max(num_directions, 1)
                )

            # Restore clean weights before next gamma
            _load_state(models[opt_i], clean_states[opt_i], device)

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
            dir_means = (sum_nll_dir[opt_i, bi] / nt).cpu().numpy()
            std = (
                dir_means.std(axis=1, ddof=1).tolist()
                if num_directions > 1
                else [0.0] * n_g
            )
            entry[name] = {
                "mean_nll": [float(x) for x in mean],
                "std_nll_across_dirs": [0.0] + [float(x) for x in std],
                "baseline_mean_nll": float(mean[0]),
            }
        bins_out[label] = entry

    return {
        "gammas": [0.0] + [float(g) for g in gammas],
        "sigmas": [0.0] + [float(g) for g in gammas],  # alias for shared plotter
        "bins": bins_out,
        "angle_bin_edges": ANGLE_BIN_EDGES,
        "num_directions": int(num_directions),
        "normalize": "weight_frobenius_rms",
        "perturbation": "gaussian_weights",
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
        load_olmo,
        build_sigmas,
        nll_from_logits,
        perturb_state_dict,
    ) = _load_helpers()

    device = detect_device(cfg.get("device"))
    print(f"Loading adamw from {cfg['adamw_model']['path']}", file=sys.stderr)
    adamw = load_olmo(cfg["adamw_model"]["path"], device)
    print(f"Loading muon from {cfg['muon_model']['path']}", file=sys.stderr)
    muon = load_olmo(cfg["muon_model"]["path"], device)

    if "gammas" in cfg and cfg["gammas"]:
        gammas = [float(g) for g in cfg["gammas"]]
    elif "sigmas" in cfg and cfg["sigmas"]:
        gammas = [float(g) for g in cfg["sigmas"]]
    else:
        gammas = build_sigmas(
            float(cfg.get("gamma_min", cfg.get("sigma_min", 1e-4))),
            float(cfg.get("gamma_max", cfg.get("sigma_max", 5e-2))),
            float(cfg.get("gamma_ratio", cfg.get("sigma_ratio", math.sqrt(2.0)))),
        )

    datasets = cfg["validation_datasets"]
    global_max = cfg.get("max_instances")
    if global_max is not None:
        for ds in datasets:
            ds.setdefault("max_instances", int(global_max))

    print(
        f"γ schedule ({len(gammas)}): {gammas[0]:.3g} → {gammas[-1]:.3g}; "
        f"K={cfg.get('num_directions', 5)}",
        file=sys.stderr,
    )

    results = collect(
        adamw, muon,
        datasets, device,
        int(cfg["batch_size"]), int(cfg["chunk_size"]),
        iter_batches_memmap, nll_from_logits, perturb_state_dict,
        gammas,
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
        f"Wrote {args.output}  tokens={results['num_tokens_total']}  "
        f"bin_counts={nonempty}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
