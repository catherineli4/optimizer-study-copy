#!/usr/bin/env python3
"""Perturb per-token student logits and measure KL vs a 1B reference.

Same relative-ℓ₂ noise as ``logit_perturb_ce_eval.py``::

    ℓ ← ℓ + σ · ‖ℓ‖₂ · ξ,     ‖ξ‖₂ = 1

but the scored quantity is the primary C4-divergence KL::

    kl_forward = KL(Q ‖ P)   where Q = 1B teacher, P = (perturbed) student

Teacher log-probs are computed once per batch; only student logits are
noised across (σ, direction).

Config YAML (written by :class:`LogitPerturbKlEvaluation`)::

    model: {type: olmo|hf, path: ...}
    reference: {path: allenai/OLMo-2-0425-1B, use_cache_if_complete: true}
    reference_device: cuda
    chunk_size: 1024
    batch_size: 2
    ...

Output JSON::

    {
      "sigmas": [...],                 # 0.0 first
      "mean_kl_forward": [...],
      "std_kl_forward_across_dirs": [...],
      "baseline_mean_kl_forward": float,
      "mean_kl_reverse": [...],        # optional secondary
      "mean_jsd": [...],
      ...
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
import torch.nn.functional as F
import yaml


def _load_eval_helpers():
    from scripts.divergence_eval import (
        detect_device,
        divergences_from_logprobs,
        iter_batches_memmap,
        load_hf,
        load_olmo,
        load_reference,
    )

    return (
        detect_device,
        divergences_from_logprobs,
        iter_batches_memmap,
        load_hf,
        load_olmo,
        load_reference,
    )


def _logits(model: torch.nn.Module, model_type: str, ids: torch.Tensor) -> torch.Tensor:
    if model_type == "hf":
        return model(input_ids=ids, use_cache=False).logits
    return model(input_ids=ids)


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
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out = torch.empty(K, V, device=device, dtype=dtype)
    for k in range(K):
        xi = torch.randn(V, generator=gen, dtype=torch.float32)
        out[k] = (xi / xi.norm()).to(device=device, dtype=dtype)
    return out


@torch.inference_mode()
def collect_logit_perturb_kl(
    student,
    student_type: str,
    reference,
    ref_device: torch.device,
    datasets,
    student_device: torch.device,
    batch_size: int,
    chunk_size: int,
    iter_batches_memmap,
    divergences_from_logprobs,
    sigmas: List[float],
    num_directions: int,
    seed: int,
) -> Dict[str, Any]:
    n_sigma = len(sigmas)
    # sums over tokens: [1+S] for baseline + each σ
    sum_klf = torch.zeros(1 + n_sigma, device=student_device, dtype=torch.float64)
    sum_klr = torch.zeros(1 + n_sigma, device=student_device, dtype=torch.float64)
    sum_jsd = torch.zeros(1 + n_sigma, device=student_device, dtype=torch.float64)
    sum_klf_dir = torch.zeros(
        n_sigma, num_directions, device=student_device, dtype=torch.float64
    )
    n_tokens = 0
    vocab = None
    dirs = None

    for ds in datasets:
        max_inst = ds.get("max_instances")
        for batch_idx, np_batch in enumerate(
            iter_batches_memmap(ds["paths"], chunk_size, batch_size, max_inst)
        ):
            ids_s = torch.from_numpy(np_batch.astype(np.int64)).to(student_device)
            logits_s = _logits(student, student_type, ids_s)[:, :-1, :].float()
            B, L, V = logits_s.shape

            ids_r = ids_s.to(ref_device) if ref_device != student_device else ids_s
            logits_q = _logits(reference, "hf", ids_r)[:, :-1, :].float()
            if ref_device != student_device:
                logits_q = logits_q.to(student_device)
            log_q = F.log_softmax(logits_q, dim=-1)
            del logits_q

            if vocab is None:
                vocab = V
                dirs = _sample_unit_dirs(
                    num_directions, V, seed, student_device, torch.float32
                )

            # Baseline σ = 0
            log_p0 = F.log_softmax(logits_s, dim=-1)
            klf0, klr0, jsd0 = divergences_from_logprobs(log_p0, log_q)
            sum_klf[0] += klf0.double().sum()
            sum_klr[0] += klr0.double().sum()
            sum_jsd[0] += jsd0.double().sum()
            n_tok = klf0.numel()
            n_tokens += n_tok
            del log_p0, klf0, klr0, jsd0

            scales = logits_s.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            for s_idx, sigma in enumerate(sigmas):
                acc_klf = 0.0
                acc_klr = 0.0
                acc_jsd = 0.0
                for k in range(num_directions):
                    noise = (sigma * scales) * dirs[k].view(1, 1, V)
                    log_p = F.log_softmax(logits_s + noise, dim=-1)
                    klf, klr, jsd = divergences_from_logprobs(log_p, log_q)
                    klf_sum = klf.double().sum()
                    sum_klf_dir[s_idx, k] += klf_sum
                    acc_klf += float(klf_sum)
                    acc_klr += float(klr.double().sum())
                    acc_jsd += float(jsd.double().sum())
                    del log_p, klf, klr, jsd
                inv_k = 1.0 / max(num_directions, 1)
                sum_klf[1 + s_idx] += acc_klf * inv_k
                sum_klr[1 + s_idx] += acc_klr * inv_k
                sum_jsd[1 + s_idx] += acc_jsd * inv_k

            print(
                f"[eval] {ds['name']} batch {batch_idx:4d} "
                f"seqs={np_batch.shape[0]} tokens_so_far={n_tokens}",
                file=sys.stderr,
                flush=True,
            )

    if n_tokens == 0:
        raise RuntimeError("No tokens scored — check validation_datasets / max_instances")

    mean_klf = (sum_klf / n_tokens).cpu().numpy()
    mean_klr = (sum_klr / n_tokens).cpu().numpy()
    mean_jsd = (sum_jsd / n_tokens).cpu().numpy()
    dir_means = (sum_klf_dir / n_tokens).cpu().numpy()
    std_klf = (
        dir_means.std(axis=1, ddof=1).tolist()
        if num_directions > 1
        else [0.0] * n_sigma
    )

    return {
        "sigmas": [0.0] + [float(s) for s in sigmas],
        "mean_kl_forward": [float(x) for x in mean_klf],
        "mean_kl_reverse": [float(x) for x in mean_klr],
        "mean_jsd": [float(x) for x in mean_jsd],
        "std_kl_forward_across_dirs": [0.0] + [float(x) for x in std_klf],
        "baseline_mean_kl_forward": float(mean_klf[0]),
        "baseline_mean_kl_reverse": float(mean_klr[0]),
        "baseline_mean_jsd": float(mean_jsd[0]),
        "num_tokens": int(n_tokens),
        "num_directions": int(num_directions),
        "vocab_size": int(vocab) if vocab is not None else None,
        "normalize": "per_token_l2",
        "kl_primary": "kl_forward=KL(Q||P)",
        "seed": int(seed),
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
        divergences_from_logprobs,
        iter_batches_memmap,
        load_hf,
        load_olmo,
        load_reference,
    ) = _load_eval_helpers()

    student_device = detect_device(cfg.get("device"))
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    path = model_cfg["path"]
    print(f"Loading student {model_type} from {path} on {student_device}", file=sys.stderr)
    if model_type == "hf":
        student = load_hf(path, student_device)
    else:
        student = load_olmo(path, student_device)

    ref_cfg = cfg["reference"]
    ref_path = ref_cfg["path"]
    ref_prefer = cfg.get("reference_device")
    print(f"Loading reference {ref_path!r}", file=sys.stderr)
    reference, ref_device = load_reference(
        ref_path,
        student_device,
        prefer=ref_prefer,
        use_cache_if_complete=bool(ref_cfg.get("use_cache_if_complete", True)),
    )

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
    global_max = cfg.get("max_instances")
    if global_max is not None:
        for ds in datasets:
            ds.setdefault("max_instances", int(global_max))

    print(
        f"σ schedule ({len(sigmas)}): {sigmas[0]:.3g} → {sigmas[-1]:.3g}; "
        f"K={num_directions}; chunk={chunk_size}; batch={batch_size}; "
        f"ref_device={ref_device}",
        file=sys.stderr,
    )

    results = collect_logit_perturb_kl(
        student,
        model_type,
        reference,
        ref_device,
        datasets,
        student_device,
        batch_size,
        chunk_size,
        iter_batches_memmap,
        divergences_from_logprobs,
        sigmas,
        num_directions,
        seed,
    )
    results["model_path"] = path
    results["model_type"] = model_type
    results["reference_path"] = ref_path
    results["reference_device"] = str(ref_device)
    results["datasets"] = [ds.get("name") for ds in datasets]
    results["chunk_size"] = chunk_size
    results["batch_size"] = batch_size
    results["max_instances"] = global_max

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(
        f"Wrote {args.output}  baseline_kl_fwd={results['baseline_mean_kl_forward']:.4f}  "
        f"tokens={results['num_tokens']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
