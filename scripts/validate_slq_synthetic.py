#!/usr/bin/env python3
"""Validate the SLQ stack against a diagonal Hessian with a known spectrum.

A diagonal ``H`` makes the matvec an elementwise multiply and the exact
eigenvalues whatever we planted, so the real Lanczos recurrence
(``scripts.hessian_sharpness.lanczos_tridiagonal``) and the real analysis code
(``new_utils.hessian_spectrum.SpectralDensity``) can be exercised at the true
parameter count in seconds, with ground truth to compare against.

This is the cheap test that catches sign errors, weight-extraction bugs, and
normalization mistakes before any GPU-hours are spent.

Memory: the Lanczos basis is ``steps × num_params`` float32 on the host, i.e.
~24 GB at the defaults. Run it on a compute node, or scale down
``--num-params`` for a quick smoke test.

Example::

    python scripts/validate_slq_synthetic.py
    python scripts/validate_slq_synthetic.py --num-params 2000000 --top-k 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from new_utils.hessian_spectrum import SpectralDensity  # noqa: E402
from scripts.hessian_sharpness import (  # noqa: E402
    lanczos_tridiagonal,
    rademacher_probe,
    slq_nodes_weights,
)

# Planted outliers, well separated from the tail.
OUTLIERS = np.array([0.70, 0.50, 0.35, 0.25, 0.20], dtype=np.float64)


def planted_spectrum(num_params: int, seed: int = 0) -> np.ndarray:
    """A few outliers, a power-law tail, and a large near-zero bulk.

    Shaped to resemble a trained-network Hessian: almost all mass at ~0, almost
    all curvature in a thin tail.
    """
    rng = np.random.default_rng(seed)
    eigenvalues = np.empty(num_params, dtype=np.float64)

    num_outliers = OUTLIERS.size
    num_tail = max(int(0.01 * num_params), 1)
    num_bulk = num_params - num_outliers - num_tail
    if num_bulk < 0:
        raise ValueError("num_params too small for the planted structure")

    eigenvalues[:num_outliers] = OUTLIERS
    tail_index = np.arange(1, num_tail + 1, dtype=np.float64)
    eigenvalues[num_outliers : num_outliers + num_tail] = 0.15 * tail_index**-0.7
    eigenvalues[num_outliers + num_tail :] = np.abs(
        rng.normal(0.0, 2e-6, size=num_bulk)
    )
    return eigenvalues


def exact_stats(eigenvalues: np.ndarray, top_k: int) -> dict:
    descending = np.sort(eigenvalues)[::-1]
    total = float(descending.sum())
    positive = descending[descending > 0]
    q = positive / total
    entropy = -float((q * np.log(q)).sum())
    return {
        "max_eigenvalue": float(descending[0]),
        "trace": total,
        "trace_sq": float((descending**2).sum()),
        "entropy_erank": float(np.exp(entropy)),
        "participation_ratio_erank": total**2 / float((descending**2).sum()),
        f"eigenvalue_at_{top_k}": float(descending[min(top_k, descending.size) - 1]),
        f"trace_fraction_top_{top_k}": float(descending[:top_k].sum()) / total,
    }


def run_slq(
    eigenvalues: np.ndarray,
    steps: int,
    num_probes: int,
    seed: int,
    device: torch.device,
) -> SpectralDensity:
    diagonal = torch.from_numpy(eigenvalues).to(device=device, dtype=torch.float32)
    num_params = eigenvalues.size

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return diagonal * v

    generator = torch.Generator().manual_seed(seed)
    nodes, weights = [], []
    for index in range(num_probes):
        q0 = rademacher_probe(num_params, generator)
        alphas, betas = lanczos_tridiagonal(
            matvec, num_params, steps, q0, device=device
        )
        theta, tau = slq_nodes_weights(alphas, betas)
        nodes.append(theta)
        weights.append(tau)
        print(f"  probe {index + 1}/{num_probes}: max Ritz {theta[0]:.6g}")

    pooled_nodes = np.concatenate(nodes)
    pooled_weights = np.concatenate(weights) / float(num_probes)
    return SpectralDensity(
        pooled_nodes,
        pooled_weights,
        num_params,
        {"lanczos_steps": steps, "num_probes": num_probes},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-params", type=int, default=60_030_976)
    parser.add_argument("--lanczos-steps", type=int, default=100)
    parser.add_argument("--num-probes", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=128_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--spectrum-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"p={args.num_params:,}  m={args.lanczos_steps}  n_v={args.num_probes}  "
        f"device={device}"
    )
    basis_gb = args.lanczos_steps * args.num_params * 4 / 1e9
    print(f"Lanczos basis on host: {basis_gb:.1f} GB")

    eigenvalues = planted_spectrum(args.num_params, seed=args.spectrum_seed)
    truth = exact_stats(eigenvalues, args.top_k)

    print("Running SLQ on the planted diagonal Hessian...")
    density = run_slq(
        eigenvalues, args.lanczos_steps, args.num_probes, args.seed, device
    )
    estimate = density.summary(args.top_k)

    print(f"\n{'statistic':<34} {'exact':>14} {'SLQ':>14} {'rel err':>10}")
    print("-" * 76)
    worst = 0.0
    for key, exact_value in truth.items():
        got = estimate.get(key)
        if got is None:
            continue
        error = abs(got - exact_value) / max(abs(exact_value), 1e-30)
        worst = max(worst, error)
        print(f"{key:<34} {exact_value:>14.6g} {got:>14.6g} {error:>10.2%}")

    print(
        f"\nnegative mass (should be ~0): {density.negative_mass:.3e}"
        f"\nworst relative error: {worst:.2%}"
    )
    print(
        "\nExpect the trace and max eigenvalue to be tight, and the depth-"
        f"{args.top_k} quantile to be looser: Lanczos places few nodes in the "
        "tail."
    )


if __name__ == "__main__":
    main()
