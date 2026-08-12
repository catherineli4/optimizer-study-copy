#!/usr/bin/env python3
"""Offline analysis of Hessian spectral-density (SLQ) artifacts.

Reads the ``…-sharpness-spectral_density-m{m}-nv{nv}.json`` files written by
``scripts/evaluate_sharpness.py --metrics spectral_density`` and derives every
downstream statistic on CPU. The GPU job stores only the raw Lanczos nodes and
quadrature weights, so anything here can be recomputed or added without
re-running the expensive part.

Recall what the stored payload is: per probe, ``m`` Ritz values (in curvature
units) and ``m`` weights summing to 1. Pooling across probes gives a discrete
measure approximating ``φ = (1/p) Σ δ(λ_i)``; multiplying weights by ``p``
converts normalized mass back into eigenvalue counts.

Examples::

    # After the sbatch rsync
    python -m new_utils.hessian_spectrum --local-dir results/SharpnessEvaluation

    # Straight from GCS, asking for a different depth
    python -m new_utils.hessian_spectrum --top-k 10000
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from new_utils.plot_jolmo_sharpness import (
    OPT_STYLE,
    REPO_ROOT,
    list_gcs_files,
    list_local_files,
    load_json,
    parse_chinchilla,
    parse_model_type,
    parse_optimizer,
)

DEFAULT_GCS_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/SharpnessEvaluation"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "hessian_spectrum")
DEFAULT_LOCAL_DIR = os.path.join(REPO_ROOT, "results", "SharpnessEvaluation")

SPECTRUM_RE = re.compile(r"-sharpness-spectral_density-m(\d+)-nv(\d+)\.json$")


# ---------------------------------------------------------------------------
# Core: pooled measure and the statistics derived from it
# ---------------------------------------------------------------------------


class SpectralDensity:
    """Pooled SLQ measure: nodes ``θ`` with masses ``w`` summing to 1."""

    # HVPs are accumulated in float32, so Ritz values within ~1e-5 (relative to
    # the spectral edge) of zero are numerical noise: in practice each probe
    # contributes one node holding ~99% of its mass whose *sign* is arbitrary.
    # Statistics that split the spectrum at exactly zero are meaningless; they
    # are cut at this resolution instead.
    DEFAULT_REL_TOL = 1e-4

    def __init__(
        self,
        nodes: np.ndarray,
        weights: np.ndarray,
        num_params: int,
        meta: dict,
        rel_tol: float = DEFAULT_REL_TOL,
    ):
        order = np.argsort(nodes)[::-1]
        self.nodes = np.asarray(nodes, dtype=np.float64)[order]
        self.weights = np.asarray(weights, dtype=np.float64)[order]
        self.num_params = int(num_params)
        self.meta = meta
        self.rel_tol = float(rel_tol)

        # Counting function and cumulative trace, both accumulated from the top.
        self.cum_count = np.cumsum(self.num_params * self.weights)
        self.cum_trace = np.cumsum(self.num_params * self.weights * self.nodes)

    @classmethod
    def from_payload(cls, payload: dict) -> "SpectralDensity":
        block = payload["spectral_density"]
        probes = block["probes"]
        if not probes:
            raise ValueError("spectral_density payload has no probes")
        nodes = np.concatenate([np.asarray(p["ritz_values"], dtype=np.float64) for p in probes])
        weights = np.concatenate([np.asarray(p["weights"], dtype=np.float64) for p in probes])
        weights = weights / float(len(probes))
        meta = {k: v for k, v in block.items() if k != "probes"}
        return cls(nodes, weights, block["num_params"], meta)

    # --- moments -----------------------------------------------------------

    @property
    def trace(self) -> float:
        """``tr(H) = p · Σ w θ``."""
        return float(self.num_params * (self.weights * self.nodes).sum())

    @property
    def trace_sq(self) -> float:
        """``tr(H²) = p · Σ w θ²``."""
        return float(self.num_params * (self.weights * self.nodes**2).sum())

    @property
    def max_eigenvalue(self) -> float:
        return float(self.nodes[0])

    @property
    def min_eigenvalue(self) -> float:
        return float(self.nodes[-1])

    @property
    def resolution(self) -> float:
        """Magnitude below which a Ritz value is indistinguishable from zero."""
        return self.rel_tol * float(np.abs(self.nodes).max())

    @property
    def negative_mass(self) -> float:
        """Fraction of the spectrum resolvably below zero."""
        return float(self.weights[self.nodes < -self.resolution].sum())

    @property
    def unresolved_mass(self) -> float:
        """Fraction of the spectrum within ``±resolution`` of zero.

        Typically ~0.99: almost every eigenvalue of a trained network is too
        close to zero for float32 HVPs to place on either side.
        """
        return float(self.weights[np.abs(self.nodes) <= self.resolution].sum())

    # --- counting function -------------------------------------------------

    def count_above(self, threshold: float) -> float:
        """Estimated number of eigenvalues at least ``threshold``."""
        return float(self.num_params * self.weights[self.nodes >= threshold].sum())

    def eigenvalue_at_index(self, index: float) -> float:
        """Estimate ``λ_index`` by inverting the counting function.

        Linear interpolation of ``λ`` against cumulative count. The nodes are
        sparse in the tail, so this is a bracket-and-interpolate estimate, not a
        resolved eigenvalue.
        """
        if index <= self.cum_count[0]:
            return float(self.nodes[0])
        if index >= self.cum_count[-1]:
            return float(self.nodes[-1])
        return float(np.interp(index, self.cum_count, self.nodes))

    def spectrum_curve(self, indices: np.ndarray) -> np.ndarray:
        """Vectorized ``eigenvalue_at_index``: estimated ``λ`` at each index."""
        return np.interp(
            np.asarray(indices, dtype=np.float64), self.cum_count, self.nodes
        )

    def trace_fraction_in_top(self, count: float) -> float:
        """Share of ``tr(H)`` carried by the top ``count`` eigenvalues.

        Can exceed 1: the Hessian is not PSD, so the positive tail carries more
        curvature than the *net* trace that the negative eigenvalues offset.
        The unresolved bulk contributes to that net at the ~10% level, so treat
        this as accurate to roughly that.
        """
        total = self.cum_trace[-1]
        if total <= 0:
            return float("nan")
        partial = float(np.interp(count, self.cum_count, self.cum_trace))
        return partial / float(total)

    # --- effective rank ----------------------------------------------------

    def participation_ratio_erank(self) -> float:
        """``(Σλ)² / Σλ²``; equals ``p`` for a flat spectrum.

        Uses the whole spectrum deliberately. Both moments are dominated by the
        outlier tail — the unresolved near-zero bulk contributes ~0 to ``Σλ``
        and far less to ``Σλ²`` — so no sign cut is needed, and omitting one
        keeps the statistic from jumping when the bulk lands on the far side of
        zero. This is the robust effective rank for this data.
        """
        first = float((self.weights * self.nodes).sum())
        second = float((self.weights * self.nodes**2).sum())
        if second <= 0:
            return float("nan")
        return float(self.num_params * first**2 / second)

    def entropy_erank(self) -> float:
        """``exp(H)`` for ``H`` the Shannon entropy of ``λ_i / Σλ``.

        Each node stands for ``p·w`` eigenvalues of size ``θ``, so
        ``H = −Σ_j (p w_j) q_j ln q_j`` with ``q_j = θ_j / Σλ``.

        Only nodes above ``resolution`` enter, since ``q ln q`` needs positive
        eigenvalues. Interpret with care: this counts *every* eigenvalue, and
        ~99% of them sit in the unresolved bulk, so the answer depends on where
        that cut falls. Prefer ``participation_ratio_erank``.
        """
        keep = self.nodes > self.resolution
        if not keep.any():
            return float("nan")
        w, t = self.weights[keep], self.nodes[keep]
        total = float(self.num_params * (w * t).sum())
        if total <= 0:
            return float("nan")
        q = t / total
        entropy = -float((self.num_params * w * q * np.log(q)).sum())
        return float(np.exp(entropy))

    # --- plotting ----------------------------------------------------------

    def broadened(
        self, grid: Optional[np.ndarray] = None, sigma: Optional[float] = None,
        num_points: int = 800,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gaussian-broadened density on a grid. ``sigma`` is a display choice."""
        span = self.max_eigenvalue - self.min_eigenvalue
        if sigma is None:
            sigma = max(span * 0.01, 1e-12)
        if grid is None:
            grid = np.linspace(
                self.min_eigenvalue - 4 * sigma,
                self.max_eigenvalue + 4 * sigma,
                num_points,
            )
        norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        exponent = -((grid[:, None] - self.nodes[None, :]) ** 2) / (2.0 * sigma**2)
        density = (self.weights[None, :] * norm * np.exp(exponent)).sum(axis=1)
        return grid, density

    def summary(self, top_k: float) -> dict:
        return {
            "num_params": self.num_params,
            "lanczos_steps": self.meta.get("lanczos_steps"),
            "num_probes": self.meta.get("num_probes"),
            "num_tokens": self.meta.get("num_tokens"),
            "max_eigenvalue": self.max_eigenvalue,
            "min_eigenvalue": self.min_eigenvalue,
            "trace": self.trace,
            "trace_sq": self.trace_sq,
            "mean_eigenvalue": self.trace / self.num_params,
            "resolution": self.resolution,
            "negative_mass": self.negative_mass,
            "unresolved_mass": self.unresolved_mass,
            "participation_ratio_erank": self.participation_ratio_erank(),
            "entropy_erank": self.entropy_erank(),
            f"eigenvalue_at_{int(top_k)}": self.eigenvalue_at_index(top_k),
            f"trace_fraction_top_{int(top_k)}": self.trace_fraction_in_top(top_k),
        }


# ---------------------------------------------------------------------------
# Collection over the model matrix
# ---------------------------------------------------------------------------


def is_spectrum_file(filename: str) -> bool:
    return bool(SPECTRUM_RE.search(filename)) and "-CPT-" not in filename


def collect(paths: List[str], top_k: float):
    """Return ``(rows, densities, by_metric)`` for the pretrain spectrum files."""
    rows: List[dict] = []
    densities: Dict[Tuple[str, float], SpectralDensity] = {}
    by_metric: Dict[str, Dict[str, Dict[float, float]]] = defaultdict(lambda: defaultdict(dict))

    files = [p for p in paths if is_spectrum_file(os.path.basename(p))]
    print(f"Found {len(files)} spectral-density JSON(s)")

    for path in files:
        filename = os.path.basename(path)
        chinchilla = parse_chinchilla(filename)
        optimizer = parse_optimizer(filename)
        if chinchilla is None or optimizer is None:
            print(f"  skip (unparsable): {filename}")
            continue
        try:
            density = SpectralDensity.from_payload(load_json(path))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip (load failed): {filename}: {exc}")
            continue

        stats = density.summary(top_k)
        rows.append(
            {
                "run_name": SPECTRUM_RE.sub("", filename),
                "filename": filename,
                "model_type": parse_model_type(filename),
                "optimizer": optimizer,
                "chinchilla": chinchilla,
                "tokens_per_param": 20.0 * float(chinchilla),
                "path": path,
                **stats,
            }
        )
        densities[(optimizer, chinchilla)] = density

        for key in (
            "max_eigenvalue",
            "trace",
            "entropy_erank",
            "participation_ratio_erank",
            f"eigenvalue_at_{int(top_k)}",
            f"trace_fraction_top_{int(top_k)}",
        ):
            value = stats.get(key)
            if value is not None and np.isfinite(value):
                by_metric[key][optimizer][chinchilla] = float(value)

        print(
            f"  ✓ {optimizer:5s} chinchilla={chinchilla}: "
            f"λmax={stats['max_eigenvalue']:.4g}  tr={stats['trace']:.4g}  "
            f"erank={stats['entropy_erank']:.4g}  "
            f"λ_{int(top_k)}={stats[f'eigenvalue_at_{int(top_k)}']:.4g}"
        )

    rows.sort(key=lambda r: (r["optimizer"], r["chinchilla"]))
    return rows, densities, by_metric


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_densities(densities, out_path: str, sigma_frac: float = 0.01) -> None:
    """One panel per optimizer; curves coloured by Chinchilla multiplier."""
    optimizers = [o for o in ("adamw", "muon") if any(k[0] == o for k in densities)]
    if not optimizers:
        return

    fig, axes = plt.subplots(
        1, len(optimizers), figsize=(6.5 * len(optimizers), 5), squeeze=False
    )
    for ax, optimizer in zip(axes[0], optimizers):
        entries = sorted(
            ((c, d) for (o, c), d in densities.items() if o == optimizer),
            key=lambda kv: kv[0],
        )
        cmap = plt.get_cmap("viridis")
        peak = 0.0
        for i, (chinchilla, density) in enumerate(entries):
            shade = cmap(i / max(len(entries) - 1, 1))
            span = density.max_eigenvalue - density.min_eigenvalue
            grid, values = density.broadened(sigma=max(span * sigma_frac, 1e-12))
            peak = max(peak, float(values.max()))
            ax.plot(grid, values, color=shade, linewidth=1.6, label=f"{chinchilla}×")
        ax.set_yscale("log")
        # Gaussian nulls between well-separated nodes underflow to ~1e-40 and
        # would otherwise squash all the structure into the top decade.
        if peak > 0:
            ax.set_ylim(bottom=peak * 1e-9, top=peak * 3)
        ax.set_xlabel("Eigenvalue λ", fontsize=12)
        ax.set_ylabel("Spectral density φ(λ)", fontsize=12)
        ax.set_title(f"{optimizer}: Hessian spectral density", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Chinchilla", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_spectrum_by_index(
    densities,
    out_path: str,
    top_k: Optional[float] = None,
    num_points: int = 400,
) -> None:
    """Estimated eigenvalue vs its index, one panel per Chinchilla multiplier.

    Curves stop where the estimate falls below the float32 resolution floor;
    past that point the counting function is inverting noise rather than
    spectrum.
    """
    chinchillas = sorted({c for _, c in densities})
    if not chinchillas:
        return

    ncols = min(4, len(chinchillas))
    nrows = -(-len(chinchillas) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False
    )

    for ax, chinchilla in zip(axes.flat, chinchillas):
        for optimizer in ("adamw", "muon"):
            density = densities.get((optimizer, chinchilla))
            if density is None:
                continue
            indices = np.unique(
                np.round(
                    np.logspace(0, np.log10(density.num_params), num_points)
                ).astype(np.int64)
            )
            values = density.spectrum_curve(indices)
            keep = values > density.resolution
            style = OPT_STYLE.get(optimizer, {"color": "gray"})
            ax.plot(
                indices[keep],
                values[keep],
                color=style["color"],
                linewidth=2,
                label=optimizer,
            )

        if top_k:
            ax.axvline(top_k, color="gray", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Chinchilla {chinchilla}×", fontsize=12)
        ax.set_xlabel("Eigenvalue index $i$", fontsize=10)
        ax.set_ylabel(r"$\hat{\lambda}_i$", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)

    for ax in axes.flat[len(chinchillas) :]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_spectrum_ratio(
    densities,
    out_path: str,
    top_k: Optional[float] = None,
    num_points: int = 400,
) -> None:
    """``λ_i(muon) / λ_i(adamw)`` against eigenvalue index, one curve per Chinchilla.

    Each curve is cut where either optimizer's estimate drops below its own
    resolution floor, since past that the ratio divides noise by noise.
    """
    chinchillas = sorted(
        c
        for c in {c for _, c in densities}
        if ("adamw", c) in densities and ("muon", c) in densities
    )
    if not chinchillas:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")

    for i, chinchilla in enumerate(chinchillas):
        adamw = densities[("adamw", chinchilla)]
        muon = densities[("muon", chinchilla)]
        indices = np.unique(
            np.round(
                np.logspace(0, np.log10(min(adamw.num_params, muon.num_params)), num_points)
            ).astype(np.int64)
        )
        va = adamw.spectrum_curve(indices)
        vm = muon.spectrum_curve(indices)
        keep = (va > adamw.resolution) & (vm > muon.resolution)
        ax.plot(
            indices[keep],
            vm[keep] / va[keep],
            color=cmap(i / max(len(chinchillas) - 1, 1)),
            linewidth=1.8,
            label=f"{chinchilla}×",
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    if top_k:
        ax.axvline(top_k, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Eigenvalue index $i$", fontsize=12)
    ax.set_ylabel(r"$\hat{\lambda}_i^{\rm muon} / \hat{\lambda}_i^{\rm adamw}$", fontsize=12)
    ax.set_title("Muon / AdamW eigenvalue ratio", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(title="Chinchilla", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_metric(
    metric: str,
    ylabel: str,
    by_opt: Dict[str, Dict[float, float]],
    out_path: str,
    *,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    all_x = set()

    for optimizer in ("adamw", "muon"):
        series = by_opt.get(optimizer) or {}
        if not series:
            continue
        xs = sorted(series.keys())
        all_x.update(xs)
        style = OPT_STYLE.get(
            optimizer, {"color": "gray", "marker": "x", "linestyle": "--"}
        )
        ax.plot(
            xs,
            [series[x] for x in xs],
            label=optimizer,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=7,
        )

    if not all_x:
        plt.close(fig)
        return

    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    xticks = sorted(all_x)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [str(int(x)) if float(x).is_integer() else str(x) for x in xticks]
    )
    ax.set_xlabel("Chinchilla multiplier", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Hessian spectrum: {metric}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Optimizer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-dir", default=DEFAULT_GCS_DIR)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--top-k",
        type=float,
        default=128000,
        help="Report the eigenvalue at this index and the trace share above it.",
    )
    parser.add_argument(
        "--sigma-frac",
        type=float,
        default=0.01,
        help="Gaussian broadening as a fraction of spectral range (display only).",
    )
    args = parser.parse_args()

    if args.local_dir:
        paths = list_local_files(args.local_dir)
        print(f"Using local dir {args.local_dir} ({len(paths)} json file(s))")
    else:
        paths = list_gcs_files(args.gcs_dir)

    rows, densities, by_metric = collect(paths, args.top_k)
    if not rows:
        print("No spectral-density artifacts found; nothing to do.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {results_path} ({len(rows)} rows)")

    plot_densities(
        densities,
        os.path.join(args.out_dir, "spectral_density.png"),
        sigma_frac=args.sigma_frac,
    )
    plot_spectrum_by_index(
        densities,
        os.path.join(args.out_dir, "spectrum_by_index.png"),
        top_k=args.top_k,
    )
    plot_spectrum_ratio(
        densities,
        os.path.join(args.out_dir, "spectrum_ratio.png"),
        top_k=args.top_k,
    )
    k = int(args.top_k)
    for metric, ylabel, log_y in (
        ("max_eigenvalue", "Max Hessian eigenvalue", False),
        ("trace", "Hessian trace", False),
        ("entropy_erank", "Hessian effective rank (entropy)", True),
        ("participation_ratio_erank", "Hessian effective rank (participation)", True),
        (f"eigenvalue_at_{k}", f"Estimated $\\lambda_{{{k}}}$", True),
        (f"trace_fraction_top_{k}", f"Trace share in top {k}", False),
    ):
        plot_metric(
            metric,
            ylabel,
            by_metric.get(metric, {}),
            os.path.join(args.out_dir, f"{metric}.png"),
            log_y=log_y,
        )


if __name__ == "__main__":
    main()
