#!/usr/bin/env python3
"""Layer-wise effective rank of matrix weights for JOLMo 60M best-LR models.

For each 2-D parameter ``W`` (biases / LayerNorm scales skipped), estimates

    effective_rank(W) = exp( H(p) ),   p_i = σ_i / Σ_j σ_j

(Shannon / Roy–Vetterli effective rank), plus the cheap stable rank
``‖W‖_F² / ‖W‖₂²``.

Singular values are obtained **exactly** via the smaller Gram matrix
(``WᵀW`` or ``WWᵀ``) when ``min(m, n) ≤ --gram-max``; otherwise a randomized
``torch.svd_lowrank`` approximation is used (top-``q`` singular values only,
so the Shannon rank is a slight underestimate).

Default models: all 60M WSD best-LR adamw / muon runs from
``launch_jolmo.pretraining_matrix.PT_LR_BY_MODEL["0.06B"]["wsd"]``, loaded from
``gs://cmu-gpucloud-catheri4/Optim-60M-tuning/JolmoModel/<run>/final-unsharded/model.pt``
(or a local mirror via ``--local-root``).

Example:
    python scripts/estimate_weight_effective_rank.py \\
        --out-dir results/effective_rank_60M_bestlr \\
        --local-root /scratch/catheri4-outputs/Optim-60M-tuning/JolmoModel
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

GCS_ROOT = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/JolmoModel"


def _lr_tag(lr: float) -> str:
    return f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def best_lr_runs_60m(chinchillas: Iterable[int] | None = None) -> list[dict]:
    """Return [{optimizer, chinchilla, run_name, ...}] for 60M WSD best-LR cells."""
    from launch_jolmo.pretraining_matrix import PT_LR_BY_MODEL

    table = PT_LR_BY_MODEL["0.06B"]["wsd"]
    chins = list(chinchillas) if chinchillas is not None else sorted(
        set(table["adamw"]) | set(table["muon"])
    )
    rows = []
    for c in chins:
        if c in table["adamw"] and table["adamw"][c] is not None:
            lr = float(table["adamw"][c])
            rows.append({
                "optimizer": "adamw",
                "chinchilla": int(c),
                "run_name": (
                    f"MuonExpt3-0.06B-chinchilla-{c}-adamw-lr{_lr_tag(lr)}-wsd"
                ),
                "lr_tag": _lr_tag(lr),
            })
        if c in table["muon"] and table["muon"][c] is not None:
            muon_lr, adamw_lr = table["muon"][c]
            rows.append({
                "optimizer": "muon",
                "chinchilla": int(c),
                "run_name": (
                    f"MuonExpt3-0.06B-chinchilla-{c}-muon-"
                    f"muonlr{_lr_tag(float(muon_lr))}-"
                    f"adamwlr{_lr_tag(float(adamw_lr))}-wsd"
                ),
                "lr_tag": f"muon{_lr_tag(float(muon_lr))}_adamw{_lr_tag(float(adamw_lr))}",
            })
    return rows


def _download_gcs(gcs_path: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    subprocess.check_call(
        ["gsutil", "-o", "GSUtil:sliced_object_download_threshold=0",
         "cp", gcs_path, local_path]
    )


def resolve_model_pt(run_name: str, local_root: str | None,
                     cache_dir: str | None) -> str:
    """Return a local path to ``model.pt`` for ``run_name`` (download if needed)."""
    rel = os.path.join(run_name, "final-unsharded", "model.pt")
    if local_root:
        cand = os.path.join(local_root, rel)
        if os.path.exists(cand):
            return cand
    if cache_dir:
        cand = os.path.join(cache_dir, rel)
        if os.path.exists(cand):
            return cand
        gcs = f"{GCS_ROOT}/{run_name}/final-unsharded/model.pt"
        print(f"  downloading {gcs} -> {cand}", flush=True)
        _download_gcs(gcs, cand)
        return cand
    # ephemeral download
    tmp = tempfile.mkdtemp(prefix="effrank_")
    local = os.path.join(tmp, "model.pt")
    gcs = f"{GCS_ROOT}/{run_name}/final-unsharded/model.pt"
    print(f"  downloading {gcs} -> {local}", flush=True)
    _download_gcs(gcs, local)
    return local


def is_matrix_weight(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2:
        return False
    # skip explicit bias tensors if any slipped through as 2-D
    if name.endswith(".bias"):
        return False
    return True


def singular_values(W: torch.Tensor, gram_max: int = 4096,
                    lowrank_q: int = 256, seed: int = 0) -> torch.Tensor:
    """Descending singular values of W (float32). Exact via Gram when possible."""
    W = W.detach().float().cpu()
    m, n = W.shape
    k = min(m, n)
    if k == 0:
        return torch.zeros(0)
    if k <= gram_max:
        # Exact: eigendecomposition of the smaller Gram matrix.
        if m >= n:
            gram = W.T @ W          # (n, n)
        else:
            gram = W @ W.T          # (m, m)
        # Numerical noise can produce tiny negatives.
        evals = torch.linalg.eigvalsh(gram).clamp(min=0.0)
        return torch.sqrt(evals).flip(0)  # descending
    # Approximate top-q spectrum with randomized SVD.
    q = min(lowrank_q, k)
    # svd_lowrank wants q + some oversampling; torch handles niter.
    gen = torch.Generator().manual_seed(seed)
    # torch.svd_lowrank doesn't take a generator on all versions; set global seed.
    torch.manual_seed(seed)
    _U, S, _V = torch.svd_lowrank(W, q=q, niter=4)
    return S  # already descending


def shannon_effective_rank(sigma: torch.Tensor) -> float:
    s = sigma.double().clamp(min=0.0)
    total = float(s.sum().item())
    if total <= 0.0 or not math.isfinite(total):
        return float("nan")
    p = (s / total).clamp(min=1e-30)
    ent = float(-(p * p.log()).sum().item())
    return float(math.exp(ent))


def stable_rank(sigma: torch.Tensor) -> float:
    s = sigma.double().clamp(min=0.0)
    if s.numel() == 0 or float(s[0].item()) <= 0.0:
        return float("nan")
    fro2 = float((s * s).sum().item())
    return fro2 / float((s[0] * s[0]).item())


def analyze_state_dict(state: dict, gram_max: int, lowrank_q: int,
                       seed: int) -> list[dict]:
    rows = []
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not is_matrix_weight(name, tensor):
            continue
        m, n = int(tensor.shape[0]), int(tensor.shape[1])
        sigma = singular_values(tensor, gram_max=gram_max,
                                lowrank_q=lowrank_q, seed=seed)
        method = "gram_exact" if min(m, n) <= gram_max else f"svd_lowrank_q{min(lowrank_q, min(m, n))}"
        fro = float(torch.linalg.norm(tensor.float()).item())
        rows.append({
            "param": name,
            "shape0": m,
            "shape1": n,
            "rank_max": min(m, n),
            "n_sigma": int(sigma.numel()),
            "method": method,
            "effective_rank": shannon_effective_rank(sigma),
            "stable_rank": stable_rank(sigma),
            "spectral_norm": float(sigma[0].item()) if sigma.numel() else float("nan"),
            "frobenius_norm": fro,
            "sigma_sum": float(sigma.sum().item()) if sigma.numel() else float("nan"),
            "erank_over_rankmax": (
                shannon_effective_rank(sigma) / min(m, n) if min(m, n) > 0 else float("nan")
            ),
        })
    return rows


def _plot_by_layer(rows_by_opt: dict[str, list[dict]], chin: int, out_path: str) -> None:
    """Grouped bar: effective rank / rank_max per param, adamw vs muon."""
    # Use adamw param order as reference; skip params missing on one side.
    ref = rows_by_opt.get("adamw") or rows_by_opt.get("muon") or []
    params = [r["param"] for r in ref]
    if not params:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(params) * 0.35), 4.5))
    x = np.arange(len(params))
    width = 0.38
    colors = {"adamw": "tab:green", "muon": "tab:orange"}
    for i, (opt, rows) in enumerate(sorted(rows_by_opt.items())):
        by_name = {r["param"]: r for r in rows}
        ys = [by_name[p]["erank_over_rankmax"] if p in by_name else np.nan
              for p in params]
        ax.bar(x + (i - 0.5) * width, ys, width=width, label=opt,
               color=colors.get(opt, "gray"), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=90, fontsize=6)
    ax.set_ylabel("effective rank / min(m, n)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"60M best-LR layer effective ranks — chinchilla={chin}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_summary(all_rows: list[dict], out_path: str) -> None:
    """Mean erank/rank_max over non-embed matrices vs chinchilla, by optimizer."""
    import pandas as pd
    df = pd.DataFrame(all_rows)
    # Focus on transformer block matrices (exclude huge embed / lm_head for mean).
    mask = ~df["param"].str.contains("embed|lm_head", case=False, regex=True)
    sub = df[mask].copy()
    if sub.empty:
        sub = df
    g = (sub.groupby(["optimizer", "chinchilla"], as_index=False)["erank_over_rankmax"]
           .mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    for opt, color in (("adamw", "tab:green"), ("muon", "tab:orange")):
        s = g[g["optimizer"] == opt].sort_values("chinchilla")
        if s.empty:
            continue
        ax.plot(s["chinchilla"], s["erank_over_rankmax"], "o-",
                color=color, label=opt)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("chinchilla multiplier")
    ax.set_ylabel("mean effective rank / min(m, n)\n(block matrices only)")
    ax.set_title("60M best-LR — mean layer effective rank vs data scale")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/effective_rank_60M_bestlr")
    ap.add_argument("--local-root", default=None,
                    help="Local JolmoModel root mirroring GCS (…/JolmoModel).")
    ap.add_argument("--cache-dir", default=None,
                    help="Download cache for model.pt files.")
    ap.add_argument("--chinchilla", type=int, nargs="*", default=None,
                    help="Subset of chinchilla multipliers (default: all tuned).")
    ap.add_argument("--gram-max", type=int, default=4096,
                    help="Use exact Gram eigendecomp when min(m,n) ≤ this.")
    ap.add_argument("--lowrank-q", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = best_lr_runs_60m(args.chinchilla)
    print(f"{len(runs)} runs to analyze")

    all_rows: list[dict] = []
    by_chin: dict[int, dict[str, list[dict]]] = {}

    for run in runs:
        print(f"\n=== {run['optimizer']} chin={run['chinchilla']} "
              f"{run['run_name']} ===", flush=True)
        pt = resolve_model_pt(run["run_name"], args.local_root, args.cache_dir)
        state = torch.load(pt, map_location=args.device, weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint type: {type(state)}")
        # Some checkpoints wrap under 'model' / 'state_dict'
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        layer_rows = analyze_state_dict(state, args.gram_max, args.lowrank_q, args.seed)
        print(f"  {len(layer_rows)} matrix weights")
        for r in layer_rows:
            r.update({
                "optimizer": run["optimizer"],
                "chinchilla": run["chinchilla"],
                "run_name": run["run_name"],
            })
            all_rows.append(r)
        by_chin.setdefault(run["chinchilla"], {})[run["optimizer"]] = layer_rows

        # Per-run CSV
        run_csv = out_dir / f"erank_{run['optimizer']}_chin{run['chinchilla']}.csv"
        with open(run_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()) if layer_rows else [])
            if layer_rows:
                w.writeheader()
                w.writerows(layer_rows)
        print(f"  wrote {run_csv}")

    # Combined CSV
    comb = out_dir / "effective_rank_all.csv"
    if all_rows:
        with open(comb, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {comb} ({len(all_rows)} rows)")

    # Plots
    for chin, opt_rows in sorted(by_chin.items()):
        _plot_by_layer(opt_rows, chin,
                       str(out_dir / f"erank_by_layer_chin{chin}.png"))
    _plot_summary(all_rows, str(out_dir / "erank_mean_vs_chinchilla.png"))

    # Small JSON summary
    summary = {
        "n_runs": len(runs),
        "n_rows": len(all_rows),
        "gram_max": args.gram_max,
        "runs": runs,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
