#!/usr/bin/env python3
"""Plot angle-binned logit metrics (margin / NLL / KL / frequency) + collect examples.

Reads ``LogitAngleBinEvaluation/*-angle_bin_metrics.npz`` (+ examples JSON)
and writes distribution / frequency / mean-vs-bin figures under
``results/logit_angle_bins/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GCS_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/LogitAngleBinEvaluation"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "logit_angle_bins")

METRIC_SPECS = [
    ("margin_adamw", "classification margin (adamw)"),
    ("margin_muon", "classification margin (muon)"),
    ("top12_adamw", "top1−top2 margin (adamw)"),
    ("top12_muon", "top1−top2 margin (muon)"),
    ("nll_adamw", "NLL (adamw)"),
    ("nll_muon", "NLL (muon)"),
    ("kl_fwd_adamw", r"KL($Q\|P$) vs 1B (adamw)"),
    ("kl_fwd_muon", r"KL($Q\|P$) vs 1B (muon)"),
    ("kl_a_to_m", r"KL(adamw $\|$ muon)"),
    ("kl_m_to_a", r"KL(muon $\|$ adamw)"),
    ("jsd_am", "JSD(adamw, muon)"),
    ("token_freq", "token frequency (count)"),
]

OPT_COLORS = {"adamw": "tab:blue", "muon": "tab:orange"}


def list_gcs_files(gcs_dir: str, suffix: str) -> List[str]:
    result = subprocess.run(
        ["gsutil", "ls", gcs_dir.rstrip("/") + "/"],
        capture_output=True, text=True, check=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().endswith(suffix)
    ]


def list_local_files(local_dir: str, suffix: str) -> List[str]:
    return sorted(
        os.path.join(local_dir, name)
        for name in os.listdir(local_dir)
        if name.endswith(suffix)
    )


def _download_if_gcs(path: str) -> str:
    if not path.startswith("gs://"):
        return path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(path)[1])
    tmp.close()
    subprocess.check_call(
        ["gsutil", "cp", path, tmp.name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return tmp.name


def parse_chin_from_name(name: str) -> Optional[int]:
    m = re.search(r"chinchilla-(\d+)-", name)
    return int(m.group(1)) if m else None


def load_runs(paths: List[str]) -> List[dict]:
    runs = []
    for path in paths:
        chin = parse_chin_from_name(os.path.basename(path))
        local = _download_if_gcs(path)
        try:
            data = dict(np.load(local, allow_pickle=False))
        finally:
            if path.startswith("gs://") and os.path.exists(local):
                os.remove(local)
        runs.append({"chinchilla": chin, "path": path, "data": data})
        print(f"  ✓ chin={chin}: tokens={len(data['angle_deg'])}")
    runs.sort(key=lambda r: (r["chinchilla"] is None, r["chinchilla"] or 0))
    return runs


def pool_runs(runs: List[dict]) -> dict:
    keys = runs[0]["data"].keys()
    return {k: np.concatenate([r["data"][k] for r in runs]) for k in keys}


def bin_labels_from_data(angle_bin: np.ndarray) -> List[Tuple[int, str]]:
    edges = list(range(0, 91, 10))
    present = sorted(set(int(x) for x in angle_bin))
    out = []
    for bi in present:
        if bi >= len(edges) - 1:
            label = f"{edges[-1]}+"
        else:
            label = f"{edges[bi]}-{edges[bi + 1]}"
        out.append((bi, label))
    return out


def plot_metric_hists_by_bin(data: dict, out_path: str, title_prefix: str) -> None:
    bins = bin_labels_from_data(data["angle_bin"])
    # Drop nearly-empty bins
    bins = [(bi, lab) for bi, lab in bins if (data["angle_bin"] == bi).sum() >= 20]
    if not bins:
        return

    metrics = [(k, lab) for k, lab in METRIC_SPECS if k in data]
    n_m, n_b = len(metrics), len(bins)
    fig, axs = plt.subplots(
        n_m, n_b, figsize=(n_b * 2.4, n_m * 2.0), squeeze=False, sharex="row"
    )
    for r, (key, lab) in enumerate(metrics):
        vals = np.asarray(data[key], dtype=float)
        # Shared x-range from pooled percentiles (robust)
        lo, hi = np.percentile(vals, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(vals)), float(np.max(vals))
        for c, (bi, blab) in enumerate(bins):
            ax = axs[r][c]
            m = data["angle_bin"] == bi
            v = vals[m]
            ax.hist(v, bins=40, range=(lo, hi), color="tab:green", alpha=0.8, density=True)
            ax.axvline(float(np.mean(v)), color="black", linestyle="--", linewidth=1.0)
            if r == 0:
                ax.set_title(f"θ∈[{blab}]°\nn={int(m.sum())}", fontsize=8)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{title_prefix}: metric distributions by logit-angle bin", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_mean_vs_bin(data: dict, out_path: str, title_prefix: str) -> None:
    bins = bin_labels_from_data(data["angle_bin"])
    bins = [(bi, lab) for bi, lab in bins if (data["angle_bin"] == bi).sum() >= 20]
    if not bins:
        return
    xs = np.arange(len(bins))
    labels = [lab for _, lab in bins]

    # Pair adamw/muon metrics on shared axes where sensible
    pairs = [
        ("margin", "margin_adamw", "margin_muon", "mean classification margin"),
        ("top12", "top12_adamw", "top12_muon", "mean top1−top2 margin"),
        ("nll", "nll_adamw", "nll_muon", "mean NLL"),
        ("kl_fwd", "kl_fwd_adamw", "kl_fwd_muon", r"mean KL($Q\|P$) vs 1B"),
    ]
    singles = [
        ("kl_a_to_m", r"mean KL(adamw $\|$ muon)"),
        ("token_freq", "mean token frequency"),
    ]

    n_rows = len(pairs) + len(singles)
    fig, axs = plt.subplots(n_rows, 1, figsize=(8, 2.4 * n_rows), squeeze=False)
    row = 0
    for _, ka, km, title in pairs:
        ax = axs[row][0]
        ya = [float(np.mean(data[ka][data["angle_bin"] == bi])) for bi, _ in bins]
        ym = [float(np.mean(data[km][data["angle_bin"] == bi])) for bi, _ in bins]
        ax.plot(xs, ya, "-o", color=OPT_COLORS["adamw"], label="adamw", markersize=5)
        ax.plot(xs, ym, "-s", color=OPT_COLORS["muon"], label="muon", markersize=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        row += 1
    for key, title in singles:
        ax = axs[row][0]
        ys = [float(np.mean(data[key][data["angle_bin"] == bi])) for bi, _ in bins]
        ax.plot(xs, ys, "-o", color="tab:green", markersize=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        row += 1
    axs[-1][0].set_xlabel(r"logit-angle bin $\theta$ (deg)", fontsize=10)
    fig.suptitle(f"{title_prefix}: mean metrics vs angle bin", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_metric_vs_freq(data: dict, out_path: str, title_prefix: str) -> None:
    """Binned mean of metrics vs log token-frequency, one panel per angle bin."""
    bins = bin_labels_from_data(data["angle_bin"])
    bins = [(bi, lab) for bi, lab in bins if (data["angle_bin"] == bi).sum() >= 50]
    if not bins:
        return
    metrics = [
        ("kl_fwd_adamw", "kl_fwd_muon", r"KL($Q\|P$) vs 1B"),
        ("nll_adamw", "nll_muon", "NLL"),
        ("margin_adamw", "margin_muon", "classification margin"),
    ]
    n_b, n_m = len(bins), len(metrics)
    fig, axs = plt.subplots(n_m, n_b, figsize=(n_b * 2.6, n_m * 2.4), squeeze=False)
    freq = np.asarray(data["token_freq"], dtype=float)
    logf = np.log10(np.clip(freq, 1, None))

    for c, (bi, blab) in enumerate(bins):
        m = data["angle_bin"] == bi
        lf = logf[m]
        # frequency quantile bins
        edges = np.quantile(lf, np.linspace(0, 1, 11))
        edges = np.unique(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for r, (ka, km, title) in enumerate(metrics):
            ax = axs[r][c]
            for key, color, label in (
                (ka, OPT_COLORS["adamw"], "adamw"),
                (km, OPT_COLORS["muon"], "muon"),
            ):
                vals = np.asarray(data[key][m], dtype=float)
                means, ns = [], []
                for i in range(len(edges) - 1):
                    sel = (lf >= edges[i]) & (
                        lf <= edges[i + 1] if i == len(edges) - 2 else lf < edges[i + 1]
                    )
                    means.append(float(np.mean(vals[sel])) if sel.any() else np.nan)
                    ns.append(int(sel.sum()))
                ax.plot(centers, means, "-o", color=color, label=label, markersize=3)
            if r == 0:
                ax.set_title(f"θ∈[{blab}]°", fontsize=9)
            if c == 0:
                ax.set_ylabel(title, fontsize=8)
            if r == n_m - 1:
                ax.set_xlabel(r"$\log_{10}$ token freq", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        f"{title_prefix}: metrics vs token frequency (by angle bin)", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_delta_kl_hist(data: dict, out_path: str, title_prefix: str) -> None:
    """ΔKL = KL_muon − KL_adamw (vs 1B), plus Δmargin / ΔNLL by angle bin."""
    bins = bin_labels_from_data(data["angle_bin"])
    bins = [(bi, lab) for bi, lab in bins if (data["angle_bin"] == bi).sum() >= 20]
    if not bins:
        return
    deltas = [
        ("kl_fwd_muon", "kl_fwd_adamw", r"$\Delta$KL vs 1B (muon−adamw)"),
        ("nll_muon", "nll_adamw", r"$\Delta$NLL (muon−adamw)"),
        ("margin_muon", "margin_adamw", r"$\Delta$margin (muon−adamw)"),
    ]
    fig, axs = plt.subplots(
        len(deltas), len(bins),
        figsize=(len(bins) * 2.4, len(deltas) * 2.2), squeeze=False,
    )
    for r, (km, ka, title) in enumerate(deltas):
        d_all = np.asarray(data[km], dtype=float) - np.asarray(data[ka], dtype=float)
        lo, hi = np.percentile(d_all, [1, 99])
        for c, (bi, blab) in enumerate(bins):
            ax = axs[r][c]
            m = data["angle_bin"] == bi
            d = d_all[m]
            ax.hist(d, bins=40, range=(lo, hi), color="tab:purple", alpha=0.8, density=True)
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.axvline(float(np.mean(d)), color="red", linestyle="--", linewidth=1.0)
            if r == 0:
                ax.set_title(f"θ∈[{blab}]°", fontsize=8)
            if c == 0:
                ax.set_ylabel(title, fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{title_prefix}: muon−adamw deltas by angle bin", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def collect_examples(example_paths: List[str], out_dir: str) -> None:
    ex_dir = os.path.join(out_dir, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    for path in example_paths:
        chin = parse_chin_from_name(os.path.basename(path))
        local = _download_if_gcs(path)
        try:
            with open(local, "r", encoding="utf-8") as f:
                payload = json.load(f)
        finally:
            if path.startswith("gs://") and os.path.exists(local):
                os.remove(local)
        out_json = os.path.join(ex_dir, f"chinchilla-{chin}-examples.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # also write txt
        out_txt = os.path.join(ex_dir, f"chinchilla-{chin}-examples.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"chinchilla={chin}\n")
            for label, xs in (payload.get("examples") or {}).items():
                f.write(f"\n{'=' * 72}\nANGLE BIN {label}°  (n={len(xs)})\n{'=' * 72}\n")
                for i, ex in enumerate(xs, 1):
                    f.write(
                        f"\n--- {i}/{len(xs)}  θ={ex['angle_deg']:.2f}°  "
                        f"tok={ex['token_id']} {ex.get('token_text')!r}\n"
                        f"nll_a={ex['nll_adamw']:.3f} nll_m={ex['nll_muon']:.3f}  "
                        f"kl_a={ex['kl_fwd_adamw']:.3f} kl_m={ex['kl_fwd_muon']:.3f}  "
                        f"margin_a={ex['margin_adamw']:.3f} margin_m={ex['margin_muon']:.3f}\n"
                        f"CONTEXT:\n{ex['context']}\n"
                    )
        print(f"  wrote examples for chin={chin}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-dir", default=DEFAULT_GCS_DIR)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.local_dir:
        npz_paths = list_local_files(args.local_dir, "-angle_bin_metrics.npz")
        ex_paths = list_local_files(args.local_dir, "-angle_bin_examples.json")
        print(f"Using local dir {args.local_dir} ({len(npz_paths)} npz)")
    else:
        npz_paths = list_gcs_files(args.gcs_dir, "-angle_bin_metrics.npz")
        ex_paths = list_gcs_files(args.gcs_dir, "-angle_bin_examples.json")

    if not npz_paths:
        raise SystemExit("No angle-bin metric npz files found.")

    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(npz_paths)

    # Pooled across chinchillas
    pooled = pool_runs(runs)
    plot_metric_hists_by_bin(
        pooled,
        os.path.join(args.out_dir, "metric_hists_by_angle_bin_pooled.png"),
        "all chinchillas",
    )
    plot_mean_vs_bin(
        pooled,
        os.path.join(args.out_dir, "mean_metrics_vs_angle_bin_pooled.png"),
        "all chinchillas",
    )
    plot_metric_vs_freq(
        pooled,
        os.path.join(args.out_dir, "metrics_vs_freq_by_angle_bin_pooled.png"),
        "all chinchillas",
    )
    plot_delta_kl_hist(
        pooled,
        os.path.join(args.out_dir, "delta_muon_minus_adamw_by_angle_bin_pooled.png"),
        "all chinchillas",
    )

    # Per-chinchilla mean curves only (lighter)
    for run in runs:
        chin = run["chinchilla"]
        tag = f"chin{chin}"
        d = run["data"]
        plot_mean_vs_bin(
            d,
            os.path.join(args.out_dir, f"mean_metrics_vs_angle_bin_{tag}.png"),
            f"chinchilla={chin}",
        )
        plot_metric_hists_by_bin(
            d,
            os.path.join(args.out_dir, f"metric_hists_by_angle_bin_{tag}.png"),
            f"chinchilla={chin}",
        )

    collect_examples(ex_paths, args.out_dir)
    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
