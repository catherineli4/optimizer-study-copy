#!/usr/bin/env python3
"""Plot per-token logit cosine similarity: adamw vs muon (by chinchilla).

Reads ``…/LogitCosineEvaluation/*-logit_cosine.json`` and writes:

  * mean_cosine_vs_chinchilla.png
  * cosine_hist_by_chin.png          — facet histograms
  * final_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GCS_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/LogitCosineEvaluation"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "logit_cosine")


def list_gcs_files(gcs_dir: str) -> List[str]:
    result = subprocess.run(
        ["gsutil", "ls", gcs_dir.rstrip("/") + "/"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().endswith(".json")
    ]


def list_local_files(local_dir: str) -> List[str]:
    return sorted(
        os.path.join(local_dir, name)
        for name in os.listdir(local_dir)
        if name.endswith(".json")
    )


def load_json(path: str) -> dict:
    if path.startswith("gs://"):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name
        try:
            subprocess.check_call(
                ["gsutil", "cp", path, tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(tmp_path, "r") as f:
                return json.load(f)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    with open(path, "r") as f:
        return json.load(f)


def collect(paths: List[str]) -> List[dict]:
    rows = []
    for path in paths:
        filename = os.path.basename(path)
        if not filename.endswith("-logit_cosine.json"):
            continue
        try:
            data = load_json(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip (load failed): {filename}: {exc}")
            continue
        chin = data.get("chinchilla")
        if chin is None:
            m = re.search(r"chinchilla-(\d+)-", filename)
            chin = int(m.group(1)) if m else None
        if chin is None:
            print(f"  skip (no chinchilla): {filename}")
            continue
        rows.append({
            "chinchilla": int(chin),
            "mean_cosine": data["mean_cosine"],
            "std_cosine": data.get("std_cosine"),
            "median_cosine": data.get("median_cosine"),
            "p10": data.get("p10"),
            "p90": data.get("p90"),
            "num_tokens": data.get("num_tokens"),
            "histogram": data.get("histogram"),
            "adamw_run": data.get("adamw_run"),
            "muon_run": data.get("muon_run"),
            "path": path,
        })
        print(f"  ✓ chin={chin}: mean_cos={data['mean_cosine']:.4f}")
    rows.sort(key=lambda r: r["chinchilla"])
    return rows


def plot_mean_vs_chin(rows: List[dict], out_path: str) -> None:
    xs = [r["chinchilla"] for r in rows]
    ys = [r["mean_cosine"] for r in rows]
    yerr_lo = [
        r["mean_cosine"] - r["p10"] if r.get("p10") is not None else 0.0 for r in rows
    ]
    yerr_hi = [
        r["p90"] - r["mean_cosine"] if r.get("p90") is not None else 0.0 for r in rows
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        xs, ys, yerr=[yerr_lo, yerr_hi],
        fmt="-o", color="tab:green", markersize=5, linewidth=1.8,
        capsize=3, label="mean ± [p10, p90]",
    )
    meds = [r.get("median_cosine") for r in rows]
    if all(m is not None for m in meds):
        ax.plot(xs, meds, "--s", color="tab:purple", markersize=4, label="median")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("chinchilla", fontsize=12)
    ax.set_ylabel(r"cosine$(\ell_{\mathrm{adamw}}, \ell_{\mathrm{muon}})$", fontsize=12)
    ax.set_title("Per-token logit cosine: adamw vs muon", fontsize=13)
    ax.set_ylim(bottom=min(0.0, min(ys) - 0.05) if ys else 0.0, top=1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _cos_to_deg(cos: np.ndarray | float) -> np.ndarray | float:
    """Angle (degrees) between unit vectors with the given cosine similarity."""
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def plot_hists(rows: List[dict], out_path: str) -> None:
    n = len(rows)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    # Show θ corresponding to cosine ∈ [0.5, 1] → angle ∈ [0°, 60°].
    angle_max = float(_cos_to_deg(0.5))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8), squeeze=False
    )
    for i, r in enumerate(rows):
        rr, cc = divmod(i, ncols)
        ax = axs[rr][cc]
        hist = r.get("histogram") or {}
        edges = hist.get("bin_edges")
        counts = hist.get("counts")
        mean_deg = float(_cos_to_deg(r["mean_cosine"]))
        if edges and counts:
            edges_c = np.asarray(edges, dtype=float)
            counts_a = np.asarray(counts, dtype=float)
            # Keep bins overlapping cosine ≥ 0.5 (angle ≤ 60°).
            left_c, right_c = edges_c[:-1], edges_c[1:]
            keep = right_c >= 0.5
            left_c, right_c, counts_a = left_c[keep], right_c[keep], counts_a[keep]
            left_c = np.maximum(left_c, 0.5)
            # arccos is decreasing in cos → left_angle > right_angle
            left_a = _cos_to_deg(left_c)
            right_a = _cos_to_deg(right_c)
            # Plot with increasing angle on x: bar from right_a to left_a
            widths = left_a - right_a
            ax.bar(
                right_a, counts_a, width=widths * 0.95, align="edge",
                color="tab:green", alpha=0.75,
            )
        ax.axvline(mean_deg, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(
            f"chin={r['chinchilla']}  μ={mean_deg:.1f}°",
            fontsize=9,
        )
        ax.set_xlim(0.0, angle_max)
        ax.tick_params(labelsize=7)
        if rr == nrows - 1:
            ax.set_xlabel(r"angle $\theta=\arccos(\mathrm{cos})$ (deg)", fontsize=8)
        if cc == 0:
            ax.set_ylabel("count", fontsize=8)
    for j in range(n, nrows * ncols):
        rr, cc = divmod(j, ncols)
        axs[rr][cc].set_visible(False)
    fig.suptitle(
        r"Per-token logit angle $\theta$ between adamw and muon",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-dir", default=DEFAULT_GCS_DIR)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.local_dir:
        paths = list_local_files(args.local_dir)
        print(f"Using local dir {args.local_dir} ({len(paths)} files)")
    else:
        paths = list_gcs_files(args.gcs_dir)

    rows = collect(paths)
    if not rows:
        raise SystemExit("No LogitCosineEvaluation JSONs found.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "final_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    plot_mean_vs_chin(rows, os.path.join(args.out_dir, "mean_cosine_vs_chinchilla.png"))
    plot_hists(rows, os.path.join(args.out_dir, "cosine_hist_by_chin.png"))


if __name__ == "__main__":
    main()
