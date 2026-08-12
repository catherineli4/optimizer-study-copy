#!/usr/bin/env python3
"""Plot Gaussian logit-perturb curves for each adamw↔muon angle-bin data group.

One plot set per angle bin (same style as ``plot_logit_perturb``): mean / ΔNLL
vs σ with adamw vs muon, optionally xmax=1e-1. Chinchillas are subplots.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

OPT_STYLE = {
    "adamw": {"color": "tab:blue", "marker": "o"},
    "muon": {"color": "tab:orange", "marker": "s"},
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GCS_DIR = (
    "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/LogitAngleBinPerturbEvaluation"
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "logit_angle_perturb")


def list_files(gcs_dir: Optional[str], local_dir: Optional[str]) -> List[str]:
    if local_dir:
        return sorted(
            os.path.join(local_dir, n)
            for n in os.listdir(local_dir)
            if n.endswith("-angle_bin_perturb.json")
        )
    result = subprocess.run(
        ["gsutil", "ls", gcs_dir.rstrip("/") + "/"],
        capture_output=True, text=True, check=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().endswith("-angle_bin_perturb.json")
    ]


def load_json(path: str) -> dict:
    if path.startswith("gs://"):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name
        try:
            subprocess.check_call(
                ["gsutil", "cp", path, tmp_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with open(tmp_path) as f:
                return json.load(f)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    with open(path) as f:
        return json.load(f)


def collect(paths: List[str]) -> List[dict]:
    rows = []
    for path in paths:
        data = load_json(path)
        chin = data.get("chinchilla")
        if chin is None:
            m = re.search(r"chinchilla-(\d+)-", os.path.basename(path))
            chin = int(m.group(1)) if m else None
        print(f"  ✓ chin={chin} tokens={data.get('num_tokens_total')}")
        rows.append(data)
    rows.sort(key=lambda d: d.get("chinchilla") or 0)
    return rows


def _delta(mean_nll: List[float], baseline: float) -> np.ndarray:
    return np.asarray(mean_nll, dtype=float) - float(baseline)


def plot_one_bin(
    rows: List[dict],
    bin_label: str,
    out_dir: str,
    *,
    degradation: bool,
    xmax: Optional[float],
    markersize: float = 1.0,
) -> None:
    # Keep chins that have this bin with tokens
    usable = []
    for r in rows:
        b = (r.get("bins") or {}).get(bin_label) or {}
        if b.get("num_tokens", 0) > 0 and "adamw" in b and "muon" in b:
            usable.append(r)
    if not usable:
        return

    n = len(usable)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0), squeeze=False
    )
    for i, r in enumerate(usable):
        rr, cc = divmod(i, ncols)
        ax = axs[rr][cc]
        chin = r["chinchilla"]
        b = r["bins"][bin_label]
        sigmas = np.asarray(r["sigmas"], dtype=float)
        for opt in ("adamw", "muon"):
            style = OPT_STYLE[opt]
            ys = np.asarray(b[opt]["mean_nll"], dtype=float)
            if degradation:
                ys = _delta(ys, b[opt]["baseline_mean_nll"])
            xs = sigmas
            if xmax is not None:
                mask = xs <= xmax + 1e-15
                xs, ys = xs[mask], ys[mask]
            ax.plot(
                xs, ys,
                label=opt,
                color=style["color"],
                marker=style["marker"],
                markersize=markersize,
                linewidth=1.6,
            )
            std = b[opt].get("std_nll_across_dirs")
            if std is not None:
                ss = np.asarray(std, dtype=float)
                if xmax is not None:
                    ss = ss[mask]
                ax.fill_between(
                    xs, ys - ss, ys + ss,
                    color=style["color"], alpha=0.15, linewidth=0,
                )
        ax.set_xscale("log")
        if xmax is not None:
            pos = [s for s in sigmas if s > 0 and (xmax is None or s <= xmax)]
            if pos:
                ax.set_xlim(left=min(pos), right=xmax)
        if degradation:
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
        ax.set_title(
            f"chin={chin}  n={b['num_tokens']}", fontsize=10
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=8)
        if rr == nrows - 1:
            ax.set_xlabel(r"σ", fontsize=9)
        if cc == 0:
            ax.set_ylabel(
                r"$\Delta$ NLL (post−pre)" if degradation else "mean NLL",
                fontsize=9,
            )
        ax.legend(fontsize=8)

    for j in range(n, nrows * ncols):
        rr, cc = divmod(j, ncols)
        axs[rr][cc].set_visible(False)

    kind = r"$\Delta$NLL" if degradation else "NLL"
    fig.suptitle(
        f"Gaussian logit perturb on angle-bin θ∈[{bin_label}]°  ({kind})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    tag = "delta_nll" if degradation else "mean_nll"
    suffix = f"_xmax{xmax:g}".replace(".", "p") if xmax is not None else ""
    # sanitize bin label for filename
    safe = bin_label.replace("/", "-")
    path = os.path.join(out_dir, f"bin_{safe}", f"{tag}_vs_sigma_by_chin{suffix}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-dir", default=DEFAULT_GCS_DIR)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    paths = list_files(args.gcs_dir if not args.local_dir else None, args.local_dir)
    if not paths:
        raise SystemExit("No angle_bin_perturb JSON files found.")
    print(f"Loading {len(paths)} runs")
    rows = collect(paths)

    # Discover bin labels that appear with data
    bin_labels = sorted(
        {
            lab
            for r in rows
            for lab, b in (r.get("bins") or {}).items()
            if b.get("num_tokens", 0) > 0
        },
        key=lambda s: int(s.split("-")[0]),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "final_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    for lab in bin_labels:
        plot_one_bin(rows, lab, args.out_dir, degradation=False, xmax=None, markersize=3)
        plot_one_bin(rows, lab, args.out_dir, degradation=True, xmax=None, markersize=3)
        plot_one_bin(
            rows, lab, args.out_dir, degradation=True, xmax=1e-1, markersize=1.0
        )
        plot_one_bin(
            rows, lab, args.out_dir, degradation=False, xmax=1e-1, markersize=1.0
        )


if __name__ == "__main__":
    main()
