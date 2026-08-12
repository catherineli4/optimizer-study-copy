#!/usr/bin/env python3
"""Plot mean KL(Q‖P) vs σ from LogitPerturbKlEvaluation JSONs.

Reads ``…/LogitPerturbKlEvaluation/*-logit_perturb_kl.json`` (local or GCS),
parses chinchilla + optimizer from the run name, and writes absolute + ΔKL
curves (same layout as ``plot_logit_perturb``).

Q = OLMo-2 1B teacher, P = (perturbed) student; primary metric is kl_forward.
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
    "adamw": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
    "muon": {"color": "tab:orange", "marker": "s", "linestyle": "-"},
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GCS_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/LogitPerturbKlEvaluation"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "logit_perturb_kl")
DEFAULT_LOCAL_DIR = os.path.join(REPO_ROOT, "results", "LogitPerturbKlEvaluation")


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


def parse_chinchilla(name: str) -> Optional[float]:
    m = re.search(r"-chinchilla-([0-9.]+)-", name)
    if not m:
        return None
    v = float(m.group(1))
    return int(v) if v.is_integer() else v


def parse_optimizer(name: str) -> Optional[str]:
    m = re.search(r"-chinchilla-[0-9.]+-(adamw|muon)-", name)
    return m.group(1) if m else None


def collect(paths: List[str]) -> List[dict]:
    rows = []
    for path in paths:
        filename = os.path.basename(path)
        if not filename.endswith("-logit_perturb_kl.json"):
            continue
        run = filename.replace("-logit_perturb_kl.json", "")
        opt = parse_optimizer(run)
        chin = parse_chinchilla(run)
        if opt is None or chin is None:
            print(f"  skip (unparsable): {filename}")
            continue
        try:
            data = load_json(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip (load failed): {filename}: {exc}")
            continue
        rows.append({
            "run_name": run,
            "optimizer": opt,
            "chinchilla": chin,
            "sigmas": data["sigmas"],
            "mean_kl": data["mean_kl_forward"],
            "std_kl_across_dirs": data.get("std_kl_forward_across_dirs"),
            "baseline_mean_kl": data.get("baseline_mean_kl_forward"),
            "num_tokens": data.get("num_tokens"),
            "path": path,
        })
        print(
            f"  ✓ {opt:5s} chin={chin}: "
            f"baseline_kl_fwd={data.get('baseline_mean_kl_forward')}"
        )
    rows.sort(key=lambda r: (r["optimizer"], r["chinchilla"]))
    return rows


def plot_all_curves(
    rows: List[dict],
    out_path: str,
    *,
    degradation: bool = False,
    xmax: Optional[float] = None,
    markersize: float = 4.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        style = OPT_STYLE.get(
            r["optimizer"], {"color": "gray", "marker": "x", "linestyle": "--"}
        )
        xs = np.asarray(r["sigmas"], dtype=float)
        ys = np.asarray(r["mean_kl"], dtype=float)
        if degradation:
            base = float(
                r["baseline_mean_kl"]
                if r.get("baseline_mean_kl") is not None
                else ys[0]
            )
            ys = ys - base
        if xmax is not None:
            mask = xs <= xmax + 1e-15
            xs, ys = xs[mask], ys[mask]
        ax.plot(
            xs,
            ys,
            label=f"{r['optimizer']} chin={r['chinchilla']}",
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            markersize=markersize,
            alpha=0.85,
        )
    ax.set_xscale("log")
    if xmax is not None:
        pos = [
            float(s)
            for r in rows
            for s in r["sigmas"]
            if float(s) > 0 and float(s) <= xmax
        ]
        if pos:
            ax.set_xlim(left=min(pos), right=xmax)
    ax.set_xlabel(r"σ  ($\Vert\Delta\ell\Vert_2 / \Vert\ell\Vert_2$)", fontsize=12)
    if degradation:
        ax.set_ylabel(r"$\Delta$ KL($Q\|P$)  (post − pre)", fontsize=12)
        ax.set_title(r"Logit perturbation: $\Delta$KL vs 1B vs σ", fontsize=13)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    else:
        ax.set_ylabel(r"mean KL($Q\|P$)  (Q=1B)", fontsize=12)
        ax.set_title("Logit perturbation: KL vs 1B vs σ", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_by_chinchilla(
    rows: List[dict],
    out_path: str,
    *,
    xmax: Optional[float] = None,
    markersize: float = 5.0,
    degradation: bool = False,
) -> None:
    chins = sorted({r["chinchilla"] for r in rows})
    n = len(chins)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0), squeeze=False
    )
    by_chin: Dict[float, List[dict]] = defaultdict(list)
    for r in rows:
        by_chin[r["chinchilla"]].append(r)

    for i, chin in enumerate(chins):
        r, c = divmod(i, ncols)
        ax = axs[r][c]
        for row in by_chin[chin]:
            style = OPT_STYLE.get(row["optimizer"], {})
            xs = np.asarray(row["sigmas"], dtype=float)
            ys = np.asarray(row["mean_kl"], dtype=float)
            if degradation:
                base = float(
                    row["baseline_mean_kl"]
                    if row.get("baseline_mean_kl") is not None
                    else ys[0]
                )
                ys = ys - base
            if xmax is not None:
                mask = xs <= xmax + 1e-15
                xs, ys = xs[mask], ys[mask]
            else:
                mask = None
            ax.plot(
                xs,
                ys,
                label=row["optimizer"],
                color=style.get("color", "gray"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=1.8,
                markersize=markersize,
            )
            std = row.get("std_kl_across_dirs")
            if std is not None:
                ss = np.asarray(std, dtype=float)
                if mask is not None:
                    ss = ss[mask]
                ax.fill_between(
                    xs, ys - ss, ys + ss,
                    color=style.get("color", "gray"), alpha=0.15, linewidth=0,
                )
        ax.set_xscale("log")
        if xmax is not None:
            pos = [
                float(s)
                for row in by_chin[chin]
                for s in row["sigmas"]
                if float(s) > 0 and float(s) <= xmax
            ]
            if pos:
                ax.set_xlim(left=min(pos), right=xmax)
        if degradation:
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
        ax.set_title(f"chinchilla={chin}", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=8)
        if r == nrows - 1:
            ax.set_xlabel(r"σ", fontsize=9)
        if c == 0:
            ax.set_ylabel(
                r"$\Delta$ KL (post−pre)" if degradation else r"mean KL($Q\|P$)",
                fontsize=9,
            )
        ax.legend(fontsize=8)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axs[r][c].set_visible(False)

    title = (
        r"Logit perturbation $\Delta$KL($Q\|P$) vs 1B (adamw vs muon)"
        if degradation
        else r"Logit perturbation KL($Q\|P$) vs 1B (adamw vs muon)"
    )
    fig.suptitle(title, fontsize=12)
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
        raise SystemExit("No LogitPerturbKlEvaluation JSONs found.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "final_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    plot_all_curves(rows, os.path.join(args.out_dir, "mean_kl_vs_sigma.png"))
    plot_by_chinchilla(
        rows, os.path.join(args.out_dir, "mean_kl_vs_sigma_by_chin.png")
    )
    plot_by_chinchilla(
        rows,
        os.path.join(args.out_dir, "mean_kl_vs_sigma_by_chin_xmax1e-1.png"),
        xmax=1e-1,
        markersize=1.0,
    )

    plot_all_curves(
        rows,
        os.path.join(args.out_dir, "delta_kl_vs_sigma.png"),
        degradation=True,
    )
    plot_by_chinchilla(
        rows,
        os.path.join(args.out_dir, "delta_kl_vs_sigma_by_chin.png"),
        degradation=True,
    )
    plot_by_chinchilla(
        rows,
        os.path.join(args.out_dir, "delta_kl_vs_sigma_by_chin_xmax1e-1.png"),
        xmax=1e-1,
        markersize=1.0,
        degradation=True,
    )


if __name__ == "__main__":
    main()
