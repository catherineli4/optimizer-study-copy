#!/usr/bin/env python3
"""Download / process / plot Jolmo SharpnessEvaluation results.

Reads ``…-pretrain-sharpness-max_eigenvalue_sum_eigenvalue.json`` artifacts
written by ``sharpness-all`` (and friends), parses adamw/muon + chinchilla from
the run name, writes a consolidated ``final_results.json``, and plots:

  * max Hessian eigenvalue vs Chinchilla
  * Hessian trace (sum of eigenvalues) vs Chinchilla

Style matches ``plot_pretrain_eval`` (adamw=blue, muon=orange).

Examples::

    # Direct from GCS
    python -m new_utils.plot_jolmo_sharpness

    # After a local rsync (sbatch path)
    python -m new_utils.plot_jolmo_sharpness --local-dir results/SharpnessEvaluation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

OPT_STYLE = {
    "adamw": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
    "muon": {"color": "tab:orange", "marker": "s", "linestyle": "-"},
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GCS_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/SharpnessEvaluation"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "sharpness")
DEFAULT_LOCAL_DIR = os.path.join(REPO_ROOT, "results", "SharpnessEvaluation")

SHARPNESS_SUFFIX = "-pretrain-sharpness-max_eigenvalue_sum_eigenvalue.json"


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


def is_pretrain_sharpness(filename: str) -> bool:
    return filename.endswith(SHARPNESS_SUFFIX) and "-CPT-" not in filename


def parse_chinchilla(filename: str) -> Optional[float]:
    match = re.search(r"-chinchilla-([0-9.]+)-", filename)
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def parse_optimizer(filename: str) -> Optional[str]:
    match = re.search(r"-chinchilla-[0-9.]+-(adamw|muon)-", filename)
    return match.group(1) if match else None


def parse_model_type(filename: str) -> Optional[str]:
    match = re.search(r"-(\d+\.\d+B)-chinchilla-", filename)
    return match.group(1) if match else None


def collect(paths: List[str]) -> Tuple[List[dict], Dict[str, Dict[str, Dict[float, float]]]]:
    """Return (rows, metric -> optimizer -> {chinchilla: value})."""
    rows: List[dict] = []
    by_metric: Dict[str, Dict[str, Dict[float, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    files = [p for p in paths if is_pretrain_sharpness(os.path.basename(p))]
    print(f"Found {len(files)} pretrain sharpness JSON(s)")

    for path in files:
        filename = os.path.basename(path)
        chinchilla = parse_chinchilla(filename)
        optimizer = parse_optimizer(filename)
        model_type = parse_model_type(filename)
        if chinchilla is None or optimizer is None:
            print(f"  skip (unparsable): {filename}")
            continue
        try:
            data = load_json(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip (load failed): {filename}: {exc}")
            continue

        max_eig = data.get("max_eigenvalue_sharpness")
        trace = data.get("sum_eigenvalue_sharpness")
        row = {
            "run_name": filename.replace(SHARPNESS_SUFFIX, ""),
            "filename": filename,
            "model_type": model_type,
            "optimizer": optimizer,
            "chinchilla": chinchilla,
            # tokens/param ≈ 20 × chinchilla under CHINCHILLA_MULT=20
            "tokens_per_param": 20.0 * float(chinchilla),
            "max_eigenvalue_sharpness": max_eig,
            "sum_eigenvalue_sharpness": trace,
            "path": path,
        }
        rows.append(row)

        if max_eig is not None:
            by_metric["max_eigenvalue"][optimizer][chinchilla] = float(max_eig)
        if trace is not None:
            by_metric["trace"][optimizer][chinchilla] = float(trace)

        print(
            f"  ✓ {optimizer:5s} chinchilla={chinchilla}: "
            f"λmax={max_eig}  tr={trace}"
        )

    rows.sort(key=lambda r: (r["optimizer"], r["chinchilla"]))
    return rows, by_metric


def plot_metric(
    metric: str,
    ylabel: str,
    by_opt: Dict[str, Dict[float, float]],
    out_path: str,
    *,
    x_as_tokens_per_param: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    all_x = set()

    for optimizer in ("adamw", "muon"):
        series = by_opt.get(optimizer) or {}
        if not series:
            continue
        xs_raw = sorted(series.keys())
        if x_as_tokens_per_param:
            xs = [20.0 * x for x in xs_raw]
            ys = [series[x] for x in xs_raw]
        else:
            xs = xs_raw
            ys = [series[x] for x in xs]
        all_x.update(xs)
        style = OPT_STYLE.get(optimizer, {"color": "gray", "marker": "x", "linestyle": "--"})
        ax.plot(
            xs,
            ys,
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
    xticks = sorted(all_x)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) if float(x).is_integer() else str(x) for x in xticks])
    if x_as_tokens_per_param:
        ax.set_xlabel("Tokens / Param", fontsize=12)
    else:
        ax.set_xlabel("Chinchilla multiplier", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Pretrain sharpness: {metric}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Optimizer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gcs-dir",
        default=DEFAULT_GCS_DIR,
        help="GCS SharpnessEvaluation dir (ignored if --local-dir is set).",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Local dir of sharpness JSONs (skip GCS listing if provided).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Where to write final_results.json and PNGs.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["chinchilla", "tokens_per_param"],
        default="chinchilla",
        help="X-axis for the plots (default: chinchilla).",
    )
    args = parser.parse_args()

    if args.local_dir:
        paths = list_local_files(args.local_dir)
        print(f"Using local dir {args.local_dir} ({len(paths)} json file(s))")
    else:
        paths = list_gcs_files(args.gcs_dir)

    rows, by_metric = collect(paths)
    os.makedirs(args.out_dir, exist_ok=True)

    results_path = os.path.join(args.out_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {results_path} ({len(rows)} rows)")

    x_tpp = args.x_axis == "tokens_per_param"
    plot_metric(
        "max_eigenvalue",
        "Max Hessian Eigenvalue",
        by_metric.get("max_eigenvalue", {}),
        os.path.join(args.out_dir, "max_eigenvalue.png"),
        x_as_tokens_per_param=x_tpp,
    )
    plot_metric(
        "trace",
        "Hessian Trace (sum of eigenvalues)",
        by_metric.get("trace", {}),
        os.path.join(args.out_dir, "trace.png"),
        x_as_tokens_per_param=x_tpp,
    )


if __name__ == "__main__":
    main()
