#!/usr/bin/env python3
"""Download / plot SharpnessEvaluation max-eigenvalue vs training step.

One figure per chinchilla; AdamW and Muon overlaid. Plots raw λ_max(H) and
(when present) preconditioned λ_max(T∘H).

Examples::

    # rsync then plot
    mkdir -p results/SharpnessEvaluation
    gsutil -m rsync -r \\
      gs://cmu-gpucloud-catheri4/Optim-60M-tuning/SharpnessEvaluation/ \\
      results/SharpnessEvaluation/
    python -m new_utils.plot_jolmo_sharpness_vs_step \\
      --local-dir results/SharpnessEvaluation

    # stream from GCS (slower)
    python -m new_utils.plot_jolmo_sharpness_vs_step
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

OPT_STYLE = {
    "adamw": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
    "muon": {"color": "tab:orange", "marker": "s", "linestyle": "-"},
}
OPT_LABEL = {"adamw": "AdamW", "muon": "Muon"}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GCS_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/SharpnessEvaluation"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "sharpness_vs_step")
DEFAULT_LOCAL_DIR = os.path.join(REPO_ROOT, "results", "SharpnessEvaluation")

# maxeig artifacts: …-pretrain[-stepN]-sharpness-max_eigenvalue.json
# Exclude sum_eigenvalue / spectral_density variants.
_MAXEIG_RE = re.compile(
    r"-pretrain(?:-step(\d+))?-sharpness-max_eigenvalue\.json$"
)


def list_gcs_files(gcs_dir: str) -> List[str]:
    result = subprocess.run(
        ["gsutil", "ls", gcs_dir.rstrip("/") + "/"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".json")]


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


def parse_chinchilla(filename: str) -> Optional[float]:
    match = re.search(r"-chinchilla-([0-9.]+)-", filename)
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def parse_optimizer(filename: str) -> Optional[str]:
    match = re.search(r"-chinchilla-[0-9.]+-(adamw|muon)-", filename)
    return match.group(1) if match else None


def parse_step_from_name(filename: str) -> Optional[int]:
    match = _MAXEIG_RE.search(filename)
    if not match:
        return None
    if match.group(1) is not None:
        return int(match.group(1))
    return None  # final (no -stepN- in name)


def is_maxeig_file(filename: str) -> bool:
    return bool(_MAXEIG_RE.search(filename)) and "-CPT-" not in filename


def _resolve_step(data: dict, filename: str) -> Optional[int]:
    """Numeric training step; ``final`` left as None for a later pass."""
    raw = data.get("step", data.get("checkpoint"))
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw == int(raw):
        return int(raw)
    if isinstance(raw, str):
        if raw == "final":
            return None
        if raw.startswith("step") and raw[4:].isdigit():
            return int(raw[4:])
        if raw.isdigit():
            return int(raw)
    return parse_step_from_name(filename)


def collect(paths: List[str]) -> List[dict]:
    rows: List[dict] = []
    files = [p for p in paths if is_maxeig_file(os.path.basename(p))]
    print(f"Found {len(files)} max_eigenvalue JSON(s)")

    for path in files:
        filename = os.path.basename(path)
        chin = parse_chinchilla(filename)
        opt = parse_optimizer(filename)
        if chin is None or opt is None:
            print(f"  skip (unparsable): {filename}")
            continue
        try:
            data = load_json(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip (load failed): {filename}: {exc}")
            continue

        step = _resolve_step(data, filename)
        raw = data.get("max_eigenvalue_sharpness")
        prec = data.get("max_eigenvalue_preconditioned_sharpness")
        if raw is None and prec is None:
            print(f"  skip (no metrics): {filename}")
            continue

        rows.append(
            {
                "filename": filename,
                "chinchilla": chin,
                "optimizer": opt,
                "step": step,
                "is_final": step is None,
                "max_eigenvalue_sharpness": float(raw) if raw is not None else None,
                "max_eigenvalue_preconditioned_sharpness": (
                    float(prec) if prec is not None else None
                ),
            }
        )

    # Map final → max numeric step for the same (chin, opt), else keep as-is.
    max_step: Dict[Tuple[Any, str], int] = {}
    for r in rows:
        if r["step"] is not None:
            key = (r["chinchilla"], r["optimizer"])
            max_step[key] = max(max_step.get(key, 0), int(r["step"]))
    for r in rows:
        if r["step"] is None:
            key = (r["chinchilla"], r["optimizer"])
            if key in max_step:
                # Place final just after the last intermediate if that step
                # already has a point; otherwise at max_step.
                r["step"] = max_step[key]
            else:
                print(
                    f"  warn: final with no other steps for "
                    f"chin={r['chinchilla']} {r['optimizer']}; dropping"
                )
                r["step"] = None

    rows = [r for r in rows if r["step"] is not None]
    rows.sort(key=lambda r: (r["chinchilla"], r["optimizer"], r["step"]))
    return rows


def plot_chinchilla(
    chin: Any,
    rows: List[dict],
    metric_key: str,
    ylabel: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for opt in ("adamw", "muon"):
        series = [
            (r["step"], r[metric_key])
            for r in rows
            if r["optimizer"] == opt
            and r["chinchilla"] == chin
            and r.get(metric_key) is not None
        ]
        if not series:
            continue
        # Dedup by step (prefer non-final overwrite order already sorted).
        by_step: Dict[int, float] = {}
        for step, val in series:
            by_step[int(step)] = val
        xs = sorted(by_step)
        ys = [by_step[x] for x in xs]
        style = OPT_STYLE[opt]
        ax.plot(
            xs,
            ys,
            label=OPT_LABEL[opt],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Chinchilla-{chin}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.replace(".png", ".pdf"))
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
        help="Local dir of downloaded JSONs (skip GCS listing).",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.local_dir:
        paths = list_local_files(args.local_dir)
    else:
        paths = list_gcs_files(args.gcs_dir)

    rows = collect(paths)
    chins = sorted({r["chinchilla"] for r in rows})
    print(f"Chinchillas: {chins}")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "final_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    for chin in chins:
        plot_chinchilla(
            chin,
            rows,
            "max_eigenvalue_sharpness",
            r"$\lambda_{\max}(H)$",
            os.path.join(args.out_dir, f"chinchilla-{chin}-max_eigenvalue.png"),
        )
        plot_chinchilla(
            chin,
            rows,
            "max_eigenvalue_preconditioned_sharpness",
            r"$\lambda_{\max}(T\!\circ\!H)$",
            os.path.join(args.out_dir, f"chinchilla-{chin}-max_eigenvalue_preconditioned.png"),
        )


if __name__ == "__main__":
    main()
