#!/usr/bin/env python3
"""Plot pretrain ModelEvaluation metrics as a function of Chinchilla multiplier.

Reads the per-model ``…-eval.json`` validation-loss files written by the
``eval-pretrain-*`` stages straight from GCS (no GPU), groups them by optimizer
(AdamW vs Muon) and Chinchilla token budget, and draws one figure per eval
metric: x = Chinchilla multiplier (log2), y = validation loss.

Convention here (per request): **adamw = blue, muon = orange**.

Run after the eval stage, e.g.::

    python -m launch_jolmo.launcher launch eval-pretrain-adamw eval-pretrain-muon
    python -m new_utils.plot_pretrain_eval     # writes results/pretrain-eval/*.png

Only the *pretrain* evals are used: files containing ``_perturbed_``, ``-CPT-``
or ``-typo`` (the CPT / perturbation / typo families) are skipped.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

# adamw = blue, muon = orange (markers/linestyle also differ for B&W legibility).
OPT_STYLE = {
    "adamw": {"color": "tab:blue",   "marker": "o", "linestyle": "-"},
    "muon":  {"color": "tab:orange", "marker": "s", "linestyle": "-"},
}

DEFAULT_GCS_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/ModelEvaluation"
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "pretrain-eval",
)


def list_gcs_files(gcs_dir: str) -> List[str]:
    """List every ``.json`` directly under a GCS directory."""
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


def load_json_from_gcs(gcs_path: str) -> dict:
    """Download one JSON file from GCS and parse it."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name
    try:
        subprocess.check_call(
            ["gsutil", "cp", gcs_path, tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(tmp_path, "r") as f:
            return json.load(f)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def is_pretrain_eval(filename: str) -> bool:
    """True only for base pretrain evals (not CPT / perturbed / typo variants)."""
    lowered = filename.lower()
    if any(tag in lowered for tag in ("_perturbed_", "-cpt-", "typo")):
        return False
    return bool(re.search(r"-chinchilla-[0-9.]+-(adamw|muon)-", filename))


def parse_chinchilla(filename: str) -> Optional[float]:
    """Extract the Chinchilla multiplier from ``…-chinchilla-<N>-…``."""
    match = re.search(r"-chinchilla-([0-9.]+)-", filename)
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def parse_optimizer(filename: str) -> Optional[str]:
    """Return 'adamw' or 'muon' from the run name."""
    match = re.search(r"-chinchilla-[0-9.]+-(adamw|muon)-", filename)
    return match.group(1) if match else None


def metric_value(data: dict, label: str) -> Optional[float]:
    """Pull a loss for ``label`` ('overall' or a by_label key) from an eval JSON."""
    if label == "overall":
        return (data.get("overall") or {}).get("loss")
    entry = (data.get("by_label") or {}).get(label)
    return entry.get("loss") if isinstance(entry, dict) else None


def collect(gcs_dir: str) -> Dict[str, Dict[str, Dict[float, float]]]:
    """metric -> optimizer -> {chinchilla: loss}.

    When several LR runs share an (optimizer, chinchilla) cell (e.g. the
    eval-pretrain-all sweep), keep the best (lowest) loss.
    """
    files = [f for f in list_gcs_files(gcs_dir) if is_pretrain_eval(os.path.basename(f))]
    print(f"Found {len(files)} pretrain eval JSON(s) in {gcs_dir}")

    # metric -> opt -> chinchilla -> loss
    out: Dict[str, Dict[str, Dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
    for gcs_file in files:
        filename = os.path.basename(gcs_file)
        chinchilla = parse_chinchilla(filename)
        optimizer = parse_optimizer(filename)
        if chinchilla is None or optimizer is None:
            print(f"  skip (unparsable): {filename}")
            continue
        try:
            data = load_json_from_gcs(gcs_file)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip (download/parse failed): {filename}: {exc}")
            continue

        labels = ["overall"] + list((data.get("by_label") or {}).keys())
        for label in labels:
            value = metric_value(data, label)
            if value is None:
                continue
            cell = out[label][optimizer]
            if chinchilla not in cell or value < cell[chinchilla]:
                cell[chinchilla] = value
        print(f"  ✓ {optimizer:5s} chinchilla={chinchilla}: {filename}")
    return out


def plot_metric(
    metric: str,
    by_opt: Dict[str, Dict[float, float]],
    out_path: str,
) -> None:
    """Draw one metric vs Chinchilla, a line per optimizer, and save it."""
    fig, ax = plt.subplots(figsize=(7, 5))

    all_x = set()
    for optimizer in sorted(by_opt):  # adamw before muon
        series = by_opt[optimizer]
        if not series:
            continue
        xs = sorted(series.keys())
        ys = [series[x] for x in xs]
        all_x.update(xs)
        style = OPT_STYLE.get(optimizer, {"color": "gray", "marker": "x", "linestyle": "--"})
        ax.plot(
            xs, ys,
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
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlabel("Chinchilla multiplier", fontsize=12)
    ax.set_ylabel(f"Validation loss ({metric})", fontsize=12)
    ax.set_title(f"Pretrain eval: {metric} vs Chinchilla", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Optimizer")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-dir", default=DEFAULT_GCS_DIR,
                        help="GCS dir holding the ModelEvaluation JSON files.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Local directory for the PNGs.")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Subset of metrics to plot (default: all found, "
                             "e.g. overall C4_val Reddit_val Wiki_val Books_val DCLM_heldout).")
    args = parser.parse_args()

    data = collect(args.gcs_dir)
    if not data:
        print("No pretrain eval data found; nothing to plot.")
        return

    metrics = args.metrics or sorted(data.keys())
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nWriting plots to {args.out_dir}")
    for metric in metrics:
        if metric not in data:
            print(f"  (no data for metric '{metric}', skipping)")
            continue
        out_path = os.path.join(args.out_dir, f"pretrain-eval-{metric}.png")
        plot_metric(metric, data[metric], out_path)


if __name__ == "__main__":
    main()
