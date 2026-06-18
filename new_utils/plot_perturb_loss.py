#!/usr/bin/env python3
"""Plot perturbation strength λ (= γ, the Gaussian-noise scale std=γ/‖W‖_F)
on the x-axis vs pretrain validation loss on the y-axis, one line per model.

Reads the PerturbedModel ModelEvaluation JSONs from GCS:
  <eval-dir>/<base_model>_perturbed_<g>-eval.json   (overall.loss at γ=g)
and, if present, the unperturbed base eval as the λ=0 anchor:
  <eval-dir>/<base_model>-eval.json

Usage:
  python new_utils/plot_perturb_loss.py \
    --model adamw=MuonEpochExpt-0.06B-chinchilla-0.025-8ep-adamw-lr1.0e-3-wsd \
    --model muon=MuonEpochExpt-0.06B-chinchilla-0.025-8ep-muon-muonlr5.0e-3-adamwlr1.0e-2-wsd \
    --out results/plots/qa_group/perturb_loss_vs_lambda.png
"""
import argparse
import json
import os
import re
import subprocess
import matplotlib.pyplot as plt

DEFAULT_EVAL_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/ModelEvaluation"
_LABEL_COLORS = {"adamw": "tab:green", "muon": "tab:orange", "gd": "tab:blue"}


def _gsutil_ls(d):
    out = subprocess.run(["gsutil", "ls", d], capture_output=True, text=True)
    return [l.strip() for l in out.stdout.splitlines() if l.strip().endswith(".json")]


def _gsutil_cat(path):
    return json.loads(subprocess.check_output(["gsutil", "cat", path], text=True))


def _parse_gamma(filename: str) -> float | None:
    """'..._perturbed_2_00e-2-eval.json' -> 0.02; base '...-eval.json' -> 0.0."""
    m = re.search(r"_perturbed_([0-9_.eE+\-]+)-eval\.json$", filename)
    if not m:
        return 0.0 if filename.endswith("-eval.json") else None
    return float(m.group(1).replace("_", "."))


def _loss_of(data: dict, key: str) -> float | None:
    if key == "overall":
        return data.get("overall", {}).get("loss")
    return data.get("by_label", {}).get(key, {}).get("loss")


def collect(eval_dir: str, base_model: str, loss_key: str) -> list[tuple[float, float]]:
    """Return sorted [(gamma, loss), ...] for one base model."""
    pts = {}
    for f in _gsutil_ls(eval_dir):
        name = os.path.basename(f)
        # match "<base_model>-eval.json" or "<base_model>_perturbed_<g>-eval.json"
        if not (name == f"{base_model}-eval.json"
                or name.startswith(f"{base_model}_perturbed_")):
            continue
        g = _parse_gamma(name)
        if g is None:
            continue
        loss = _loss_of(_gsutil_cat(f), loss_key)
        if loss is None:
            print(f"  [warn] no '{loss_key}' loss in {name}")
            continue
        pts[g] = float(loss)
    return sorted(pts.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="label=base_model_name (repeatable).")
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    ap.add_argument("--loss-key", default="overall",
                    help="'overall' or a by_label dataset (e.g. C4_val).")
    ap.add_argument("--title", default="Perturbation vs Pretrain Loss")
    ap.add_argument("--out", default="results/plots/qa_group/perturb_loss_vs_lambda.png")
    ap.add_argument("--max-lambda", type=float, default=None,
                    help="Drop perturbation points with λ (γ) above this "
                         "(the λ=0 base point is always kept).")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for spec in args.model:
        label, base_model = spec.split("=", 1)
        series = collect(args.eval_dir, base_model, args.loss_key)
        if args.max_lambda is not None:
            series = [(g, v) for g, v in series if g == 0.0 or g <= args.max_lambda]
        if not series:
            print(f"[warn] no eval points for {label} ({base_model})")
            continue
        xs = [g for g, _ in series]
        ys = [l for _, l in series]
        ax.plot(xs, ys, marker="o", label=label,
                color=_LABEL_COLORS.get(label.lower()), linewidth=2, markersize=6)
        print(f"{label}: {len(series)} points  λ={xs}")

    ax.set_xlabel(r"perturbation $\lambda$  ($\Vert\mathrm{noise}\Vert_F / \Vert W\Vert_F$)",
                  fontsize=12)
    ax.set_ylabel("pretrain loss", fontsize=12)
    ax.set_title(args.title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
