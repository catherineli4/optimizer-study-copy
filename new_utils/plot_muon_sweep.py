#!/usr/bin/env python3
"""Muon alpha-sweep plot: one sweep figure showing all alpha variants together.

alpha = the muon→adamw LR ratio (adamw_component_lr = alpha · muon_lr) swept by
the `muon-sweep` experiment. For the muon CPT runs this plots:

  x = CPT muon learning rate   (log)
  y = loss  (DCLM held-out by default; --loss-key for finetune/overall/…)
  one line per alpha            (all variants overlaid in the same panel)

Reads the LOCAL aggregate written by preprocess_cpt_evals.py (cpt_results.json),
so run that first (it now records `alpha` / `cpt_muon_lr` for muon runs):

  python -m new_utils.preprocess_cpt_evals      # download + aggregate
  python -m new_utils.plot_muon_sweep           # plot from local file

One subplot per (dataset × Chinchilla); with a single active CPT dataset and one
Chinchilla that's a single panel with one line per alpha.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

# Distinct color per alpha (sweep grid is small).
_ALPHA_COLORS = {0.15: "tab:blue", 0.25: "tab:orange", 0.35: "tab:green", 0.45: "tab:red"}


def _snap_alpha(alpha):
    """Snap a measured alpha to the nearest 0.x5 target (0.15/0.25/0.35/0.45/…).

    The swept alphas all end in 5; small deviations (e.g. 0.26, 0.34, 0.46) are
    LR-rounding artifacts — fold them into the closest target rather than dropping
    them or showing a spurious extra curve."""
    return round(round((alpha - 0.05) / 0.1) * 0.1 + 0.05, 2)


def _alpha_color(alpha, all_alphas):
    if alpha in _ALPHA_COLORS:
        return _ALPHA_COLORS[alpha]
    ordered = sorted(all_alphas)
    frac = ordered.index(alpha) / max(1, len(ordered) - 1)
    return plt.get_cmap("viridis")(frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/cpt/cpt_results.json",
                    help="Aggregated file from preprocess_cpt_evals.py.")
    ap.add_argument("--loss-key", default="DCLM_heldout",
                    help="losses[] key for the y-axis (default DCLM_heldout; "
                         "use overall/C4_val, or 'finetune' for the CPT val loss).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset of CPT datasets (default: all muon runs in results).")
    ap.add_argument("--out-dir", default="results/cpt/muon_sweep")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"{args.results} not found — run `python -m new_utils.preprocess_cpt_evals` first.")
        return
    with open(args.results) as f:
        records = json.load(f)

    def y_of(r):
        return r["finetune_loss"] if args.loss_key == "finetune" else r["losses"].get(args.loss_key)

    # dataset -> chinchilla -> alpha -> list of (muon_lr, y)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n = 0
    for r in records:
        if r.get("cpt_optimizer") != "muon" or r.get("alpha") is None:
            continue
        if args.datasets and r["dataset"] not in args.datasets:
            continue
        x = r.get("cpt_muon_lr")
        y = y_of(r)
        if x is None or y is None:
            continue
        data[r["dataset"]][r["chinchilla"]][_snap_alpha(r["alpha"])].append((x, y))
        n += 1

    if not data:
        print(f"No muon-sweep records with '{args.loss_key}'. Run eval-muon-sweep + "
              f"re-run preprocess_cpt_evals (and check the aggregate has 'alpha').")
        return
    all_alphas = sorted({a for ds in data.values() for ch in ds.values() for a in ch})
    print(f"{n} muon record(s); alphas = {all_alphas}; datasets = {sorted(data)}")
    os.makedirs(args.out_dir, exist_ok=True)

    for dataset in sorted(data):
        by_chin = data[dataset]
        chins = sorted(by_chin)
        fig, axs = plt.subplots(1, len(chins), figsize=(max(1, len(chins)) * 5.5, 4.5),
                                sharey=True, squeeze=False)
        axs = list(axs[0])
        for col, (chin, ax) in enumerate(zip(chins, axs)):
            for alpha in sorted(by_chin[chin]):
                pts = sorted(by_chin[chin][alpha])          # by muon_lr
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, marker="o", label=f"α={alpha:g}",
                        color=_alpha_color(alpha, all_alphas), linewidth=2, markersize=6)
            ax.set_xscale("log")
            ax.set_title(f"Chinchilla = {chin}", fontsize=12)
            ax.set_xlabel("CPT muon LR", fontsize=11)
            if col == 0:
                ylab = "finetune loss" if args.loss_key == "finetune" else f"{args.loss_key} loss"
                ax.set_ylabel(ylab, fontsize=12)
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(title="muon→adamw ratio", fontsize=9)
        fig.suptitle(f"Muon alpha sweep — CPT {dataset}", fontsize=14)
        out = os.path.join(args.out_dir, f"muon-sweep-{dataset}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
