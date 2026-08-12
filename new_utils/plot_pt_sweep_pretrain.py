#!/usr/bin/env python3
"""Plot the PRETRAIN evals of the 60M chinchilla-4 sweep (no finetuning).

A pretrain eval yields one loss per model, not a tradeoff curve, so this is a
loss-vs-hyperparameter plot rather than the Pareto figures in plot_pt_sweep.py:

  <out>/60M-<hp>-sweep-pretrain-loss.{png,pdf}
      x = the swept hyperparameter, y = pretrain loss, one line per optimizer.

Arm isolation, the hparam table, and the colour/marker conventions are imported
from plot_pt_sweep so the two scripts cannot drift apart.

Usage:
  python -m new_utils.preprocess_pt_sweep_evals --kind pretrain
  python -m new_utils.plot_pt_sweep_pretrain
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from new_utils.plot_pt_sweep import (
    HPARAMS, HP_NOTE, OPTIM_COLOR, OPTIM_LABEL, OPTIM_MARKER,
    isolate_arm, _fixed_str, save,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="results/pt_sweep_60m/pt_sweep_pretrain_results.json")
    ap.add_argument("--loss-key", default="DCLM_heldout",
                    help="losses[] key to plot (default DCLM_heldout).")
    ap.add_argument("--out-dir", default="colm-moss-latex/plots/60M")
    ap.add_argument("--hparams", nargs="*", default=None, choices=list(HPARAMS))
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"{args.results} not found — run "
              f"`python -m new_utils.preprocess_pt_sweep_evals --kind pretrain` first.")
        return
    with open(args.results) as f:
        records = json.load(f)
    print(f"{len(records)} pretrain eval record(s)")

    plotted = []
    for hp in (args.hparams or list(HPARAMS)):
        get, label, fmt, sortkey = HPARAMS[hp]
        series = {}
        for opt in ("adamw", "muon"):
            arm, fixed = isolate_arm(records, opt, hp)
            if not arm:
                continue
            pts = []
            for r in arm:
                y = (r.get("losses") or {}).get(args.loss_key)
                if y is not None:
                    pts.append((get(r), y))
            if len(pts) >= 2:
                series[opt] = (sorted(pts, key=lambda t: sortkey(t[0])), fixed)
        if not series:
            print(f"[{hp}] no arm with >=2 evaluated points — skipping")
            continue

        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        fixed_bits = []
        # One shared, SORTED category axis across both series. Passing raw string
        # x-values per series instead makes matplotlib order categories by first
        # appearance — e.g. 0.1, 0.3, 0.4, 0, 0.2 — which draws fake zigzags.
        all_vals = sorted({p[0] for pts, _ in series.values() for p in pts},
                          key=sortkey)
        pos = {v: i for i, v in enumerate(all_vals)}
        for opt, (pts, fixed) in series.items():
            xs = [pos[p[0]] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=OPTIM_COLOR[opt], marker=OPTIM_MARKER[opt],
                    markersize=7, linewidth=2, label=OPTIM_LABEL[opt],
                    markeredgecolor="white", markeredgewidth=0.6)
            n = len(pts)
            fixed_bits.append(f"{OPTIM_LABEL[opt]}: {_fixed_str(fixed)} (n={n})"
                              if fixed else f"{OPTIM_LABEL[opt]} (n={n})")

        ax.set_xticks(range(len(all_vals)))
        ax.set_xticklabels([fmt(v) for v in all_vals])
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel(f"Pretrain loss ({args.loss_key})", fontsize=12)
        ax.set_title(f"60M chinchilla-4 PT sweep — pretrain loss vs {label.lower()}",
                     fontsize=13)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=10, frameon=False)
        caption = ["held fixed — " + " | ".join(fixed_bits)] if fixed_bits else []
        if hp in HP_NOTE:
            caption.append(HP_NOTE[hp])
        if caption:
            fig.text(0.5, -0.02, "   •   ".join(caption), ha="center",
                     fontsize=8, color="0.35")
        fig.tight_layout()
        print(f"[{hp}] {label}: " + ", ".join(
            f"{OPTIM_LABEL[o]} {len(v[0])} pt(s)" for o, v in series.items()))
        save(fig, args.out_dir, f"60M-{hp}-sweep-pretrain-loss")
        plotted.append(hp)

    print(f"plotted: {', '.join(plotted) if plotted else '(nothing)'}")


if __name__ == "__main__":
    main()
