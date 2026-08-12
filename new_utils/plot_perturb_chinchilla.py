#!/usr/bin/env python3
"""Plot Chinchilla multiplier (x) vs held-out DCLM validation loss (y), with one
line per perturbation amount λ (=γ). Two subplots: adamw | muon.

This is the transpose of ``plot_perturb_loss.py`` (which fixes the model and
sweeps λ on the x-axis). Here λ is held fixed per line and the x-axis is the
Chinchilla token budget, so each curve shows how a given perturbation strength's
DCLM loss scales with pretraining compute. λ=0 is the unperturbed base model.

Like ``plot_perturb_loss.py`` it is config-driven: the (optimizer, Chinchilla)
base models and the expected γ grid come from the experiment definitions
(launch_jolmo/perturb.py + pretraining_matrix.py), so stale runs in the GCS
ModelEvaluation dir are ignored. The DCLM held-out loss is read from
by_label["DCLM_heldout"] (folded in by extra_val_chunks); pass --loss-key to use
a different val set.

Usage:
  python -m new_utils.plot_perturb_chinchilla
  python -m new_utils.plot_perturb_chinchilla --loss-key overall
"""
import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt

# Reuse the GCS/parse helpers and config-driven model list from the sibling plot.
from new_utils.plot_perturb_loss import (
    DEFAULT_EVAL_DIR,
    _gsutil_ls,
    collect,
    models_from_config,
)


def _gamma_label(g: float) -> str:
    return "0 (base)" if g == 0.0 else f"λ={g:.0e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    ap.add_argument("--loss-key", default="DCLM_heldout",
                    help="by_label dataset to plot (default: DCLM_heldout). "
                         "Use 'overall' or e.g. C4_val for others.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-base", action="store_true",
                    help="Omit the λ=0 (unperturbed) reference line.")
    ap.add_argument("--delta", action="store_true",
                    help="Plot Δloss = loss(λ) − loss(base) on the y-axis (the "
                         "DCLM-loss degradation from perturbation) instead of raw loss.")
    args = ap.parse_args()

    if args.title is None:
        args.title = (f"Chinchilla vs Δ DCLM Held-out Loss (vs base, per λ)" if args.delta
                      else "Chinchilla vs DCLM Held-out Loss (per perturbation λ)")
    if args.out is None:
        args.out = ("results/perturb/perturb_delta_loss_vs_chinchilla.png" if args.delta
                    else "results/perturb/perturb_loss_vs_chinchilla.png")

    bases, expected_gammas = models_from_config()
    print(f"[config] {len(bases)} base models, {len(expected_gammas)} γ "
          f"({len(bases) * len(expected_gammas)} expected perturbed models)")

    all_files = _gsutil_ls(args.eval_dir.rstrip("/") + "/")

    # data[optimizer][gamma][chinchilla] = loss
    data = defaultdict(lambda: defaultdict(dict))
    found = 0
    for base_model, (opt, chin) in bases.items():
        if opt is None or chin is None:
            print(f"[warn] unparsable base model, skipping: {base_model}")
            continue
        for g, loss in collect(all_files, base_model, args.loss_key,
                               allowed_gammas=expected_gammas):
            data[opt][g][chin] = loss
            if g > 0.0:
                found += 1
    n_exp = len(bases) * len(expected_gammas)
    status = "✓ all found" if found == n_exp else "⚠ INCOMPLETE"
    print(f"{status}: {found}/{n_exp} perturbed models located on GCS")

    # One subplot per optimizer; one line per γ, shaded by γ magnitude.
    panel_opts = ["adamw", "muon"]
    fig, axes = plt.subplots(1, len(panel_opts), figsize=(7 * len(panel_opts), 6),
                             sharey=True)
    axes = list(axes)
    gammas_sorted = ([] if args.no_base else [0.0]) + sorted(expected_gammas)
    cmap = plt.get_cmap("viridis")

    all_chins = set()
    for opt, ax in zip(panel_opts, axes):
        by_gamma = data.get(opt, {})
        base = by_gamma.get(0.0, {})  # {chinchilla: base loss} for Δ mode
        for i, g in enumerate(gammas_sorted):
            series = by_gamma.get(g, {})
            if not series:
                continue
            if args.delta:
                # Δloss vs the unperturbed base at the same Chinchilla; drop the
                # γ=0 line (it is zero by definition) and any cell with no base.
                if g == 0.0:
                    continue
                xs = sorted(x for x in series if x in base)
                ys = [series[x] - base[x] for x in xs]
            else:
                xs = sorted(series)
                ys = [series[x] for x in xs]
            if not xs:
                continue
            all_chins.update(xs)
            if g == 0.0:
                color, lw, ls, marker = "black", 2.5, "--", "o"
            else:
                frac = i / max(1, len(gammas_sorted) - 1)
                color, lw, ls, marker = cmap(frac), 2, "-", "o"
            ax.plot(xs, ys, marker=marker, label=_gamma_label(g),
                    color=color, linewidth=lw, linestyle=ls, markersize=5)
        if args.delta:
            ax.axhline(0.0, color="black", linewidth=1, linestyle=":", alpha=0.6)
        ax.set_title(opt, fontsize=13)
        ax.set_xlabel("Chinchilla multiplier", fontsize=12)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

    if all_chins:
        xticks = sorted(all_chins)
        for ax in axes:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])
    ylabel = (f"Δ {args.loss_key} loss  (λ − base)" if args.delta
              else f"{args.loss_key} loss")
    axes[0].set_ylabel(ylabel, fontsize=12)
    if axes[-1].get_legend_handles_labels()[0]:
        axes[-1].legend(title="perturbation", fontsize=8, ncol=2)
    fig.suptitle(args.title, fontsize=14)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
