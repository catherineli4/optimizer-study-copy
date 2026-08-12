#!/usr/bin/env python3
"""Stage 2 of the CPT Pareto pipeline: plot from the LOCAL aggregated results
(no GCS access). Same plot as catastrophic-forgetting/plotting/plot_pareto.py
(make_tradeoff_pareto_optim_token):

  x = pretrain / forgetting loss   (DCLM held-out by default)
  y = fine-tuning loss             (the CPT dataset val loss)
  scatter over all CPT learning rates + a convex-hull Pareto front per optimizer
  (adamw vs muon); one subplot per pretrain Chinchilla, one figure per dataset.

Run `preprocess_cpt_evals.py` first to download + aggregate the eval JSONs into
results/cpt/cpt_results.json, then this reads that file offline.

Usage:
  python -m new_utils.preprocess_cpt_evals       # stage 1: download + aggregate
  python -m new_utils.plot_cpt_pareto            # stage 2: plot (x = DCLM_heldout)
  python -m new_utils.plot_cpt_pareto --pretrain-key C4_val   # before DCLM re-run
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

# muon = orange, adamw = green (repo convention; see README §7).
COLOR_MAP = {"adamw": "tab:green", "muon": "tab:orange"}
OPTIM_MAP = {"adamw": "AdamW", "muon": "Muon"}


def scatter_and_pareto(ax, xs, ys, label, color):
    """Scatter points and draw the lower-left convex-hull Pareto front."""
    if not xs:
        return
    ax.scatter(xs, ys, label=label, marker="o", color=color, s=30, alpha=0.8)
    if len(xs) <= 2:
        return
    pts = np.column_stack([xs, ys])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return
    h = np.append(hull.vertices, hull.vertices[0])
    hp = pts[h]
    # Rotate so the smallest-x vertex is first, keep the increasing-x lower arc.
    hp = np.roll(hp, -int(np.argmin(hp[:, 0])), axis=0)
    for i in range(len(hp) - 1):
        if hp[i][0] > hp[i + 1][0]:
            hp = hp[: i + 1]
            break
    ax.plot(hp[:, 0], hp[:, 1], color=color, linewidth=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/cpt/cpt_results.json",
                    help="Aggregated file from preprocess_cpt_evals.py.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset of CPT datasets to plot (default: all in results).")
    ap.add_argument("--pretrain-key", default="DCLM_heldout",
                    help="losses[] key for the x-axis forgetting loss "
                         "(default DCLM_heldout; use C4_val/overall before DCLM re-run).")
    ap.add_argument("--group-by", default="pretrain", choices=["pretrain", "cpt"],
                    help="Which optimizer defines the two fronts on each plot: "
                         "'pretrain' (the base model's optimizer — the Muon-vs-AdamW "
                         "study axis, default) or 'cpt' (the finetune optimizer).")
    ap.add_argument("--annotate-lr", action="store_true",
                    help="Label each scatter point with its CPT LR.")
    ap.add_argument("--out-dir", default="results/cpt/pareto")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"{args.results} not found — run `python -m new_utils.preprocess_cpt_evals` first.")
        return
    with open(args.results) as f:
        records = json.load(f)

    opt_field = "pretrain_optimizer" if args.group_by == "pretrain" else "cpt_optimizer"
    other_field = "cpt_optimizer" if args.group_by == "pretrain" else "pretrain_optimizer"
    # dataset -> chinchilla -> optimizer -> list of (x_pretrain, y_finetune, cpt_lr)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    other_opts = defaultdict(set)  # dataset -> set of the *other* optimizer (for context)
    n_missing_x = 0
    for r in records:
        if args.datasets and r["dataset"] not in args.datasets:
            continue
        x = r["losses"].get(args.pretrain_key)
        y = r["finetune_loss"]
        if x is None or y is None:
            n_missing_x += 1
            continue
        data[r["dataset"]][r["chinchilla"]][r[opt_field]].append((x, y, r["cpt_lr"]))
        other_opts[r["dataset"]].add(r[other_field])

    if not data:
        msg = f"No plottable records for pretrain-key='{args.pretrain_key}'."
        if args.pretrain_key == "DCLM_heldout":
            msg += (" The aggregate has no DCLM_heldout — re-run eval-cpt (so the "
                    "JSONs get the DCLM split), re-run preprocess, or use "
                    "--pretrain-key C4_val.")
        print(msg)
        return
    if n_missing_x:
        print(f"[note] {n_missing_x} record(s) skipped (no '{args.pretrain_key}' loss)")
    print(f"plotting {len(data)} dataset(s): {', '.join(sorted(data))}")
    os.makedirs(args.out_dir, exist_ok=True)

    for dataset in sorted(data):
        by_chin = data[dataset]
        chins = sorted(by_chin)
        fig, axs = plt.subplots(1, len(chins), figsize=(max(1, len(chins)) * 5, 4.5),
                                sharey=True, sharex=True)
        axs = [axs] if len(chins) == 1 else list(axs)

        for col, (chin, ax) in enumerate(zip(chins, axs)):
            for opt in ("adamw", "muon"):
                pts = by_chin[chin].get(opt, [])
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                lbl = f"{OPTIM_MAP[opt]} {args.group_by}"
                scatter_and_pareto(ax, xs, ys, lbl, COLOR_MAP[opt])
                if args.annotate_lr:
                    for x, y, lr in pts:
                        ax.text(x, y, f"{lr:.0e}", fontsize=6, ha="right",
                                va="bottom", color=COLOR_MAP[opt])
            ax.set_title(f"Chinchilla = {chin}", fontsize=13)
            ax.set_xlabel(f"Pretrain loss ({args.pretrain_key})", fontsize=11)
            if col == 0:
                ax.set_ylabel("Fine-tuning loss", fontsize=12)
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=9)

        # Note the *other* (held-fixed) optimizer so the comparison is unambiguous,
        # e.g. group_by=pretrain → "finetune: adamw" when all CPT runs used adamw.
        other = sorted(other_opts.get(dataset, []))
        ctx = f"  ({other_field.split('_')[0]}: {', '.join(other)})" if other else ""
        fig.suptitle(f"CPT {dataset}: fine-tuning vs pretrain-loss tradeoff{ctx}",
                     fontsize=14)
        out = os.path.join(args.out_dir, f"cpt-pareto-{dataset}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
