#!/usr/bin/env python3
"""Stage 2 of the 60M chinchilla-4 PT-sweep pipeline: learning/forgetting plots,
one pair of figures per SWEPT pretrain hyperparameter (LR, weight decay, batch).

Reads the local aggregate from preprocess_pt_sweep_evals.py (no GCS access) and,
for each hyperparameter that actually varies, writes:

  <out>/60M-<hp>-sweep-tradeoff-side-by-side.{png,pdf}
      AdamW left, Muon right, one Pareto frontier per value of that hparam.
  <out>/60M-<hp>-sweep-tradeoff-combined.{png,pdf}
      Every point on one axes, one frontier per optimizer.

Axes (same convention as plot_cpt_pareto.py):
  x = pretrain / forgetting loss   (DCLM held-out)
  y = fine-tuning loss             (the CPT dataset val loss)
Each point is one CPT learning rate.

ARM ISOLATION. The aggregate mixes the sweep stages, so naively grouping by
weight decay would compare cells that also differ in LR. For hparam H we keep
only records whose OTHER swept hparams are held constant — the largest such
group, chosen per optimizer — and the held-fixed values are printed in the
subtitle so the comparison is never ambiguous.

Usage:
  python -m new_utils.preprocess_pt_sweep_evals    # stage 1
  python -m new_utils.plot_pt_sweep                # stage 2: every varied hparam
  python -m new_utils.plot_pt_sweep --hparams lr   # just one
"""
import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Repo convention (README §7): muon = orange, adamw = green.
OPTIM_COLOR = {"adamw": "tab:green", "muon": "tab:orange"}
OPTIM_LABEL = {"adamw": "AdamW", "muon": "Muon"}
# Sequential ramp per optimizer: every swept hparam here is an ordered magnitude,
# so it gets one hue light->dark rather than arbitrary categorical hues.
OPTIM_CMAP = {"adamw": "Greens", "muon": "Oranges"}
# Distinct marker per optimizer so the combined plot never encodes identity by
# colour alone (green/orange is a weak pair under deuteranopia).
OPTIM_MARKER = {"adamw": "o", "muon": "^"}


def _bs_tokens(bs):
    """'512k' -> 524288, '1M' -> 1048576, so batch sizes sort by magnitude."""
    m = re.match(r"^([0-9]+)([kM]?)$", str(bs))
    if not m:
        return float("inf")
    n = int(m.group(1))
    return n * {"k": 1024, "M": 1024 * 1024, "": 1}[m.group(2)]


def _lr_tag(v):
    return f"{v:.1e}".replace("e-0", "e-")


# key -> (record accessor, axis/legend label, value formatter, sort key)
HPARAMS = {
    "lr": (lambda r: r["sweep_lr"], "Pretrain LR", _lr_tag, float),
    "wd": (lambda r: r["weight_decay"], "Weight decay", lambda v: f"{v:g}", float),
    "bs": (lambda r: r["batch_size"], "Batch size", str, _bs_tokens),
}

# Which OTHER hparams must be constant for a group to count as a clean arm.
# The batch-size sweep deliberately scales the LR with sqrt(batch) (7.0e-3 ->
# 9.9e-3 -> 1.4e-2 for adamw; 1.0e-2 -> 1.4e-2 -> 2.0e-2 for muon, each x sqrt2
# per doubling), so requiring a fixed LR there would find no arm at all. LR is
# therefore allowed to co-vary with batch, and the figure says so.
HOLD_FIXED = {
    "lr": ("wd", "bs"),
    "wd": ("lr", "bs"),
    "bs": ("wd",),
}

# Extra caveat printed under a figure where something co-varies by design.
HP_NOTE = {
    "bs": "LR scaled ∝ √batch (co-varies by design)",
}


def pareto_front(xs, ys):
    """Lower-left convex-hull arc, matching scatter_and_pareto in plot_cpt_pareto.py
    so these figures are comparable with the existing CPT Pareto plots."""
    if len(xs) <= 2:
        return None
    from scipy.spatial import ConvexHull
    pts = np.column_stack([xs, ys])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    h = np.append(hull.vertices, hull.vertices[0])
    hp = pts[h]
    hp = np.roll(hp, -int(np.argmin(hp[:, 0])), axis=0)
    for i in range(len(hp) - 1):
        if hp[i][0] > hp[i + 1][0]:
            hp = hp[: i + 1]
            break
    return hp


def _shades(opt, n):
    """n colours from this optimizer's hue, mid->dark (skip the near-white end,
    which is invisible on a white surface)."""
    cmap = plt.get_cmap(OPTIM_CMAP[opt])
    if n == 1:
        return [cmap(0.75)]
    return [cmap(v) for v in np.linspace(0.40, 0.92, n)]


def isolate_arm(records, opt, hp):
    """Records for `opt` where the OTHER swept hparams are constant.

    Returns (records, fixed_dict). Picks the largest such group so a partially
    finished sweep still plots. Returns ([], {}) when the hparam does not vary.
    """
    others = list(HOLD_FIXED[hp])
    groups = defaultdict(list)
    for r in records:
        if r["pretrain_optimizer"] != opt:
            continue
        key = tuple(HPARAMS[k][0](r) for k in others)
        groups[key].append(r)
    if not groups:
        return [], {}
    get = HPARAMS[hp][0]
    # Most distinct values of hp, tie-broken by record count.
    key = max(groups, key=lambda k: (len({get(r) for r in groups[k]}), len(groups[k])))
    arm = groups[key]
    if len({get(r) for r in arm}) < 2:
        return [], {}
    return arm, dict(zip(others, key))


def _points(arm, hp, pretrain_key):
    """value-of-hp -> [(x, y, cpt_lr)], skipping records with no x loss."""
    get = HPARAMS[hp][0]
    out = defaultdict(list)
    n_missing = 0
    for r in arm:
        x = (r.get("losses") or {}).get(pretrain_key)
        y = r.get("finetune_loss")
        if x is None or y is None:
            n_missing += 1
            continue
        out[get(r)].append((x, y, r["cpt_lr"]))
    return out, n_missing


def _fixed_str(fixed):
    if not fixed:
        return ""
    parts = []
    for k, v in fixed.items():
        fmt = HPARAMS[k][2]
        parts.append(f"{HPARAMS[k][1].lower()} {fmt(v) if not isinstance(v, str) else v}")
    return ", ".join(parts)


def plot_side_by_side(by_opt, hp, args, dataset):
    label, fmt, sortkey = HPARAMS[hp][1], HPARAMS[hp][2], HPARAMS[hp][3]
    opts = [o for o in ("adamw", "muon") if by_opt.get(o, (None, None))[0]]
    fig, axs = plt.subplots(1, max(1, len(opts)), figsize=(5.6 * max(1, len(opts)), 4.8),
                            sharex=True, sharey=True)
    axs = [axs] if len(opts) <= 1 else list(axs)

    fixed_bits = []
    for ax, opt in zip(axs, opts):
        pts_by_val, fixed = by_opt[opt]
        if fixed:
            fixed_bits.append(f"{OPTIM_LABEL[opt]}: {_fixed_str(fixed)}")
        vals = sorted(pts_by_val, key=sortkey)
        for v, c in zip(vals, _shades(opt, len(vals))):
            pts = pts_by_val[v]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, color=c, marker=OPTIM_MARKER[opt], s=42, alpha=0.9,
                       edgecolors="white", linewidths=0.6,
                       label=f"{fmt(v)}", zorder=3)
            hpf = pareto_front(xs, ys)
            if hpf is not None:
                ax.plot(hpf[:, 0], hpf[:, 1], color=c, linewidth=2, zorder=2)
            if args.annotate_lr:
                for x, y, clr in pts:
                    ax.text(x, y, f"{clr:.0e}", fontsize=5.5, ha="right",
                            va="bottom", color=c)
        ax.set_title(f"{OPTIM_LABEL[opt]} pretrain", fontsize=13)
        ax.set_xlabel(f"Pretrain loss ({args.pretrain_key})", fontsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, title=label, title_fontsize=8, frameon=False, loc="best")
    axs[0].set_ylabel(f"Fine-tuning loss ({dataset})", fontsize=12)

    fig.suptitle(f"60M chinchilla-4 PT sweep — {label.lower()} vs "
                 f"learning/forgetting tradeoff ({dataset})", fontsize=14)
    caption = []
    if fixed_bits:
        caption.append("held fixed — " + " | ".join(fixed_bits))
    if hp in HP_NOTE:
        caption.append(HP_NOTE[hp])
    if caption:
        fig.text(0.5, 0.005, "   •   ".join(caption),
                 ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    return fig


def plot_combined(by_opt, hp, args, dataset):
    label, fmt, sortkey = HPARAMS[hp][1], HPARAMS[hp][2], HPARAMS[hp][3]
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    handles = []
    for opt in ("adamw", "muon"):
        pts_by_val, _ = by_opt.get(opt, (None, None))
        if not pts_by_val:
            continue
        xs_all, ys_all = [], []
        vals = sorted(pts_by_val, key=sortkey)
        for v, c in zip(vals, _shades(opt, len(vals))):
            pts = pts_by_val[v]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xs_all += xs
            ys_all += ys
            ax.scatter(xs, ys, color=c, marker=OPTIM_MARKER[opt], s=36, alpha=0.85,
                       edgecolors="white", linewidths=0.5, zorder=3)
        hpf = pareto_front(xs_all, ys_all)
        if hpf is not None:
            ax.plot(hpf[:, 0], hpf[:, 1], color=OPTIM_COLOR[opt], linewidth=2.4, zorder=2)
        handles.append(Line2D([0], [0], color=OPTIM_COLOR[opt],
                              marker=OPTIM_MARKER[opt], linewidth=2.4, markersize=7,
                              label=f"{OPTIM_LABEL[opt]} (all {label.lower()}s)"))

    ax.set_xlabel(f"Pretrain loss ({args.pretrain_key})", fontsize=12)
    ax.set_ylabel(f"Fine-tuning loss ({dataset})", fontsize=12)
    ax.set_title(f"60M chinchilla-4 PT sweep — all points, {label.lower()} varied "
                 f"({dataset})", fontsize=13)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if handles:
        ax.legend(handles=handles, fontsize=10, frameon=False, loc="best")
    note = f"shade = {label.lower()} (light→dark)"
    if hp in HP_NOTE:
        note += f"\n{HP_NOTE[hp]}"
    ax.text(0.99, 0.01, note, transform=ax.transAxes, fontsize=7.5,
            ha="right", va="bottom", color="0.35")
    fig.tight_layout()
    return fig


def save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/pt_sweep_60m/pt_sweep_results.json")
    ap.add_argument("--pretrain-key", default="DCLM_heldout",
                    help="losses[] key for the x-axis (default DCLM_heldout).")
    ap.add_argument("--out-dir", default="colm-moss-latex/plots/60M")
    ap.add_argument("--hparams", nargs="*", default=None, choices=list(HPARAMS),
                    help="Which swept hparams to plot (default: every one that varies).")
    ap.add_argument("--annotate-lr", action="store_true",
                    help="Label each point with its CPT LR.")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"{args.results} not found — run "
              f"`python -m new_utils.preprocess_pt_sweep_evals` first.")
        return
    with open(args.results) as f:
        records = json.load(f)

    datasets = sorted({r["dataset"] for r in records})
    dataset = datasets[0] if len(datasets) == 1 else "+".join(datasets)
    wanted = args.hparams or list(HPARAMS)

    plotted = []
    for hp in wanted:
        by_opt, total, missing = {}, 0, 0
        for opt in ("adamw", "muon"):
            arm, fixed = isolate_arm(records, opt, hp)
            if not arm:
                continue
            pts, n_miss = _points(arm, hp, args.pretrain_key)
            missing += n_miss
            if not pts:
                continue
            by_opt[opt] = (pts, fixed)
            total += sum(len(v) for v in pts.values())
        if not by_opt:
            print(f"[{hp}] no sweep arm — {HPARAMS[hp][1].lower()} does not vary "
                  f"in {args.results}")
            continue
        print(f"[{hp}] {HPARAMS[hp][1]}: {total} point(s)"
              + (f", {missing} skipped (no '{args.pretrain_key}')" if missing else ""))
        for opt, (pts, fixed) in by_opt.items():
            vals = sorted(pts, key=HPARAMS[hp][3])
            print(f"    {OPTIM_LABEL[opt]}: {len(vals)} value(s) "
                  f"[{', '.join(HPARAMS[hp][2](v) for v in vals)}]"
                  + (f"  (fixed: {_fixed_str(fixed)})" if fixed else ""))
        save(plot_side_by_side(by_opt, hp, args, dataset),
             args.out_dir, f"60M-{hp}-sweep-tradeoff-side-by-side")
        save(plot_combined(by_opt, hp, args, dataset),
             args.out_dir, f"60M-{hp}-sweep-tradeoff-combined")
        plotted.append(hp)

    if not plotted:
        print("nothing plotted.")
    else:
        print(f"plotted: {', '.join(plotted)}")


if __name__ == "__main__":
    main()
