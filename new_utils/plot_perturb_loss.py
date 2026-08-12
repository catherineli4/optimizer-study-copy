#!/usr/bin/env python3
"""Plot perturbation strength λ (= γ, the Gaussian-noise scale std=γ/‖W‖_F)
on the x-axis vs held-out DCLM validation loss on the y-axis.

Like ``plot_pretrain_eval.py``, this AUTO-DISCOVERS every perturbed-model eval
JSON in the GCS ModelEvaluation dir and groups them by (optimizer, Chinchilla),
drawing one line per base model — so it covers the whole optimal-model sweep
without hand-listing run names. Pass ``--model label=run_name`` to restrict the
plot to specific base models instead.

Reads the PerturbedModel ModelEvaluation JSONs from GCS:
  <eval-dir>/<base_model>_perturbed_<g>-eval.json   (by_label.DCLM_heldout.loss at γ=g)
and, if present, the unperturbed base eval as the λ=0 anchor:
  <eval-dir>/<base_model>-eval.json

The DCLM held-out loss lives under by_label["DCLM_heldout"], folded into the eval
JSON by extra_val_chunks (see launch_jolmo/perturb.py). Pass --loss-key overall
(or a diversity set like C4_val) to plot a different val set instead.

Usage:
  # all optimal models across Chinchilla (auto-discovered), DCLM held-out loss:
  python -m new_utils.plot_perturb_loss

  # restrict to specific base models:
  python -m new_utils.plot_perturb_loss \
    --model muon=MuonEpochExpt-0.06B-chinchilla-32-muon-... \
    --model adamw=MuonEpochExpt-0.06B-chinchilla-32-adamw-...
"""
import argparse
import json
import os
import re
import subprocess
import matplotlib.pyplot as plt

DEFAULT_EVAL_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/ModelEvaluation"
# Base color per optimizer (muon = orange, adamw = green); individual Chinchilla
# lines are shaded along the optimizer's colormap so the sweep stays readable.
_OPT_BASE_COLORS = {"adamw": "tab:green", "muon": "tab:orange", "gd": "tab:blue"}
_OPT_CMAPS = {"adamw": "Greens", "muon": "Oranges", "gd": "Blues"}


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


def _parse_chinchilla(base_model: str):
    m = re.search(r"-chinchilla-([0-9.]+)-", base_model)
    if not m:
        return None
    v = float(m.group(1))
    return int(v) if v.is_integer() else v


def _parse_optimizer(base_model: str):
    m = re.search(r"-chinchilla-[0-9.]+-(adamw|muon)-", base_model)
    return m.group(1) if m else None


def _loss_of(data: dict, key: str) -> float | None:
    if key == "overall":
        return data.get("overall", {}).get("loss")
    return data.get("by_label", {}).get(key, {}).get("loss")


def discover_base_models(eval_dir: str) -> set[str]:
    """Every base model that has at least one ``_perturbed_`` eval in eval_dir."""
    bases = set()
    for f in _gsutil_ls(eval_dir):
        name = os.path.basename(f)
        m = re.match(r"^(.*)_perturbed_[0-9_.eE+\-]+-eval\.json$", name)
        if m:
            bases.add(m.group(1))
    return bases


def collect(files: list[str], base_model: str, loss_key: str,
            allowed_gammas=None) -> list[tuple[float, float]]:
    """Return sorted [(gamma, loss), ...] for one base model from a file listing.

    If `allowed_gammas` is given, only those γ (plus the γ=0 base anchor) are
    read — so stale runs at γ values outside the current grid are skipped, rather
    than warned about for a missing loss key."""
    allowed = None if allowed_gammas is None else {round(g, 12) for g in allowed_gammas}
    pts = {}
    for f in files:
        name = os.path.basename(f)
        # match "<base_model>-eval.json" or "<base_model>_perturbed_<g>-eval.json"
        if not (name == f"{base_model}-eval.json"
                or name.startswith(f"{base_model}_perturbed_")):
            continue
        g = _parse_gamma(name)
        if g is None:
            continue
        if allowed is not None and g != 0.0 and round(g, 12) not in allowed:
            continue  # stale γ outside the current grid
        loss = _loss_of(_gsutil_cat(f), loss_key)
        if loss is None:
            print(f"  [warn] no '{loss_key}' loss in {name}")
            continue
        pts[g] = float(loss)
    return sorted(pts.items())


def models_from_config():
    """Canonical (base_model, optimizer, chinchilla) + expected γ grid from the
    experiment definitions — so the plot targets exactly the defined sweep
    (e.g. 16 base models × len(DEFAULT_GAMMAS) γ) rather than whatever stale runs
    happen to sit in the GCS dir."""
    from launch_jolmo import pretraining_matrix as pm
    from launch_jolmo.perturb import DEFAULT_GAMMAS

    bases = {}  # base_model run_name -> (optimizer, chinchilla)
    for m in list(pm.perturbed_adamw_models) + list(pm.perturbed_muon_models):
        bm = m.source_model.run_name
        bases[bm] = (_parse_optimizer(bm), _parse_chinchilla(bm))
    return bases, list(DEFAULT_GAMMAS)


def _line_color(optimizer: str | None, chin, all_chins):
    """Shade an optimizer's colormap by where this Chinchilla sits in the sweep."""
    cmap_name = _OPT_CMAPS.get((optimizer or "").lower())
    if cmap_name is None or chin is None or len(all_chins) <= 1:
        return _OPT_BASE_COLORS.get((optimizer or "").lower())
    ordered = sorted(all_chins)
    frac = ordered.index(chin) / max(1, len(ordered) - 1)
    return plt.get_cmap(cmap_name)(0.4 + 0.55 * frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None,
                    help="label=base_model_name (repeatable). Overrides the default "
                         "config-driven model list.")
    ap.add_argument("--discover", action="store_true",
                    help="Ignore the experiment config and instead plot EVERY "
                         "perturbed base model found in --eval-dir (includes stale runs).")
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    ap.add_argument("--loss-key", default="DCLM_heldout",
                    help="by_label dataset to plot (default: DCLM_heldout, the "
                         "held-out DCLM split). Use 'overall' or e.g. C4_val for others.")
    ap.add_argument("--title", default="Perturbation vs DCLM Held-out Loss")
    ap.add_argument("--out", default="results/perturb/perturb_loss_vs_lambda.png")
    ap.add_argument("--max-lambda", type=float, default=None,
                    help="Drop perturbation points with λ (γ) above this "
                         "(the λ=0 base point is always kept).")
    args = ap.parse_args()

    # Resolve the set of base models to plot, plus the γ grid we EXPECT per model.
    expected_gammas = None
    if args.model:
        bases = {}
        for spec in args.model:
            label, base_model = spec.split("=", 1)
            bases[base_model] = (_parse_optimizer(base_model), _parse_chinchilla(base_model))
    elif args.discover:
        found = discover_base_models(args.eval_dir)
        print(f"[--discover] {len(found)} perturbed base model(s) in {args.eval_dir} "
              "(may include stale runs)")
        bases = {b: (_parse_optimizer(b), _parse_chinchilla(b)) for b in found}
    else:
        bases, expected_gammas = models_from_config()
        n_exp = len(bases) * len(expected_gammas)
        print(f"[config] {len(bases)} base models × {len(expected_gammas)} γ "
              f"= {n_exp} expected perturbed models")

    # Build specs: (label, base_model, optimizer, chinchilla), sorted opt then chinchilla.
    specs = []
    for base_model, (opt, chin) in bases.items():
        label = f"{opt or '?'} c={chin}" if chin is not None else (opt or base_model)
        specs.append((label, base_model, opt, chin))
    specs.sort(key=lambda s: ((s[2] or "z"), s[3] if s[3] is not None else 1e9))

    # List the eval dir ONCE and reuse for every model.
    all_files = _gsutil_ls(args.eval_dir.rstrip("/") + "/")
    all_chins = sorted({chin for *_, chin in specs if chin is not None})

    # One subplot per optimizer (adamw | muon); each line is a Chinchilla cell.
    panel_opts = ["adamw", "muon"]
    fig, axes = plt.subplots(1, len(panel_opts), figsize=(7 * len(panel_opts), 6),
                             sharey=True)
    axes = list(axes)
    panels = dict(zip(panel_opts, axes))

    found_perturbed = 0          # perturbed eval JSONs actually plotted (γ>0)
    for label, base_model, opt, chin in specs:
        ax = panels.get((opt or "").lower())
        if ax is None:  # optimizer outside the adamw/muon panels — skip
            print(f"[warn] no panel for optimizer '{opt}' ({base_model})")
            continue
        series = collect(all_files, base_model, args.loss_key, allowed_gammas=expected_gammas)
        if args.max_lambda is not None:
            series = [(g, v) for g, v in series if g == 0.0 or g <= args.max_lambda]
        if not series:
            print(f"[warn] no eval points for {label} ({base_model})")
            continue
        xs = [g for g, _ in series]
        ys = [l for _, l in series]
        n_pert = sum(1 for g in xs if g > 0.0)
        found_perturbed += n_pert
        cl = f"c={chin}" if chin is not None else label
        ax.plot(xs, ys, marker="o", label=cl,
                color=_line_color(opt, chin, all_chins), linewidth=2, markersize=6)
        # If we know the expected γ grid, flag any missing cells for this model.
        if expected_gammas is not None:
            missing = sorted(g for g in expected_gammas if g not in dict(series))
            tag = f"  [MISSING {len(missing)}: {missing}]" if missing else ""
        else:
            tag = ""
        print(f"{opt} {cl}: {n_pert} perturbed pts  λ={[g for g in xs if g>0]}{tag}")

    if expected_gammas is not None:
        n_exp = len(specs) * len(expected_gammas)
        status = "✓ all found" if found_perturbed == n_exp else "⚠ INCOMPLETE"
        print(f"\n{status}: {found_perturbed}/{n_exp} perturbed models located on GCS")

    xlabel = r"perturbation $\lambda$  ($\Vert\mathrm{noise}\Vert_F / \Vert W\Vert_F$)"
    for opt, ax in panels.items():
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(opt, fontsize=13)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(title="Chinchilla", fontsize=9)
    axes[0].set_ylabel(f"{args.loss_key} loss", fontsize=12)
    fig.suptitle(args.title, fontsize=14)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
