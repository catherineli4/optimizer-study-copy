#!/usr/bin/env python3
"""Plot perturbation strength λ (= γ, std=γ/‖W‖_F) vs model metrics, one line
per model. Reads two GCS sources for each (base) model and its perturbations:

  pretrain loss        <eval-dir>/<name>[_perturbed_<g>]-eval.json  -> overall.loss
  fta / acc / ce       <qa-dir>/<name>[_perturbed_<g>]-qa.json      -> metrics.overall.*

Produces, per metric: a raw "<metric> vs λ" plot and a "degradation vs λ" plot
(value(λ) − value(λ=0)), the latter only when the λ=0 base point exists.

Usage:
  python new_utils/plot_perturb_metrics.py \
    --model adamw=MuonEpochExpt-0.06B-chinchilla-0.025-8ep-adamw-lr1.0e-3-wsd \
    --model muon=MuonEpochExpt-0.06B-chinchilla-0.025-8ep-muon-muonlr5.0e-3-adamwlr1.0e-2-wsd \
    --out-dir results/plots/qa_group/perturb
"""
import argparse
import json
import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt

EVAL_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/ModelEvaluation"
QA_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/QAGroupEvaluation"
_COLORS = {"adamw": "tab:green", "muon": "tab:orange", "gd": "tab:blue"}

# metric key -> (source, json-path, y-label)
METRICS = {
    "loss": ("eval", ("overall", "loss"), "pretrain loss"),
    "fta":  ("qa", ("metrics", "overall", "fta"), "first-token accuracy"),
    "acc":  ("qa", ("metrics", "overall", "acc"), "accuracy"),
}


def _ls(d):
    out = subprocess.run(["gsutil", "ls", d], capture_output=True, text=True)
    return [l.strip() for l in out.stdout.splitlines() if l.strip().endswith(".json")]


def _cat(path):
    return json.loads(subprocess.check_output(["gsutil", "cat", path], text=True))


def _gamma(name: str, suffix: str):
    m = re.search(r"_perturbed_([0-9_.eE+\-]+)" + re.escape(suffix) + r"$", name)
    if m:
        return float(m.group(1).replace("_", "."))
    return 0.0 if name.endswith(suffix) else None


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _matching_files(base_model: str, d: str, suffix: str):
    """Yield (gamma, gs_path) for the base model and all its perturbations."""
    for f in _ls(d):
        name = os.path.basename(f)
        if not (name == f"{base_model}{suffix}"
                or name.startswith(f"{base_model}_perturbed_")):
            continue
        g = _gamma(name, suffix)
        if g is None:
            continue
        yield g, f


def collect(base_model: str, source: str, jpath, eval_dir, qa_dir):
    """Return {gamma: value} for one model + metric."""
    d, suffix = (eval_dir, "-eval.json") if source == "eval" else (qa_dir, "-qa.json")
    pts = {}
    for g, f in _matching_files(base_model, d, suffix):
        v = _dig(_cat(f), jpath)
        if v is not None:
            pts[g] = float(v)
    return pts


def collect_qa_raw(base_model: str, qa_dir: str):
    """Return {gamma: metrics_dict} for one model — the whole ``metrics`` block
    (overall + per-group) catted once per file, for per-group plotting."""
    out = {}
    for g, f in _matching_files(base_model, qa_dir, "-qa.json"):
        out[g] = _cat(f).get("metrics", {})
    return out


def _plot(models, metric, degradation, out_path, xscale="symlog"):
    source, jpath, ylabel = METRICS[metric]
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    pos_lams = []   # positive λ values, for the symlog linear-threshold
    for label, series in models:
        if not series:
            continue
        if degradation:
            if 0.0 not in series:
                print(f"  [skip degradation {metric} {label}: no λ=0 base point]")
                continue
            base = series[0.0]
            items = sorted((g, v - base) for g, v in series.items())
        else:
            items = sorted(series.items())
        xs = [g for g, _ in items]
        ys = [v for _, v in items]
        pos_lams.extend(g for g in xs if g > 0)
        ax.plot(xs, ys, marker="o", label=label,
                color=_COLORS.get(label.lower()), linewidth=2, markersize=6)
        any_data = True
    if not any_data:
        plt.close(fig)
        return False
    # Log-spaced x-axis. Pure 'log' can't show the λ=0 anchor, so default to
    # 'symlog' (log beyond the smallest positive λ, linear below it through 0).
    if xscale == "symlog":
        linthresh = min(pos_lams) if pos_lams else 1e-3
        ax.set_xscale("symlog", linthresh=linthresh)
    elif xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(r"perturbation $\lambda$  ($\Vert\mathrm{noise}\Vert_F/\Vert W\Vert_F$)",
                  fontsize=12)
    if degradation:
        ax.set_ylabel(f"Δ {ylabel}  (λ − λ=0)", fontsize=12)
        ax.set_title(f"OLMo 60M — {ylabel} degradation vs perturbation", fontsize=13)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    else:
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"OLMo 60M — {ylabel} vs perturbation", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def _plot_slope(models, metric, out_path, xscale, negate):
    """Plot the slope d(metric)/dλ vs λ (one line per model). The derivative is a
    non-uniform central difference (np.gradient) of the metric-vs-λ curve at each
    sampled λ; ``negate`` flips the sign (so a decreasing metric like fta reads as
    a positive "degradation rate"). λ=0 has no log-x position, so only λ>0 points
    are drawn (their derivative still uses the λ=0 neighbour)."""
    _, _, ylabel = METRICS[metric]
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    pos_lams = []
    for label, series in models:
        if len(series) < 2:
            print(f"  [skip slope {metric} {label}: <2 points]")
            continue
        items = sorted(series.items())
        xs = np.array([g for g, _ in items], dtype=float)
        ys = np.array([v for _, v in items], dtype=float)
        d = np.gradient(ys, xs)                 # dY/dλ at each λ (non-uniform)
        if negate:
            d = -d
        m = xs > 0                              # log x can't show λ=0
        if not m.any():
            continue
        pos_lams.extend(xs[m])
        ax.plot(xs[m], d[m], marker="o", label=label,
                color=_COLORS.get(label.lower()), linewidth=2, markersize=6)
        any_data = True
    if not any_data:
        plt.close(fig)
        return False
    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=min(pos_lams) if pos_lams else 1e-3)
    elif xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(r"perturbation $\lambda$  ($\Vert\mathrm{noise}\Vert_F/\Vert W\Vert_F$)",
                  fontsize=12)
    sign = "$-$" if negate else ""
    ax.set_ylabel(rf"{sign}d({ylabel})/d$\lambda$", fontsize=12)
    ax.set_title(f"OLMo 60M — {'negative ' if negate else ''}slope of "
                 f"{ylabel} vs perturbation", fontsize=13)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="label=base_model_name (repeatable).")
    ap.add_argument("--eval-dir", default=EVAL_DIR)
    ap.add_argument("--qa-dir", default=QA_DIR)
    ap.add_argument("--metrics", default="loss,fta,acc",
                    help="comma list from loss/fta/acc.")
    ap.add_argument("--out-dir", default="results/plots/qa_group/perturb")
    ap.add_argument("--xscale", default="symlog", choices=["symlog", "log", "linear"],
                    help="x-axis (λ) scale. 'symlog' (default) is log-spaced but "
                         "still shows the λ=0 anchor; 'log' drops/omits λ=0.")
    ap.add_argument("--max-lambda", type=float, default=None,
                    help="Drop perturbation points with λ (γ) above this "
                         "(the λ=0 base point is always kept).")
    ap.add_argument("--slope", action="store_true",
                    help="Instead of metric-vs-λ, plot the SLOPE d(metric)/dλ vs λ "
                         "(numeric derivative of the curve). Pair with --xscale log.")
    ap.add_argument("--negate", action="store_true",
                    help="With --slope: plot −d(metric)/dλ (so a decreasing metric "
                         "like fta shows as a positive degradation rate).")
    ap.add_argument("--per-group", action="store_true",
                    help="Instead of the overall metric, emit one vs-λ and one "
                         "degradation plot PER group (group00, group01, ...) into "
                         "<out-dir>/groups/<groupNN>/. Only QA metrics (fta/acc) "
                         "have a per-group breakdown; loss is skipped.")
    args = ap.parse_args()

    specs = [s.split("=", 1) for s in args.model]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    def _cap(series):
        if args.max_lambda is None:
            return series
        return {g: v for g, v in series.items()
                if g == 0.0 or g <= args.max_lambda}

    if args.per_group:
        run_per_group(specs, metrics, args)
        return

    if args.slope:
        for metric in metrics:
            source, jpath, _ = METRICS[metric]
            models = []
            for label, base in specs:
                series = _cap(collect(base, source, jpath, args.eval_dir, args.qa_dir))
                models.append((label, series))
                print(f"slope {metric} {label}: {len(series)} pts  λ={sorted(series)}")
            tag = "neg_slope" if args.negate else "slope"
            _plot_slope(models, metric,
                        os.path.join(args.out_dir, f"perturb_{metric}_{tag}_vs_lambda.png"),
                        args.xscale, args.negate)
        return

    for metric in metrics:
        source, jpath, _ = METRICS[metric]
        models = []
        for label, base in specs:
            series = _cap(collect(base, source, jpath, args.eval_dir, args.qa_dir))
            models.append((label, series))
            print(f"{metric} {label}: {len(series)} pts  λ={sorted(series)}")
        _plot(models, metric, False, os.path.join(args.out_dir, f"perturb_{metric}_vs_lambda.png"), args.xscale)
        _plot(models, metric, True, os.path.join(args.out_dir, f"perturb_{metric}_degradation.png"), args.xscale)


def run_per_group(specs, metrics, args):
    """One vs-λ + one degradation plot per group, per QA metric."""
    qa_metrics = [m for m in metrics if METRICS[m][0] == "qa"]
    for m in metrics:
        if m not in qa_metrics:
            print(f"[skip {m}: source is not QA — no per-group breakdown]")
    if not qa_metrics:
        print("[per-group: no QA metrics requested (fta/acc); nothing to do]")
        return

    # Cat each model's qa.json files once; reuse across all groups + metrics.
    raw = {}
    group_keys = set()
    for label, base in specs:
        rmap = collect_qa_raw(base, args.qa_dir)
        if args.max_lambda is not None:
            rmap = {g: md for g, md in rmap.items()
                    if g == 0.0 or g <= args.max_lambda}
        raw[label] = rmap
        for md in rmap.values():
            group_keys.update(k for k in md if k.startswith("group"))
    if not group_keys:
        print("[per-group: no 'groupNN' entries found in qa.json]")
        return
    group_keys = sorted(group_keys)
    print(f"groups: {group_keys}")

    for gk in group_keys:
        gdir = os.path.join(args.out_dir, "groups", gk)
        for metric in qa_metrics:
            field = METRICS[metric][1][-1]   # e.g. ("metrics","overall","fta") -> "fta"
            models = []
            for label, _base in specs:
                series = {g: float(md[gk][field])
                          for g, md in raw[label].items()
                          if gk in md and field in md[gk]
                          and md[gk][field] is not None}
                models.append((label, series))
                print(f"{gk} {metric} {label}: {len(series)} pts")
            _plot(models, metric, False,
                  os.path.join(gdir, f"perturb_{metric}_vs_lambda.png"), args.xscale)
            _plot(models, metric, True,
                  os.path.join(gdir, f"perturb_{metric}_degradation.png"), args.xscale)


if __name__ == "__main__":
    main()
