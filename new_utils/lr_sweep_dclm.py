#!/usr/bin/env python3
"""Per-Chinchilla LR sweep: DCLM held-out loss vs pretrain learning rate.

For every Chinchilla budget, this evaluates *all* trained AdamW and Muon models
(every pretrain LR that exists on GCS, not just the tuned cell) and plots the
held-out DCLM validation loss against the pretrain LR — one figure per
Chinchilla. AdamW = blue, Muon = orange. For Muon the x-axis is the Muon LR
(the AdamW-component LR is fixed per Chinchilla).

Two subcommands:

  eval   Enumerate the trained JolmoModel runs on GCS, rebuild each one,
         wrap it in the same DCLM-held-out ModelEvaluation the pretrain eval
         pipeline uses (`_pretrain_eval`), and submit them as SLURM jobs.
         Use --dry to preview without submitting.

             python -m new_utils.lr_sweep_dclm eval --dry
             python -m new_utils.lr_sweep_dclm eval          # actually submit

  plot   After the evals finish, read the `…-eval.json` files back from GCS and
         write results/pretrain-eval/lr-sweep/chinchilla-<N>-dclm-vs-lr.png.

             python -m new_utils.lr_sweep_dclm plot

NOTE: `eval` submits one GPU job per (Chinchilla × optimizer × LR) model — that
is a lot of jobs. Preview with --dry first, and wait for `squeue` to drain
before running `plot`. The DCLM metric only appears in evals produced after the
held-out DCLM shard was added, so models must be (re)evaluated here first.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional

# --- Project context (must be initialized before importing the matrix) --------
from experiments import Project, SlurmExecutor

Project.init("Optim-60M-tuning")

from launch_jolmo.training import JolmoModel  # noqa: E402
from launch_jolmo.pretraining_matrix import (  # noqa: E402
    MODEL_TYPE,
    SCHEDULER,
    SHARED_MODEL_PARAMS,
    _tokens_for,
    _dclm_chunks_for_tokens,
    _pretrain_eval,
)
from launch_jolmo.utils import remote_path  # noqa: E402

# Color convention (per request): adamw = blue, muon = orange.
OPT_STYLE = {
    "adamw": {"color": "tab:blue",   "marker": "o"},
    "muon":  {"color": "tab:orange", "marker": "s"},
}

DCLM_METRIC = "DCLM_heldout"
JOLMO_GS = remote_path("JolmoModel")
EVAL_GS = remote_path("ModelEvaluation")
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "pretrain-eval", "lr-sweep",
)

# A trained run name, e.g.
#   MuonExpt3-0.06B-chinchilla-4-adamw-lr7.0e-3-wsd
#   MuonExpt3-0.06B-chinchilla-4-muon-muonlr1.4e-2-adamwlr7.0e-3-wsd
_NUM = r"[0-9.]+e-?[0-9]+|[0-9.]+"
ADAMW_RE = re.compile(
    rf"^MuonExpt3-{re.escape(MODEL_TYPE)}-chinchilla-(?P<c>[0-9.]+)-adamw-lr(?P<lr>{_NUM})-(?P<sched>wsd|cosine)$"
)
MUON_RE = re.compile(
    rf"^MuonExpt3-{re.escape(MODEL_TYPE)}-chinchilla-(?P<c>[0-9.]+)-muon-muonlr(?P<mlr>{_NUM})-adamwlr(?P<alr>{_NUM})-(?P<sched>wsd|cosine)$"
)


def _gsutil_ls(gs_dir: str) -> List[str]:
    r = subprocess.run(
        ["gsutil", "ls", gs_dir.rstrip("/") + "/"],
        capture_output=True, text=True,
    )
    return [ln.rstrip("/") for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else []


def _chinchilla(value: str) -> float:
    v = float(value)
    return int(v) if v.is_integer() else v


def parse_run(name: str) -> Optional[Dict]:
    """Parse a JolmoModel dir name into (optimizer, chinchilla, lr, ...)."""
    m = ADAMW_RE.match(name)
    if m:
        return {
            "name": name, "optimizer": "adamw",
            "chinchilla": _chinchilla(m.group("c")),
            "scheduler": m.group("sched"),
            "pretrain_lr": float(m.group("lr")),      # x-axis value
            "learning_rate": float(m.group("lr")),
        }
    m = MUON_RE.match(name)
    if m:
        return {
            "name": name, "optimizer": "muon",
            "chinchilla": _chinchilla(m.group("c")),
            "scheduler": m.group("sched"),
            "pretrain_lr": float(m.group("mlr")),     # x-axis value = muon LR
            "muon_lr": float(m.group("mlr")),
            "learning_rate": float(m.group("alr")),   # adamw-component LR (fixed per chinchilla)
        }
    return None


def discover_runs(scheduler: str, chinchillas: Optional[List[float]]) -> List[Dict]:
    """List trained JolmoModel runs on GCS and parse the ones we can plot."""
    runs = []
    for entry in _gsutil_ls(JOLMO_GS):
        info = parse_run(os.path.basename(entry))
        if not info:
            continue
        if info["scheduler"] != scheduler:
            continue
        if chinchillas is not None and info["chinchilla"] not in chinchillas:
            continue
        runs.append(info)
    runs.sort(key=lambda r: (r["chinchilla"], r["optimizer"], r["pretrain_lr"]))
    return runs


def build_model(info: Dict) -> JolmoModel:
    """Rebuild the exact JolmoModel for a parsed run (run_name == model_name)."""
    schedule = _tokens_for(info["chinchilla"])
    train_chunks = _dclm_chunks_for_tokens(schedule["n_tokens"])
    common = {
        **SHARED_MODEL_PARAMS, **schedule,
        "scheduler": info["scheduler"], "train_chunks": train_chunks,
        "model_name": info["name"],
    }
    if info["optimizer"] == "adamw":
        return JolmoModel(**common, optimizer="adamw", learning_rate=info["learning_rate"])
    return JolmoModel(
        **common, optimizer="muon",
        muon_lr=info["muon_lr"], learning_rate=info["learning_rate"],
    )


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #

def _setup_command() -> str:
    if Project.config.cluster == "babel":
        return "; ".join([
            "source ~/miniconda3/etc/profile.d/conda.sh",
            "conda activate optim-study",
        ])
    if Project.config.cluster == "orchard":
        return "; ".join([
            "source /home/jspringe/.bashrc",
            "source /home/jspringe/.secrets",
            "source /home/jspringe/env/train/bin/activate",
        ])
    raise ValueError(f"Unknown cluster: {Project.config.cluster}")


def cmd_eval(args: argparse.Namespace) -> None:
    chinchillas = [_chinchilla(c) for c in args.chinchillas] if args.chinchillas else None
    runs = discover_runs(args.scheduler, chinchillas)
    if not runs:
        print("No trained runs matched on GCS; nothing to evaluate.")
        return

    print(f"Discovered {len(runs)} trained run(s) to evaluate (DCLM held-out):")
    for r in runs:
        print(f"  chinchilla={r['chinchilla']:<4} {r['optimizer']:<5} lr={r['pretrain_lr']:.1e}  {r['name']}")

    models = [build_model(r) for r in runs]
    evals = [_pretrain_eval(m) for m in models]

    executor = SlurmExecutor(setup_command=_setup_command(), dry_run=args.dry)
    # The models must be registered too so the eval's dependency on its
    # JolmoModel resolves (the dependency graph spans all registered stages).
    # They already exist on GCS, so they're skipped — not retrained — and only
    # serve to satisfy the dependency, exactly as launcher.py registers them.
    executor.stage("pretrain-lrsweep-models", models)
    executor.stage("eval-pretrain-lrsweep", evals)
    # rerun=True forces evaluation even if a JSON exists (we want fresh DCLM numbers).
    executor.execute(["eval-pretrain-lrsweep"], rerun=args.rerun, head=args.head)


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #

def _load_json(gs_path: str) -> Optional[dict]:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name
    try:
        rc = subprocess.run(["gsutil", "cp", gs_path, tmp_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if rc != 0:
            return None
        with open(tmp_path) as f:
            return json.load(f)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def collect_dclm(scheduler: str, chinchillas: Optional[List[float]], metric: str
                 ) -> Dict[float, Dict[str, Dict[float, float]]]:
    """chinchilla -> optimizer -> {pretrain_lr: dclm_loss}."""
    out: Dict[float, Dict[str, Dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
    for entry in _gsutil_ls(EVAL_GS):
        fname = os.path.basename(entry)
        if not fname.endswith("-eval.json"):
            continue
        low = fname.lower()
        if any(t in low for t in ("_perturbed_", "-cpt-", "typo")):
            continue
        info = parse_run(fname[: -len("-eval.json")])
        if not info or info["scheduler"] != scheduler:
            continue
        if chinchillas is not None and info["chinchilla"] not in chinchillas:
            continue
        data = _load_json(entry)
        if not data:
            continue
        entry_metric = (data.get("by_label") or {}).get(metric)
        if not isinstance(entry_metric, dict) or entry_metric.get("loss") is None:
            continue
        out[info["chinchilla"]][info["optimizer"]][info["pretrain_lr"]] = float(entry_metric["loss"])
    return out


def plot_chinchilla(chinchilla: float, by_opt: Dict[str, Dict[float, float]],
                    metric: str, out_path: str) -> bool:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    drew = False
    for opt in sorted(by_opt):                     # adamw before muon
        series = by_opt[opt]
        if not series:
            continue
        xs = sorted(series.keys())
        ys = [series[x] for x in xs]
        style = OPT_STYLE.get(opt, {"color": "gray", "marker": "x"})
        ax.plot(xs, ys, label=opt, color=style["color"], marker=style["marker"],
                linewidth=2, markersize=7)
        drew = True

    if not drew:
        plt.close(fig)
        return False

    ax.set_xscale("log")
    ax.set_xlabel("Pretrain learning rate", fontsize=12)
    ax.set_ylabel(f"DCLM held-out val loss ({metric})", fontsize=12)
    ax.set_title(f"Chinchilla {chinchilla}: DCLM loss vs pretrain LR", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Optimizer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def cmd_plot(args: argparse.Namespace) -> None:
    chinchillas = [_chinchilla(c) for c in args.chinchillas] if args.chinchillas else None
    data = collect_dclm(args.scheduler, chinchillas, args.metric)
    if not data:
        print(f"No '{args.metric}' data found in any pretrain eval JSON. "
              f"Run `eval` first and wait for the SLURM jobs to finish.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Writing per-Chinchilla plots to {args.out_dir}")
    for chinchilla in sorted(data):
        out_path = os.path.join(args.out_dir, f"chinchilla-{chinchilla}-dclm-vs-lr.png")
        n = {o: len(s) for o, s in data[chinchilla].items()}
        if plot_chinchilla(chinchilla, data[chinchilla], args.metric, out_path):
            print(f"  chinchilla={chinchilla}: {n} -> {out_path}")
        else:
            print(f"  chinchilla={chinchilla}: no usable points, skipped")


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("eval", help="Submit DCLM evals for all trained LR models.")
    pe.add_argument("--scheduler", default=SCHEDULER)
    pe.add_argument("--chinchillas", nargs="*", default=None,
                    help="Restrict to these Chinchilla budgets (default: all found).")
    pe.add_argument("--dry", action="store_true", help="Preview SLURM jobs without submitting.")
    pe.add_argument("--rerun", action="store_true", default=True,
                    help="Force re-eval even if a JSON exists (default: on).")
    pe.add_argument("--no-rerun", dest="rerun", action="store_false",
                    help="Skip models whose eval JSON already exists.")
    pe.add_argument("--head", type=int, default=None, help="Only the first N evals.")
    pe.set_defaults(func=cmd_eval)

    pp = sub.add_parser("plot", help="Plot DCLM loss vs pretrain LR, one fig per Chinchilla.")
    pp.add_argument("--scheduler", default=SCHEDULER)
    pp.add_argument("--chinchillas", nargs="*", default=None)
    pp.add_argument("--metric", default=DCLM_METRIC)
    pp.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    pp.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
