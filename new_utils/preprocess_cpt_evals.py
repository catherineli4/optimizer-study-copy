#!/usr/bin/env python3
"""Stage 1 of the CPT Pareto pipeline: download the CPT eval JSONs from GCS and
aggregate them into ONE local results file, mirroring catastrophic-forgetting's
preprocess_results.py → final_results.json.

  1. list + download every CPT ModelEvaluation JSON (run-prefix filtered) to a
     local cache dir, then
  2. parse each filename for (dataset, chinchilla, optimizers, cpt_lr) and pull
     the relevant losses out of the JSON, and
  3. write a single aggregate `cpt_results.json`.

Then plot offline with `plot_cpt_pareto.py` (no GCS access needed).

For each CPT eval we record the fine-tuning loss (by_label["<dataset>-validation"])
and every available "pretrain/forgetting" loss (DCLM_heldout if present, else the
diversity sets C4_val/… and overall), so the plotter can pick the x-axis without
re-downloading.

Usage:
  python -m new_utils.preprocess_cpt_evals                 # download + aggregate
  python -m new_utils.preprocess_cpt_evals --no-download   # re-aggregate local cache
"""
import argparse
import json
import os
import re
import subprocess

DEFAULT_EVAL_DIR = "gs://cmu-gpucloud-catheri4/Optim-60M-tuning/ModelEvaluation"
_DIVERSITY = ("C4_val", "Reddit_val", "Wiki_val", "Books_val")

_PRE_RE = re.compile(r"-chinchilla-([0-9.]+)-(adamw|muon)-")
_CPT_RE = re.compile(
    r"-CPT-([a-z0-9]+)-\d+M-(adamw|muon)-lr([0-9._eE+\-]+?)"
    r"(?:-exp-muonlr([0-9._eE+\-]+))?-eval\.json$"
)


def _gsutil_ls(d):
    out = subprocess.run(["gsutil", "ls", d], capture_output=True, text=True)
    return [l.strip() for l in out.stdout.splitlines() if l.strip().endswith(".json")]


def _num(s):
    return float(s.replace("_", "."))


def parse_name(filename):
    """dict(dataset, chinchilla, pretrain_optimizer, cpt_optimizer, cpt_lr, ...) or None.

    For muon CPT runs the name is `…-muon-lr{adamw_lr}-exp-muonlr{muon_lr}`, so we
    also record cpt_muon_lr / cpt_adamw_lr and alpha = adamw_lr / muon_lr (the
    muon→adamw LR ratio swept by the muon-sweep experiment)."""
    cpt = _CPT_RE.search(filename)
    pre = _PRE_RE.search(filename)
    if not cpt or not pre:
        return None
    ds, copt, adamw_lr, muon_lr = cpt.groups()
    chin_s, popt = pre.groups()
    chin = float(chin_s)
    chin = int(chin) if chin.is_integer() else chin
    cpt_lr = _num(muon_lr) if (copt == "muon" and muon_lr) else _num(adamw_lr)
    info = dict(dataset=ds, chinchilla=chin, pretrain_optimizer=popt,
                cpt_optimizer=copt, cpt_lr=cpt_lr)
    if copt == "muon" and muon_lr:
        a_lr, m_lr = _num(adamw_lr), _num(muon_lr)
        info["cpt_muon_lr"] = m_lr
        info["cpt_adamw_lr"] = a_lr
        info["alpha"] = round(a_lr / m_lr, 4) if m_lr else None
    return info


def download(eval_dir, local_dir, run_prefix):
    """Download every (run-prefix) CPT eval JSON in eval_dir into local_dir."""
    os.makedirs(local_dir, exist_ok=True)
    all_files = _gsutil_ls(eval_dir.rstrip("/") + "/")
    cpt_files = [f for f in all_files
                 if "-CPT-" in os.path.basename(f)
                 and (not run_prefix or os.path.basename(f).startswith(run_prefix))]
    print(f"{len(cpt_files)} CPT eval JSON(s) to download → {local_dir}")
    if not cpt_files:
        return cpt_files
    # gsutil -m cp -I reads object names from stdin; -n skips already-downloaded.
    proc = subprocess.run(
        ["gsutil", "-m", "cp", "-n", "-I", local_dir],
        input="\n".join(cpt_files), text=True,
    )
    if proc.returncode != 0:
        print(f"[warn] gsutil cp exited {proc.returncode} (some files may be missing)")
    return cpt_files


def aggregate(local_dir, run_prefix):
    """Parse every local CPT eval JSON into a flat list of records."""
    records = []
    skipped = 0
    for name in sorted(os.listdir(local_dir)):
        if not name.endswith("-eval.json"):
            continue
        if run_prefix and not name.startswith(run_prefix):
            continue
        info = parse_name(name)
        if info is None:
            continue
        try:
            with open(os.path.join(local_dir, name)) as f:
                d = json.load(f)
        except Exception as exc:
            print(f"[warn] read failed {name}: {exc}")
            continue
        by_label = d.get("by_label") or {}
        ft = by_label.get(f"{info['dataset']}-validation", {}).get("loss")
        if ft is None:
            skipped += 1
            continue
        # Every available forgetting/pretrain loss, so the plotter can choose x.
        losses = {"overall": (d.get("overall") or {}).get("loss")}
        for k in ("DCLM_heldout",) + _DIVERSITY:
            if k in by_label:
                losses[k] = by_label[k].get("loss")
        records.append({**info, "finetune_loss": ft, "losses": losses, "file": name})
    print(f"aggregated {len(records)} record(s) ({skipped} skipped: no finetune loss)")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    ap.add_argument("--local-dir", default="results/cpt/eval_json",
                    help="Local cache dir for the downloaded eval JSONs.")
    ap.add_argument("--run-prefix", default="MuonExpt3",
                    help="Only files starting with this (excludes stale families "
                         "like MuonEpochExpt). Empty = no filter.")
    ap.add_argument("--out", default="results/cpt/cpt_results.json",
                    help="Aggregated results file consumed by plot_cpt_pareto.py.")
    ap.add_argument("--no-download", action="store_true",
                    help="Skip GCS download; just re-aggregate the local cache.")
    args = ap.parse_args()

    if not args.no_download:
        download(args.eval_dir, args.local_dir, args.run_prefix)
    else:
        print(f"--no-download: aggregating local cache {args.local_dir}")

    records = aggregate(args.local_dir, args.run_prefix)
    if not records:
        print("No CPT records aggregated; nothing written.")
        return

    has_dclm = sum(1 for r in records if r["losses"].get("DCLM_heldout") is not None)
    datasets = sorted({r["dataset"] for r in records})
    print(f"datasets: {', '.join(datasets)}")
    print(f"records with DCLM_heldout: {has_dclm}/{len(records)} "
          f"({'re-run eval-cpt to populate' if has_dclm < len(records) else 'all present'})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
