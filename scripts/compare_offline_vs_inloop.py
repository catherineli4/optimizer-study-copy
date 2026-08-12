#!/usr/bin/env python3
"""Compare offline ModelEvaluation losses against the in-loop wandb eval curve.

A healthy run's offline loss lands on the last in-loop point. A run whose saved
checkpoint does not match the model the trainer was evaluating shows a positive
delta, and the offline value maps back to an *earlier* training step.
"""
import argparse
import glob
import json
import os
import re

import wandb

LABELS = ["C4_val", "Books_val", "Wiki_val", "Reddit_val"]


def implied_step(curve, offline_loss):
    """First step whose in-loop CE has already dropped to the offline value."""
    for step, val in curve:
        if val <= offline_loss:
            return step
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="ModelEvaluation")
    ap.add_argument("--pattern", default="*0.1B*wsd-eval.json")
    ap.add_argument("--entity", default="catheri4-carnegie-mellon-university")
    ap.add_argument("--project", default="Optim-100M-tuning.Optim-100M-tuning")
    ap.add_argument("--label", default="C4_val")
    args = ap.parse_args()

    api = wandb.Api()
    key = f"eval/lm/{args.label}/CE loss"

    files = sorted(glob.glob(os.path.join(args.eval_dir, args.pattern)))
    files = [f for f in files if "CPT" not in f and "perturb" not in f]

    rows = []
    for path in files:
        run_name = os.path.basename(path)[: -len("-eval.json")]
        offline = json.load(open(path))["by_label"].get(args.label)
        if offline is None:
            continue
        matches = list(api.runs(f"{args.entity}/{args.project}",
                                filters={"display_name": run_name}))
        if not matches:
            print(f"  (no wandb run for {run_name})")
            continue
        hist = [(h["_step"], h[key]) for h in matches[0].scan_history(keys=["_step", key])
                if h.get(key) is not None]
        if not hist:
            continue
        last_step, last_val = hist[-1]
        o = offline["loss"]
        m = re.search(r"chinchilla-(\d+)", run_name)
        rows.append({
            "run": run_name,
            "opt": "muon" if "-muon-" in run_name else "adamw",
            "chin": int(m.group(1)) if m else -1,
            "offline": o,
            "inloop": last_val,
            "delta": o - last_val,
            "last_step": last_step,
            "implied_step": implied_step(hist, o),
        })

    rows.sort(key=lambda r: (r["opt"], r["chin"]))
    print(f"{'optimizer':9s} {'chin':>5s} {'offline':>9s} {'in-loop':>9s} "
          f"{'delta':>8s} {'last':>6s} {'implied':>8s}")
    for r in rows:
        imp = r["implied_step"]
        print(f"{r['opt']:9s} {r['chin']:5d} {r['offline']:9.4f} {r['inloop']:9.4f} "
              f"{r['delta']:+8.4f} {r['last_step']:6d} "
              f"{imp if imp is not None else 'never':>8}")


if __name__ == "__main__":
    main()
