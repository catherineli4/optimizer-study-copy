#!/usr/bin/env python3
"""For one pretrained checkpoint: perturb each matrix weight once and eval.

Writes one ``*-eval.json`` per matrix (same naming as single-param PerturbedModel
evals) so ``process_evals.py`` can ingest them. Intended to run as one Slurm job
per base model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturb_weights import get_perturbed_model_name, perturb_state_dict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-checkpoint-dir", required=True)
    ap.add_argument("--base-run-name", required=True,
                    help="Unperturbed JolmoModel run_name (for output naming).")
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--param-names", required=True,
                    help="Comma-separated state-dict keys to perturb one-at-a-time.")
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--validate-script", required=True)
    ap.add_argument("--output-dir", required=True,
                    help="Local ModelEvaluation/ dir for *-eval.json files.")
    ap.add_argument("--done-marker", required=True,
                    help="Written last so the launcher can skip finished jobs.")
    args = ap.parse_args()

    param_names = [p.strip() for p in args.param_names.split(",") if p.strip()]
    if not param_names:
        raise SystemExit("--param-names must be a non-empty comma-separated list")

    model_pt = os.path.join(args.base_checkpoint_dir, "model.pt")
    if not os.path.isfile(model_pt):
        raise FileNotFoundError(model_pt)

    print(f"Loading base checkpoint {model_pt} …")
    base_state = torch.load(model_pt, map_location="cpu", weights_only=False)

    with open(args.eval_config) as f:
        eval_cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="per_matrix_pert_") as tmp:
        for i, param in enumerate(param_names):
            run_name = get_perturbed_model_name(
                args.base_run_name, args.gamma, param_name=param,
            )
            out_json = os.path.join(args.output_dir, f"{run_name}-eval.json")
            if os.path.isfile(out_json):
                print(f"[{i+1}/{len(param_names)}] skip existing {run_name}")
                continue

            print(f"[{i+1}/{len(param_names)}] perturb {param}")
            perturbed = perturb_state_dict(
                base_state, args.gamma, seed=args.seed, param_names={param},
            )
            sample_dir = os.path.join(tmp, "sample")
            if os.path.isdir(sample_dir):
                shutil.rmtree(sample_dir)
            os.makedirs(sample_dir)
            for name in ("config.json",):
                src = os.path.join(args.base_checkpoint_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(sample_dir, name))
            torch.save(perturbed, os.path.join(sample_dir, "model.pt"))

            cfg = dict(eval_cfg)
            cfg["model"] = dict(eval_cfg["model"])
            cfg["model"]["path"] = sample_dir
            cfg_path = os.path.join(tmp, "eval.yaml")
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f)

            subprocess.check_call(
                [sys.executable, args.validate_script, cfg_path, "--output", out_json]
            )
            print(f"  wrote {out_json}")

    os.makedirs(os.path.dirname(args.done_marker) or ".", exist_ok=True)
    with open(args.done_marker, "w") as f:
        json.dump(
            {
                "base_run_name": args.base_run_name,
                "gamma": args.gamma,
                "seed": args.seed,
                "n_params": len(param_names),
                "param_names": param_names,
            },
            f,
            indent=2,
        )
    print(f"Done marker: {args.done_marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
