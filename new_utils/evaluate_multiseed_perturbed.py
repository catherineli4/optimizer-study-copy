#!/usr/bin/env python3
"""Evaluate saved multi-seed perturbed checkpoints and average DCLM_heldout loss.

For each seed under ``{parent}/seed_XXX/final-unsharded/``, runs validate.py
with the provided eval config (model.path overwritten per seed), then writes a
JSON with per-seed losses and their mean.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Sequence

import yaml


def _losses_from_eval(data: dict) -> Dict[str, float]:
    """Extract scalar losses from a validate.py JSON output."""
    losses: Dict[str, float] = {}
    overall = (data.get("overall") or {}).get("loss")
    if overall is not None:
        losses["overall"] = float(overall)
    for label, entry in (data.get("by_label") or {}).items():
        loss = (entry or {}).get("loss")
        if loss is not None:
            losses[label] = float(loss)
    return losses


def seed_subdir(seed: int) -> str:
    return f"seed_{seed:03d}"


def average_seed_losses(
    sample_losses: List[Dict[str, float]],
    *,
    seeds: Sequence[int],
    gamma: float,
) -> dict:
    """Mean loss per label across seeds; also keep per-seed breakdown."""
    if not sample_losses:
        raise ValueError("no sample losses to average")
    if len(sample_losses) != len(seeds):
        raise ValueError(
            f"len(sample_losses)={len(sample_losses)} != len(seeds)={len(seeds)}"
        )

    labels = sorted({label for losses in sample_losses for label in losses})
    out: Dict[str, Any] = {}

    if "overall" in labels:
        vals = [losses["overall"] for losses in sample_losses if "overall" in losses]
        out["overall"] = {"loss": sum(vals) / len(vals)}

    by_label: Dict[str, Any] = {}
    for label in labels:
        if label == "overall":
            continue
        vals = [losses[label] for losses in sample_losses if label in losses]
        if vals:
            by_label[label] = {"loss": sum(vals) / len(vals)}
    if by_label:
        out["by_label"] = by_label

    per_seed = {
        str(seed): dict(losses)
        for seed, losses in zip(seeds, sample_losses)
    }
    out["perturbation"] = {
        "gamma": gamma,
        "seeds": list(seeds),
        "num_seeds": len(seeds),
        "loss_averaging": "mean_over_saved_seeds",
        "per_seed": per_seed,
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Average validate.py loss over saved multi-seed perturbations",
    )
    parser.add_argument(
        "--parent-dir",
        required=True,
        help="Local PerturbedModel/{base}_perturbed_{γ}/ directory containing seed_*/",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated seeds matching seed_XXX subdirs (e.g. 0,1,...,9)",
    )
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument(
        "--eval-config",
        required=True,
        help="validate.py YAML config (model.path is overwritten per seed)",
    )
    parser.add_argument(
        "--validate-script",
        required=True,
        help="Path to JOLMo/src/scripts/validate.py",
    )
    parser.add_argument("--output", required=True, help="Path for averaged eval JSON")
    args = parser.parse_args()

    seeds = [int(p.strip()) for p in args.seeds.split(",") if p.strip()]
    if not seeds:
        raise SystemExit("--seeds must be a non-empty comma-separated list")

    with open(args.eval_config, "r", encoding="utf-8") as f:
        eval_cfg_template = yaml.safe_load(f)

    sample_losses: List[Dict[str, float]] = []
    work_dir = os.path.join(args.parent_dir, "_eval_tmp")
    os.makedirs(work_dir, exist_ok=True)

    for i, seed in enumerate(seeds):
        ckpt_dir = os.path.join(args.parent_dir, seed_subdir(seed), "final-unsharded")
        model_pt = os.path.join(ckpt_dir, "model.pt")
        if not os.path.exists(model_pt):
            raise FileNotFoundError(f"missing seed checkpoint: {model_pt}")

        eval_cfg = dict(eval_cfg_template)
        eval_cfg["model"] = dict(eval_cfg_template.get("model") or {})
        eval_cfg["model"]["path"] = ckpt_dir

        sample_cfg_path = os.path.join(work_dir, f"seed_{seed:03d}-eval.yaml")
        sample_out_path = os.path.join(work_dir, f"seed_{seed:03d}-eval.json")
        with open(sample_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(eval_cfg, f)

        print(f"  seed {i + 1}/{len(seeds)} (seed={seed}) → {ckpt_dir}")
        subprocess.check_call(
            ["python3", args.validate_script, sample_cfg_path, "--output", sample_out_path]
        )
        with open(sample_out_path, "r", encoding="utf-8") as f:
            sample_losses.append(_losses_from_eval(json.load(f)))

    averaged = average_seed_losses(
        sample_losses, seeds=seeds, gamma=args.gamma,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(averaged, f, indent=2)
    print(f"Wrote averaged multi-seed eval to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
