#!/usr/bin/env python3
"""Evaluate a model under many random weight perturbations and average the losses.

For each of ``num_samples`` independent Gaussian noise draws at a fixed gamma,
perturbs the base checkpoint in memory, runs validate.py, and averages the
reported losses across all draws.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturb_weights import perturb_state_dict


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


def average_eval_results(
    sample_losses: List[Dict[str, float]],
    *,
    gamma: float,
    num_samples: int,
    seed: int,
) -> dict:
    """Average per-dataset losses across perturbation samples."""
    if not sample_losses:
        raise ValueError("no sample losses to average")

    labels = sorted({label for losses in sample_losses for label in losses})
    out: Dict[str, Any] = {}

    if "overall" in labels:
        vals = [losses["overall"] for losses in sample_losses if "overall" in losses]
        out["overall"] = {"loss": sum(vals) / len(vals)}

    by_label = {}
    for label in labels:
        if label == "overall":
            continue
        vals = [losses[label] for losses in sample_losses if label in losses]
        if vals:
            by_label[label] = {"loss": sum(vals) / len(vals)}
    if by_label:
        out["by_label"] = by_label

    out["perturbation"] = {
        "gamma": gamma,
        "num_samples": num_samples,
        "seed": seed,
        "loss_averaging": "mean_over_noise_draws",
    }
    return out


def _prepare_sample_checkpoint(
    base_checkpoint_dir: str,
    sample_dir: str,
    perturbed_state: dict,
) -> str:
    os.makedirs(sample_dir, exist_ok=True)
    for name in ("config.json",):
        src = os.path.join(base_checkpoint_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(sample_dir, name))
    torch.save(perturbed_state, os.path.join(sample_dir, "model.pt"))
    return sample_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Average validation loss over many random weight perturbations",
    )
    parser.add_argument(
        "--base-checkpoint-dir",
        required=True,
        help="Directory containing the unperturbed final-unsharded/model.pt",
    )
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument(
        "--eval-config",
        required=True,
        help="validate.py YAML config (model.path is overwritten per sample)",
    )
    parser.add_argument(
        "--validate-script",
        required=True,
        help="Path to JOLMo/src/scripts/validate.py",
    )
    parser.add_argument("--output", required=True, help="Path for averaged eval JSON")
    args = parser.parse_args()

    if args.num_samples < 1:
        raise SystemExit("num_samples must be >= 1")

    base_model_path = os.path.join(args.base_checkpoint_dir, "model.pt")
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"missing base checkpoint: {base_model_path}")

    with open(args.eval_config, "r", encoding="utf-8") as f:
        eval_cfg_template = yaml.safe_load(f)

    print(
        f"Loading base checkpoint and averaging {args.num_samples} perturbation "
        f"eval(s) at gamma={args.gamma}..."
    )
    base_state = torch.load(base_model_path, map_location="cpu")
    if isinstance(base_state, dict) and "state_dict" in base_state:
        base_state = base_state["state_dict"]
    elif isinstance(base_state, dict) and "model" in base_state:
        base_state = base_state["model"]

    sample_losses: List[Dict[str, float]] = []
    work_dir = tempfile.mkdtemp(prefix="perturb_avg_")
    try:
        for i in range(args.num_samples):
            sample_seed = args.seed + i
            perturbed_state = perturb_state_dict(
                base_state,
                args.gamma,
                seed=sample_seed,
            )
            sample_dir = os.path.join(work_dir, f"sample_{i:03d}")
            _prepare_sample_checkpoint(args.base_checkpoint_dir, sample_dir, perturbed_state)

            eval_cfg = dict(eval_cfg_template)
            eval_cfg["model"] = dict(eval_cfg_template.get("model") or {})
            eval_cfg["model"]["path"] = sample_dir

            sample_cfg_path = os.path.join(sample_dir, "eval.yaml")
            sample_out_path = os.path.join(sample_dir, "eval.json")
            with open(sample_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(eval_cfg, f)

            print(f"  sample {i + 1}/{args.num_samples} (seed={sample_seed})")
            subprocess.check_call(
                ["python3", args.validate_script, sample_cfg_path, "--output", sample_out_path]
            )
            with open(sample_out_path, "r", encoding="utf-8") as f:
                sample_losses.append(_losses_from_eval(json.load(f)))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    averaged = average_eval_results(
        sample_losses,
        gamma=args.gamma,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(averaged, f, indent=2)
    print(f"Wrote averaged eval to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
