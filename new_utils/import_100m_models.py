#!/usr/bin/env python3
"""Import the optimal-LR 0.1B (100M) pretrained models into the personal GCS
bucket in the SAME layout/naming as the existing 60M models.

Source (jgai's seq4k / 1M-token-batch sweep, one dir per run, HF-style
single safetensors). Only directories whose name contains ``-seq4k-bs1m-``
are imported — the older non-seq4k runs are a different training setup:
    gs://cmu-gpucloud-jspringe/shared/jgai/optim_study/0.1B/<src_name>/model.safetensors
        src_name:  jolmo-0.1B-wsd-c<chin>-seq4k-bs1m-adamw-lr<lr>
                   jolmo-0.1B-wsd-c<chin>-seq4k-bs1m-muon-mlr<mlr>-alr<alr>

Target (mirrors gs://cmu-gpucloud-catheri4/Optim-60M-tuning/JolmoModel/..., but a
SEPARATE 100M prefix), in OLMo-core unsharded format the eval/CPT/perturb pipeline
reads (final-unsharded/{model.pt,config.json}):
    gs://cmu-gpucloud-catheri4/Optim-100M-tuning/JolmoModel/<run_name>/final-unsharded/
        run_name (60M convention, size tag kept as 0.1B):
            MuonExpt3-0.1B-chinchilla-<chin>-adamw-lr<lrtag>-wsd
            MuonExpt3-0.1B-chinchilla-<chin>-muon-muonlr<mtag>-adamwlr<atag>-wsd

Why a conversion (not a plain copy): the source is a single HF-style
``model.safetensors`` whose tensor keys already match the OLMo-core ``Transformer``
module names (blocks.*, embeddings.weight, lm_head.*). The 60M pipeline instead
loads ``final-unsharded/model.pt`` (a flat torch state_dict) + ``config.json``
(see JOLMo/src/scripts/validate.py::eval_olmo). So per model we:
  1. download model.safetensors,
  2. re-save the (flat) state_dict as model.pt,
  3. write a config.json for the 0.1B architecture,
  4. upload both to the target final-unsharded/ dir.

Usage (from repo root, optim-study env so gsutil/torch/safetensors are present):
    python -m new_utils.import_100m_models --dry-run      # show the source->target plan
    python -m new_utils.import_100m_models                 # do the conversion + upload
    python -m new_utils.import_100m_models --only 8 16 32  # restrict to some chinchillas
    python -m new_utils.import_100m_models --overwrite      # re-upload even if target exists
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
SRC_PREFIX = "gs://cmu-gpucloud-jspringe/shared/jgai/optim_study/0.1B"
DST_PREFIX = "gs://cmu-gpucloud-catheri4/Optim-100M-tuning/JolmoModel"
SIZE_TAG = "0.1B"          # kept in the run name (these really are 100M models)
SCHEDULER = "wsd"
# Marker that distinguishes the seq=4096 / global-batch=1M sweep from the older
# jgai runs (which omit this tag and used a different training setup).
SRC_SETUP_TAG = "seq4k-bs1m"

# --------------------------------------------------------------------------- #
# Optimal-LR selection (chinchilla -> source LR tag, exactly as in the src dir).
# Edit here to add/drop models. LR strings MUST match the source directory names.
#   AdamW: lr tag.
#   Muon : list of (muon_lr_tag, adamw_component_lr_tag) pairs; >1 entry means
#          several LRs were equally optimal and all are imported.
# Only chinchillas that have a ``-seq4k-bs1m-`` source dir are listed.
# --------------------------------------------------------------------------- #
ADAMW_OPTIMAL = {
    8: "7e-3",
    16: "7e-3",
    32: "7e-3",  # tied with 5e-3; first pick kept
}

MUON_OPTIMAL = {
    8: [("1e-2", "7e-3")],
    16: [("1e-2", "7e-3")],
    32: [("5e-3", "7e-3")],
}

# 0.1B architecture (training.MODEL_ARCHS["0.1B"]: d_model=128*3, n_layers=3*3,
# n_heads=2*3, hidden=512*3) — verified against the safetensors shapes.
ARCH_0_1B = dict(d_model=384, n_layers=9, n_heads=6, hidden_size=1536)
VOCAB_SIZE = 100352

# Model section of config.json, copied from a current 60M config and re-pointed
# to the 0.1B architecture. validate.py only reads exp_cfg["model"], so this is
# the only section needed. Matches the olmo_core version that produced the 60M
# checkpoints (attention backend "torch", reordered_norm block).
MODEL_CONFIG = {
    "d_model": ARCH_0_1B["d_model"],
    "vocab_size": VOCAB_SIZE,
    "n_layers": ARCH_0_1B["n_layers"],
    "block": {
        "attention": {
            "name": "default",
            "n_heads": ARCH_0_1B["n_heads"],
            "bias": False,
            "rope": {
                "name": "default",
                "theta": 500000,
                "full_precision": True,
                "_CLASS_": "olmo_core.nn.rope.RoPEConfig",
            },
            "qk_norm": {
                "name": "rms",
                "eps": 1e-06,
                "bias": False,
                "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig",
            },
            "backend": "torch",
            "dtype": "float32",
            "_CLASS_": "olmo_core.nn.attention.AttentionConfig",
        },
        "layer_norm": {
            "name": "rms",
            "eps": 1e-06,
            "bias": False,
            "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig",
        },
        "feed_forward": {
            "hidden_size": ARCH_0_1B["hidden_size"],
            "name": "default",
            "bias": False,
            "dtype": "float32",
            "_CLASS_": "olmo_core.nn.feed_forward.FeedForwardConfig",
        },
        "name": "reordered_norm",
        "_CLASS_": "olmo_core.nn.transformer.config.TransformerBlockConfig",
    },
    "lm_head": {
        "name": "default",
        "layer_norm": {
            "name": "rms",
            "eps": 1e-06,
            "bias": False,
            "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig",
        },
        "bias": False,
        "dtype": "float32",
        "loss_implementation": "default",
        "_CLASS_": "olmo_core.nn.lm_head.LMHeadConfig",
    },
    "name": "default",
    "dtype": "float32",
    "init_method": "normal",
    "init_seed": 0,
    "init_std": 0.02,
    "_CLASS_": "olmo_core.nn.transformer.config.TransformerConfig",
}


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
def lr_tag(lr: float) -> str:
    """60M convention: f'{lr:.1e}' with the leading zero of the exponent dropped."""
    return f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def src_adamw_dir(chin: int, lr: str) -> str:
    return (
        f"{SRC_PREFIX}/jolmo-{SIZE_TAG}-{SCHEDULER}-c{chin}-"
        f"{SRC_SETUP_TAG}-adamw-lr{lr}"
    )


def src_muon_dir(chin: int, mlr: str, alr: str) -> str:
    return (
        f"{SRC_PREFIX}/jolmo-{SIZE_TAG}-{SCHEDULER}-c{chin}-"
        f"{SRC_SETUP_TAG}-muon-mlr{mlr}-alr{alr}"
    )


def dst_adamw_name(chin: int, lr: str) -> str:
    return f"MuonExpt3-{SIZE_TAG}-chinchilla-{chin}-adamw-lr{lr_tag(float(lr))}-{SCHEDULER}"


def dst_muon_name(chin: int, mlr: str, alr: str) -> str:
    return (f"MuonExpt3-{SIZE_TAG}-chinchilla-{chin}-muon-"
            f"muonlr{lr_tag(float(mlr))}-adamwlr{lr_tag(float(alr))}-{SCHEDULER}")


def build_plan(only):
    """Return a list of (src_dir, dst_run_name) pairs to import."""
    plan = []
    for chin, lr in sorted(ADAMW_OPTIMAL.items()):
        if only and chin not in only:
            continue
        plan.append((src_adamw_dir(chin, lr), dst_adamw_name(chin, lr)))
    for chin, pairs in sorted(MUON_OPTIMAL.items()):
        if only and chin not in only:
            continue
        for mlr, alr in pairs:
            plan.append((src_muon_dir(chin, mlr, alr), dst_muon_name(chin, mlr, alr)))
    return plan


# --------------------------------------------------------------------------- #
# GCS helpers
# --------------------------------------------------------------------------- #
def gs_exists(gs_path: str) -> bool:
    return subprocess.run(["gsutil", "-q", "stat", gs_path]).returncode == 0


def gs_cp(local: str, gs_path: str) -> None:
    subprocess.run(["gsutil", "-m", "cp", local, gs_path], check=True)


def gs_cat_to(local: str, gs_path: str) -> None:
    subprocess.run(["gsutil", "-m", "cp", gs_path, local], check=True)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def convert_one(src_dir: str, dst_name: str, workdir: str, overwrite: bool,
                verify: bool) -> str:
    import torch
    from safetensors.torch import load_file

    dst_dir = f"{DST_PREFIX}/{dst_name}/final-unsharded"
    dst_model = f"{dst_dir}/model.pt"
    dst_config = f"{dst_dir}/config.json"

    if not overwrite and gs_exists(dst_model):
        print(f"  [skip] target already exists: {dst_model}")
        return "skipped"

    src_safet = f"{src_dir}/model.safetensors"
    if not gs_exists(src_safet):
        print(f"  [MISS] source not found: {src_safet}")
        return "missing"

    local_safet = os.path.join(workdir, "model.safetensors")
    local_model = os.path.join(workdir, "model.pt")
    local_config = os.path.join(workdir, "config.json")

    print(f"  download {src_safet}")
    gs_cat_to(local_safet, src_safet)

    print("  load safetensors -> flat state_dict")
    state = load_file(local_safet)   # {key: bf16 tensor}, keys already olmo-core
    print(f"    {len(state)} tensors")

    if verify:
        _verify_loads(state)

    print("  save model.pt + config.json")
    torch.save(state, local_model)
    with open(local_config, "w") as f:
        json.dump({"model": MODEL_CONFIG}, f, indent=2)

    print(f"  upload -> {dst_dir}/")
    gs_cp(local_model, dst_model)
    gs_cp(local_config, dst_config)

    for p in (local_safet, local_model, local_config):
        if os.path.exists(p):
            os.remove(p)
    return "done"


def _verify_loads(state: dict) -> None:
    """Best-effort: build the 0.1B Transformer and strict-load the state dict so a
    key/shape mismatch is caught before upload. Skipped if olmo_core isn't importable."""
    try:
        from olmo_core.nn.transformer import Transformer, TransformerConfig
    except Exception as exc:  # noqa: BLE001
        print(f"    [verify] skipped (olmo_core import failed: {exc})")
        return
    model: Transformer = TransformerConfig.from_dict(MODEL_CONFIG).build(init_device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch vs 0.1B Transformer:\n  missing={list(missing)}\n"
            f"  unexpected={list(unexpected)}")
    print("    [verify] strict state_dict match OK")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the source->target plan and exit (no downloads).")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="Restrict to these chinchilla multipliers (e.g. --only 1 2 4).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-convert/upload even if the target model.pt exists.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the olmo_core strict-load check before upload.")
    ap.add_argument("--workdir", default=None,
                    help="Scratch dir for downloads (default: a temp dir).")
    args = ap.parse_args()

    plan = build_plan(set(args.only) if args.only else None)
    print(f"Plan: {len(plan)} model(s)\n  SRC {SRC_PREFIX}\n  DST {DST_PREFIX}\n")
    for src, dst in plan:
        print(f"  {src.split('/')[-1]}\n      -> {dst}")
    print()

    if args.dry_run:
        return 0

    workdir = args.workdir or tempfile.mkdtemp(prefix="import100m_")
    os.makedirs(workdir, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "missing": 0}
    for src, dst in plan:
        print(f"[{dst}]")
        try:
            res = convert_one(src, dst, workdir, args.overwrite, not args.no_verify)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {exc}")
            res = "missing"
        counts[res] = counts.get(res, 0) + 1
    print(f"\nDone: {counts}")
    return 0 if counts["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
