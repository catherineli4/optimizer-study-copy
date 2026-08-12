#!/usr/bin/env python3
"""Give the imported 0.1B (100M) models a sharded ``final/`` DCP checkpoint.

Why: ``new_utils/import_100m_models.py`` only wrote the *unsharded* artifact
(``final-unsharded/{model.pt,config.json}``) that the EVAL pipeline reads
(JOLMo/src/scripts/validate.py). But CPT / finetuning loads the base via
``trainer.load_checkpoint`` -> ``Checkpointer.latest_checkpoint`` -> a DCP
checkpoint at ``<base>/final``. Without it CPT dies with::

    FileNotFoundError: No checkpoints found in 'gs://.../JolmoModel/<run>'

A real trained model has both ``final/`` (DCP, model+optim+trainer state) and
``final-unsharded/``. We only have the latter, so here we reconstruct a
model-only ``final/`` from ``final-unsharded/model.pt``:

  1. download final-unsharded/{config.json, model.pt},
  2. build the Transformer from config["model"], strict-load model.pt,
  3. ``save_model_and_optim_state(<tmp>/final, model, optim=None)`` -> writes the
     DCP files (``.metadata`` + ``__*.distcp``) DIRECTLY into ``final/``,
  4. drop a ``final/config.json`` next to them (matches real layout / eval),
  5. ``gsutil cp -r`` the local ``final/`` up to ``<base>/final/``.

Writing the DCP straight into ``final/`` (rather than ``final/model_and_optim``)
means ``final/.metadata`` exists, which ``Checkpointer.dir_is_checkpoint`` accepts
as a "just model state, no trainer state" checkpoint. On load, CPT passes
``load_trainer_state=False`` / ``load_optim_state=False``, so the checkpointer's
fallback reads model weights from ``final/`` directly — no optim/trainer state
needed.

Run on a node in the optim-study env (needs torch + gsutil + GCS creds)::

    python -m new_utils.shard_100m_models --dry-run        # show the plan
    python -m new_utils.shard_100m_models                  # shard all imported bases
    python -m new_utils.shard_100m_models --only 4         # just chinchilla-4
    python -m new_utils.shard_100m_models --overwrite      # rebuild even if final/ exists
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

# Reuse the EXACT import plan so target run names line up with what was uploaded.
from new_utils.import_100m_models import DST_PREFIX, build_plan


def gs_exists(gs_path: str) -> bool:
    return subprocess.run(["gsutil", "-q", "stat", gs_path]).returncode == 0


def gs_cp_down(gs_path: str, local: str) -> None:
    subprocess.run(["gsutil", "-m", "cp", gs_path, local], check=True)


def gs_cp_up_dir(local_dir: str, gs_dir: str) -> None:
    # Upload the CONTENTS of local_dir into gs_dir/. Use rsync (not `cp -r *`)
    # because the wildcard skips dotfiles and DCP writes a hidden `.metadata`
    # marker that the checkpoint loader (dir_is_checkpoint) requires.
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", local_dir, gs_dir.rstrip("/")],
        check=True,
    )


def _init_single_process_group() -> None:
    """DCP save needs a process group; spin up a trivial 1-rank gloo group."""
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29547")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="gloo")


def shard_one(run_name: str, workdir: str, overwrite: bool) -> str:
    import torch
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.distributed.checkpoint import save_model_and_optim_state

    base = f"{DST_PREFIX}/{run_name}"
    final_meta = f"{base}/final/.metadata"
    if not overwrite and gs_exists(final_meta):
        print(f"  [skip] final/ already exists: {base}/final/")
        return "skipped"

    src_unsharded = f"{base}/final-unsharded"
    src_model = f"{src_unsharded}/model.pt"
    src_config = f"{src_unsharded}/config.json"
    if not gs_exists(src_model):
        print(f"  [MISS] no unsharded model: {src_model}")
        return "missing"

    run_dir = os.path.join(workdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    local_model = os.path.join(run_dir, "model.pt")
    local_config = os.path.join(run_dir, "config.json")
    local_final = os.path.join(run_dir, "final")

    print(f"  download {src_model}")
    gs_cp_down(src_model, local_model)
    gs_cp_down(src_config, local_config)

    with open(local_config, "r", encoding="utf-8") as f:
        exp_cfg = json.load(f)
    if "model" not in exp_cfg:
        raise RuntimeError(f"config.json missing 'model' section: {src_config}")

    print("  build Transformer + strict-load model.pt")
    model = TransformerConfig.from_dict(exp_cfg["model"]).build(init_device="cpu")
    state = torch.load(local_model, map_location="cpu")
    model.load_state_dict(state, strict=True)

    # Clean any stale local final/ so save_overwrite logic is simple.
    if os.path.exists(local_final):
        import shutil

        shutil.rmtree(local_final)

    print(f"  write DCP -> {local_final}/ (model-only)")
    save_model_and_optim_state(local_final, model, optim=None, save_overwrite=True)
    # Keep a config.json beside the shards (real `final/` has one; eval also looks here).
    with open(os.path.join(local_final, "config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_cfg, f, indent=2)

    print(f"  upload -> {base}/final/")
    gs_cp_up_dir(local_final, f"{base}/final")

    # Tidy up local scratch for this run.
    import shutil

    shutil.rmtree(run_dir, ignore_errors=True)
    return "done"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="Restrict to these chinchilla multipliers (e.g. --only 4).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rebuild final/ even if it already exists.")
    ap.add_argument("--workdir", default=None, help="Scratch dir (default: temp).")
    args = ap.parse_args()

    plan = build_plan(set(args.only) if args.only else None)
    run_names = [dst for _src, dst in plan]
    print(f"Plan: shard {len(run_names)} model(s) under {DST_PREFIX}\n")
    for r in run_names:
        print(f"  {r}")
    print()

    if args.dry_run:
        return 0

    _init_single_process_group()

    workdir = args.workdir or tempfile.mkdtemp(prefix="shard100m_")
    os.makedirs(workdir, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "missing": 0}
    for run_name in run_names:
        print(f"[{run_name}]")
        try:
            res = shard_one(run_name, workdir, args.overwrite)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {exc}")
            res = "missing"
        counts[res] = counts.get(res, 0) + 1
    print(f"\nDone: {counts}")
    return 0 if counts["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
