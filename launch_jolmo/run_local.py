#!/usr/bin/env python3
"""Run JolmoModel pretraining locally on the current node (no Slurm).

Uses the same stage names and --artifacts filters as launch_jolmo/launcher.py::

    python -m launch_jolmo.run_local stages
    python -m launch_jolmo.run_local list pretrain-all-wsd
    python -m launch_jolmo.run_local list pretrain-all-wsd --optimizer adamw
    python -m launch_jolmo.run_local queue pretrain-all-wsd --optimizer adamw
    OPTIM_SIZE=100M python -m launch_jolmo.run_local launch pretrain-adamw-wsd

Commands: launch, drylaunch, queue, list, stages
"""

import argparse
import multiprocessing
import os
import subprocess
import sys
import yaml

from experiments import Project

# Model-size profile selected by $OPTIM_SIZE (default 60M). The project's
# project.json sets remote_path = the GCS prefix, so this routes all artifacts to
# the right study bucket (Optim-60M-tuning / Optim-100M-tuning / Optim-300M-tuning),
# matching launcher.py. pretraining_matrix reads the same profile for MODEL_TYPE /
# CHINCHILLAS, so they stay in sync. Examples:
#   OPTIM_SIZE=100M python -m launch_jolmo.run_local launch all
#   OPTIM_SIZE=300M python -m launch_jolmo.run_local launch pretrain-all-wsd
from launch_jolmo.sizes import active_profile

_SIZE, _PROFILE = active_profile()
Project.init(_PROFILE["project"])

from launch_jolmo.pretraining_matrix import (
    pretrain_adamw_wsd,
    pretrain_adamw_cosine,
    pretrain_muon_wsd,
    pretrain_muon_cosine,
    pretrain_all_wsd,
)
# 60M 4-chinchilla PT Sweep — separate experiment, PTSweep60M-* names (own GCS
# subtree). Run these with OPTIM_SIZE=60M; see launch_jolmo/pt_sweep_60m_chin4.py.
from launch_jolmo.pt_sweep_60m_chin4 import (
    pt60m4_lr_sweep,
    pt60m4_wd_sweep,
    pt60m4_bs_sweep,
    pt60m4_all,
    pt60m4_cpt_lr_sweep,
    pt60m4_cpt_wd_sweep,
    pt60m4_cpt_bs_sweep,
    pt60m4_cpt_all,
)
from launch_jolmo.training import JolmoModel
from launch_jolmo.utils import local_path, local_cache_path, remote_path
from launch_jolmo.training import _resolve_chunk_path, _download_chunk_dirs


# Named pretraining stages, mirroring the Slurm launcher (launch_jolmo/launcher.py)
# so the same run commands work locally. `pretrain-all-wsd` is the full LR sweep;
# the others are the tuned/optimal-LR cells per chinchilla.
STAGES: "dict[str, list]" = {
    "pretrain-adamw-wsd":    list(pretrain_adamw_wsd),
    "pretrain-adamw-cosine": list(pretrain_adamw_cosine),
    "pretrain-muon-wsd":     list(pretrain_muon_wsd),
    "pretrain-muon-cosine":  list(pretrain_muon_cosine),
    "pretrain-all-wsd":      list(pretrain_all_wsd),
    # --- 60M 4-chinchilla PT Sweep (run in this order) ---
    "pt60m4-lr-sweep":       list(pt60m4_lr_sweep),
    "pt60m4-wd-sweep":       list(pt60m4_wd_sweep),
    "pt60m4-bs-sweep":       list(pt60m4_bs_sweep),
    "pt60m4-all":            list(pt60m4_all),
    "pt60m4-cpt-lr-sweep":    list(pt60m4_cpt_lr_sweep),
    "pt60m4-cpt-wd-sweep":    list(pt60m4_cpt_wd_sweep),
    "pt60m4-cpt-bs-sweep":    list(pt60m4_cpt_bs_sweep),
    "pt60m4-cpt-all":         list(pt60m4_cpt_all),
}


def select_models(stage, optimizer=None, head=None, tail=None, index=None):
    """Resolve a stage name (+ optional filters) into a list of models.

    Mirrors the launcher's selection: a stage plus optional ``--head`` / ``--tail``
    slicing. ``--optimizer`` is a local convenience filter (adamw/muon) since all
    pretrain artifacts share the same JolmoModel class name.
    """
    if stage not in STAGES:
        print(f"Error: unknown stage '{stage}'. Available stages:")
        for name in STAGES:
            print(f"  - {name}")
        sys.exit(1)

    models = list(STAGES[stage])

    if optimizer:
        opts = {o.lower() for o in optimizer}
        models = [m for m in models if m.optimizer.lower() in opts]

    if head is not None:
        models = models[:head]
    if tail is not None:
        models = models[-tail:]

    if index:
        picked = []
        for i in index:
            if i < 0 or i >= len(models):
                print(f"Error: index {i} out of range 0-{len(models) - 1} for the selected set")
                sys.exit(1)
            picked.append(models[i])
        models = picked

    return models


def print_models(models):
    # JolmoModel has .model_name; CPTModel (the finetune artifact) names itself
    # via .run_name instead, so fall back rather than raising AttributeError.
    for i, m in enumerate(models):
        name = getattr(m, "model_name", None) or getattr(m, "run_name", type(m).__name__)
        print(f"  [{i}] {name}  (optimizer={m.optimizer}, lr={m.learning_rate})")


def build_config_and_download(model: JolmoModel, skip_download: bool = False):
    """Generate the YAML config and download data. Returns the config path."""
    data_dir = local_path()
    cache_dir = local_cache_path()

    save_folder = remote_path(model.relpath)
    output_dir = os.path.join(data_dir, model.relpath)
    dataset_cache_dir = os.path.join(cache_dir, "datasets")
    work_dir = os.path.join(cache_dir, "training", model.run_name)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dataset_cache_dir, exist_ok=True)

    # Download data
    if not skip_download:
        all_chunks = list(model.train_chunks) + [c for _, c in model.validation_chunks]
        for chunk in all_chunks:
            local = _resolve_chunk_path(chunk, dataset_cache_dir)
            if chunk.uri.startswith("gs://"):
                parent_gs = chunk.uri.rsplit("/", 1)[0]
                parent_local = os.path.dirname(local)
                if not os.path.exists(local):
                    os.makedirs(parent_local, exist_ok=True)
                    print(f"Downloading {parent_gs} -> {parent_local}")
                    subprocess.run(
                        ["gsutil", "-m", "rsync", "-r", parent_gs, parent_local],
                        check=True,
                    )
            elif chunk.uri.startswith("/"):
                if not os.path.exists(local):
                    os.makedirs(os.path.dirname(local), exist_ok=True)
                    os.symlink(chunk.uri, local)

    # Resolve paths
    train_paths = [_resolve_chunk_path(c, dataset_cache_dir) for c in model.train_chunks]
    val_datasets = {
        label: [_resolve_chunk_path(c, dataset_cache_dir)]
        for label, c in model.validation_chunks
    }

    # Generate YAML
    yaml_config = model._build_yaml_config(save_folder, train_paths, val_datasets, work_dir)
    config_path = os.path.join(output_dir, "config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    print(f"Config written to: {config_path}")
    return config_path


def _add_stage_args(p):
    """Shared stage/filter args, mirroring the Slurm launcher CLI."""
    p.add_argument("stage", help="Pretraining stage name (same as launcher.py, e.g. pretrain-all-wsd)")
    p.add_argument("--optimizer", nargs="*", choices=["adamw", "muon"], default=None,
                   help="Filter to optimizer(s) within the stage (local convenience)")
    p.add_argument("--head", type=int, default=None, help="Run only the first N models in the stage")
    p.add_argument("--tail", type=int, default=None, help="Run only the last N models in the stage")
    p.add_argument("--index", type=int, nargs="*", default=None,
                   help="Run only these indices within the (filtered) stage list")
    p.add_argument("--skip-download", action="store_true", help="Skip data download")


def main():
    parser = argparse.ArgumentParser(
        description="Run JolmoModel pretraining locally on the current node.",
        epilog=(
            "Stage names match launch_jolmo/launcher.py. Examples:\n"
            "  python -m launch_jolmo.run_local list pretrain-all-wsd\n"
            "  python -m launch_jolmo.run_local launch pretrain-all-wsd --optimizer adamw\n"
            "  python -m launch_jolmo.run_local queue pretrain-adamw-wsd\n"
            "  OPTIM_SIZE=100M python -m launch_jolmo.run_local launch pretrain-all-wsd --optimizer adamw"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # launch
    p_launch = subparsers.add_parser("launch", help="Launch training on the current node")
    _add_stage_args(p_launch)
    p_launch.add_argument("--nproc", type=int, default=None, help="Override GPUs per model")

    # drylaunch
    p_dry = subparsers.add_parser("drylaunch", help="Generate config and print it (no training)")
    _add_stage_args(p_dry)

    # queue
    p_queue = subparsers.add_parser("queue", help="Queue models, run N at a time sequentially")
    _add_stage_args(p_queue)
    p_queue.add_argument("--parallel", type=int, default=1, help="How many models to run at once (default: 1)")
    p_queue.add_argument("--nproc", type=int, default=None, help="Override GPUs per model")

    # list
    p_list = subparsers.add_parser("list", help="List models in a stage")
    p_list.add_argument("stage", help="Pretraining stage name")
    p_list.add_argument("--optimizer", nargs="*", choices=["adamw", "muon"], default=None,
                        help="Filter to optimizer(s) within the stage")

    # stages
    p_stages = subparsers.add_parser("stages", help="List available pretraining stage names")

    args = parser.parse_args()

    print(f"[size={_SIZE}] project={_PROFILE['project']} -> {Project.config.remote_path}")

    if args.command == "stages" or args.command is None:
        print("Available pretraining stages:")
        for name, models in STAGES.items():
            print(f"  {name:25s}  ({len(models)} model(s))")
        if args.command is None:
            print("\nCommands: launch, drylaunch, queue, list, stages")
        return

    if args.command == "list":
        selected = select_models(args.stage, optimizer=args.optimizer)
        print(f"[stage={args.stage}] {len(selected)} model(s):")
        print_models(selected)
        return

    selected = select_models(
        args.stage,
        optimizer=args.optimizer,
        head=args.head,
        tail=args.tail,
        index=args.index,
    )
    print(f"[stage={args.stage}] Selected {len(selected)} model(s):")
    for m in selected:
        print(f"  - {m.model_name}")

    if args.command == "launch":
        run(selected, nproc=args.nproc, skip_download=args.skip_download)
    elif args.command == "queue":
        queue(selected, parallel=args.parallel, nproc=args.nproc, skip_download=args.skip_download)
    elif args.command == "drylaunch":
        for model in selected:
            dryrun(model, skip_download=args.skip_download)


def _run_single(model, nproc=None, skip_download=False, gpu_offset=0):
    """Run a single model. gpu_offset sets CUDA_VISIBLE_DEVICES for parallel runs."""
    os.environ["WANDB_API_KEY"] = Project.config.wandb_api_key
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    nproc = nproc or model.num_processes
    if gpu_offset > 0 or nproc < 8:
        gpus = ",".join(str(gpu_offset + i) for i in range(nproc))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print(f"[{model.model_name}] Using GPUs: {gpus}")

    config_path = build_config_and_download(model, skip_download=skip_download)

    launch_script = os.path.join(Project.config.jolmo_path, "src", "scripts", "launch_from_yaml.py")

    cmd = [
        "torchrun",
        "--rdzv-endpoint=localhost:0",
        "--rdzv-backend=c10d",
        f"--nproc_per_node={nproc}",
        launch_script,
        config_path,
    ]
    print(f"[{model.model_name}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run(models, nproc=None, skip_download=False):
    """Launch one or more models. Multiple models run in parallel processes."""
    if len(models) == 1:
        _run_single(models[0], nproc=nproc, skip_download=skip_download)
        return

    nproc_per = nproc or models[0].num_processes
    total_gpus = int(os.environ.get("TOTAL_GPUS", nproc_per * len(models)))
    gpus_per_model = total_gpus // len(models)

    if gpus_per_model < 1:
        print(f"Error: not enough GPUs ({total_gpus}) for {len(models)} models")
        sys.exit(1)

    print(f"Launching {len(models)} models in parallel ({gpus_per_model} GPU(s) each, {total_gpus} total)")

    processes = []
    for i, model in enumerate(models):
        gpu_offset = i * gpus_per_model
        p = multiprocessing.Process(
            target=_run_single,
            args=(model,),
            kwargs={"nproc": gpus_per_model, "skip_download": skip_download, "gpu_offset": gpu_offset},
        )
        p.start()
        processes.append((model.model_name, p))

    failed = []
    for name, p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append(name)

    if failed:
        print(f"\nFailed models: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(models)} models completed successfully.")


def queue(models, parallel=1, nproc=None, skip_download=False):
    """Run models in batches. Waits for a batch to finish before starting the next."""
    total_gpus_available = int(os.environ.get("TOTAL_GPUS", (nproc or models[0].num_processes) * parallel))
    gpus_per_model = total_gpus_available // parallel

    if gpus_per_model < 1:
        print(f"Error: not enough GPUs ({total_gpus_available}) for {parallel} parallel models")
        sys.exit(1)

    batches = [models[i:i + parallel] for i in range(0, len(models), parallel)]
    print(f"Queued {len(models)} model(s) in {len(batches)} batch(es) "
          f"({parallel} parallel, {gpus_per_model} GPU(s) each)")

    all_failed = []
    for batch_idx, batch in enumerate(batches):
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{len(batches)}: {', '.join(m.model_name for m in batch)}")
        print(f"{'='*60}")

        if len(batch) == 1:
            try:
                _run_single(batch[0], nproc=nproc or gpus_per_model, skip_download=skip_download)
            except Exception as e:
                print(f"Failed: {batch[0].model_name}: {e}")
                all_failed.append(batch[0].model_name)
        else:
            processes = []
            for i, model in enumerate(batch):
                gpu_offset = i * gpus_per_model
                p = multiprocessing.Process(
                    target=_run_single,
                    args=(model,),
                    kwargs={"nproc": gpus_per_model, "skip_download": skip_download, "gpu_offset": gpu_offset},
                )
                p.start()
                processes.append((model.model_name, p))

            for name, p in processes:
                p.join()
                if p.exitcode != 0:
                    all_failed.append(name)

    print(f"\n{'='*60}")
    if all_failed:
        print(f"Done. Failed models: {', '.join(all_failed)}")
        sys.exit(1)
    else:
        print(f"Done. All {len(models)} model(s) completed successfully.")


def dryrun(model, skip_download=False):
    """Generate config and print it without launching training."""
    os.environ["WANDB_API_KEY"] = Project.config.wandb_api_key
    config_path = build_config_and_download(model, skip_download=skip_download)
    print("\n--- Generated config ---")
    with open(config_path) as f:
        print(f.read())
    return config_path


if __name__ == "__main__":
    main()
