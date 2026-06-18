#!/usr/bin/env python3
"""Run a JolmoModel training job locally on the current node (no Slurm).

Usage (from inside an interactive node):
    python launch_jolmo/run_local.py launch 0          # launch model 0
    python launch_jolmo/run_local.py launch 0 1 2      # launch models 0,1,2 in parallel
    python launch_jolmo/run_local.py launch all         # launch all models in parallel
    python launch_jolmo/run_local.py queue 0 1 2 3      # queue models, run --parallel at a time
    python launch_jolmo/run_local.py queue all --parallel 2  # run 2 at a time until all done
    python launch_jolmo/run_local.py drylaunch 0        # print config without running
    python launch_jolmo/run_local.py list               # list available models

The script reuses the same Project config, data download, and YAML generation
logic as the Slurm-based launcher, but calls torchrun directly.
"""

import argparse
import multiprocessing
import os
import subprocess
import sys
import yaml

from experiments import Project

Project.init("60m-muonxadamw")

from launch_jolmo.pretraining_matrix import (
    pretrain_adamw_models,
    pretrain_muon_models,
)
from launch_jolmo.training import JolmoModel
from launch_jolmo.utils import local_path, local_cache_path, remote_path
from launch_jolmo.training import _resolve_chunk_path, _download_chunk_dirs


def get_all_models():
    return list(pretrain_adamw_models) + list(pretrain_muon_models)


def print_models(models):
    for i, m in enumerate(models):
        print(f"  [{i}] {m.model_name}  (optimizer={m.optimizer}, lr={m.learning_rate})")


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


def parse_model_indices(raw_indices, num_models):
    """Parse model indices from CLI args. Supports integers and 'all'."""
    if not raw_indices:
        # Interactive prompt
        print("Available models:")
        print_models(get_all_models())
        choice = input("\nSelect model index (or comma-separated, or 'all'): ").strip()
        raw_indices = choice.replace(",", " ").split()

    indices = []
    for val in raw_indices:
        if val.lower() == "all":
            return list(range(num_models))
        try:
            idx = int(val)
            if idx < 0 or idx >= num_models:
                print(f"Error: index {idx} out of range 0-{num_models - 1}")
                sys.exit(1)
            indices.append(idx)
        except ValueError:
            print(f"Invalid index: {val}")
            sys.exit(1)
    return indices


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


def main():
    parser = argparse.ArgumentParser(description="Run JolmoModel training locally on the current node.")
    subparsers = parser.add_subparsers(dest="command")

    # launch
    p_launch = subparsers.add_parser("launch", help="Launch training on the current node")
    p_launch.add_argument("model_indices", nargs="*", help="Model indices, or 'all' (interactive if omitted)")
    p_launch.add_argument("--skip-download", action="store_true", help="Skip data download")
    p_launch.add_argument("--nproc", type=int, default=None, help="Override GPUs per model")

    # drylaunch
    p_dry = subparsers.add_parser("drylaunch", help="Generate config and print it (no training)")
    p_dry.add_argument("model_indices", nargs="*", help="Model indices, or 'all' (interactive if omitted)")
    p_dry.add_argument("--skip-download", action="store_true", help="Skip data download")

    # queue
    p_queue = subparsers.add_parser("queue", help="Queue models, run N at a time sequentially")
    p_queue.add_argument("model_indices", nargs="*", help="Model indices, or 'all' (interactive if omitted)")
    p_queue.add_argument("--parallel", type=int, default=1, help="How many models to run at once (default: 1)")
    p_queue.add_argument("--skip-download", action="store_true", help="Skip data download")
    p_queue.add_argument("--nproc", type=int, default=None, help="Override GPUs per model")

    # list
    subparsers.add_parser("list", help="List available models")

    args = parser.parse_args()
    all_models = get_all_models()

    if args.command == "list" or args.command is None:
        print("Available models:")
        print_models(all_models)
        if args.command is None:
            print("\nCommands: launch, drylaunch, list")
        return

    indices = parse_model_indices(args.model_indices, len(all_models))
    selected = [all_models[i] for i in indices]
    print(f"Selected {len(selected)} model(s):")
    for m in selected:
        print(f"  - {m.model_name}")

    if args.command == "launch":
        run(selected, nproc=args.nproc, skip_download=args.skip_download)
    elif args.command == "queue":
        queue(selected, parallel=args.parallel, nproc=args.nproc, skip_download=args.skip_download)
    elif args.command == "drylaunch":
        for model in selected:
            dryrun(model, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
