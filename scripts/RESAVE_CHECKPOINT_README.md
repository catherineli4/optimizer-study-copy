# Re-saving Checkpoints for Python 3.10 Compatibility

This script fixes the `ModuleNotFoundError: No module named 'pathlib._local'` error that occurs when trying to load a checkpoint saved with Python 3.13+ using Python 3.10.

## Quick Start

The easiest way is to use the wrapper script:

```bash
cd /home/catheri4/catastrophic-forgetting
./scripts/resave_checkpoint_wrapper.sh /scratch/catheri4/outputs/adamw/dHgFiHXg/PretrainedModel/OLMo-tk64B-adamw-lr3e-4-wd1e-1-bs256/final-unsharded
```

## Manual Usage

Alternatively, you can use the Python script directly with the `--use-python313` flag:

```bash
cd /home/catheri4/catastrophic-forgetting
/home/catheri4/miniconda3/envs/myenv310/bin/python3.10 scripts/resave_checkpoint.py \
    /scratch/catheri4/outputs/adamw/dHgFiHXg/PretrainedModel/OLMo-tk64B-adamw-lr3e-4-wd1e-1-bs256/final-unsharded \
    --use-python313
```

## Options

- `--files`: Specify which checkpoint files to re-save (default: train.pt optim.pt model.pt)
- `--all-files`: Re-save all .pt files in the directory
- `--no-backup`: Don't create backup files (not recommended)
- `--use-python313`: Use Python 3.13 to load files first (required for pathlib._local compatibility)
- `--python313-path`: Path to Python 3.13 executable (default: /home/catheri4/miniconda3/bin/python3.13)

## What it does

1. Creates a backup of the original checkpoint files (unless `--no-backup` is used)
2. Loads the checkpoint using Python 3.13 (which can read the pathlib._local format)
3. Re-saves the checkpoint using Python 3.10 (which will pickle it in a Python 3.10-compatible format)

## Example

```bash
# Re-save train.pt, optim.pt, and model.pt
python3.10 scripts/resave_checkpoint.py /path/to/checkpoint --use-python313

# Re-save only train.pt
python3.10 scripts/resave_checkpoint.py /path/to/checkpoint --files train.pt --use-python313

# Re-save all .pt files without backup
python3.10 scripts/resave_checkpoint.py /path/to/checkpoint --all-files --no-backup --use-python313
```

