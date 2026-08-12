#!/bin/bash
#SBATCH --job-name=muon-ckpt-diff
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/muon_ckpt_diag/diff-%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/muon_ckpt_diag/diff-%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --nodelist=babel-w9-24

# Reuses the unsharded checkpoints left in /tmp by sbatch_diagnose_muon_checkpoint.sh
# (hence the nodelist pin), and reports which tensors still move between saves.
#
# Submit:
#   sbatch /home/catheri4/optimizer-study-copy/job_scripts/sbatch_diff_muon_checkpoints.sh

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
WORK=/tmp/muon_ckpt_diag

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study

ls -la "${WORK}" || { echo "scratch gone; re-run sbatch_diagnose_muon_checkpoint.sh" >&2; exit 1; }

for pair in "step6000 step7000" "step7000 step7500" "step7500 final"; do
  set -- $pair
  echo "================================================================"
  echo "  $1  ->  $2"
  echo "================================================================"
  python "${REPO}/scripts/diff_checkpoint_params.py" \
    --a "${WORK}/$1-unsharded" --b "${WORK}/$2-unsharded"
done
