#!/bin/bash
#SBATCH --job-name=plot-logit-cosine
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/logit_cosine_dl_plot/%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/logit_cosine_dl_plot/%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --gres=gpu:1

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
GCS_DIR="${GCS_DIR:-gs://cmu-gpucloud-catheri4/Optim-60M-tuning/LogitCosineEvaluation}"
LOCAL_DIR="${REPO}/results/LogitCosineEvaluation"
OUT_DIR="${REPO}/results/logit_cosine"

mkdir -p "${REPO}/logs/logit_cosine_dl_plot" "${LOCAL_DIR}" "${OUT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study
cd "${REPO}"

echo "Downloading from ${GCS_DIR} → ${LOCAL_DIR}"
gsutil -m rsync -r "${GCS_DIR}/" "${LOCAL_DIR}/" || true

python -m new_utils.plot_logit_cosine \
  --local-dir "${LOCAL_DIR}" \
  --out-dir "${OUT_DIR}"

echo "Done. Plots in ${OUT_DIR}/"
ls -la "${OUT_DIR}/"
