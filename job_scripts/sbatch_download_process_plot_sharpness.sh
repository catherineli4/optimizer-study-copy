#!/bin/bash
#SBATCH --job-name=jolmo-sharp-plot
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/sharpness_dl_plot/%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/sharpness_dl_plot/%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --gres=gpu:1
# Babel preempt QoS requires ≥1 GPU; this job is CPU-only but must request one.

# Download Jolmo SharpnessEvaluation JSONs → process → plot.
#
# Submit:
#   sbatch /home/catheri4/optimizer-study-copy/job_scripts/sbatch_download_process_plot_sharpness.sh

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
GS=gs://cmu-gpucloud-catheri4/Optim-60M-tuning/SharpnessEvaluation
LOCAL="${REPO}/results/SharpnessEvaluation"
OUT="${REPO}/results/sharpness"

mkdir -p "${REPO}/logs/sharpness_dl_plot" "${LOCAL}" "${OUT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study

cd "${REPO}"

echo "================================================================"
echo "1) Download SharpnessEvaluation from GCS"
echo "================================================================"
gsutil -m rsync -r "${GS}/" "${LOCAL}/"
echo "Local files: $(ls -1 "${LOCAL}"/*.json 2>/dev/null | wc -l)"

echo "================================================================"
echo "2) Process + plot → ${OUT}"
echo "================================================================"
python -m new_utils.plot_jolmo_sharpness \
  --local-dir "${LOCAL}" \
  --out-dir "${OUT}" \
  --x-axis chinchilla

echo "================================================================"
echo "Done."
echo "  JSONs:  ${LOCAL}/"
echo "  Table:  ${OUT}/final_results.json"
echo "  Plots:  ${OUT}/max_eigenvalue.png"
echo "          ${OUT}/trace.png"
echo "================================================================"
