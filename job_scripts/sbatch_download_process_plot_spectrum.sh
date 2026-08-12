#!/bin/bash
#SBATCH --job-name=jolmo-spectrum-plot
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/spectrum_dl_plot/%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/spectrum_dl_plot/%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --gres=gpu:1
# Babel preempt QoS requires ≥1 GPU; this job is CPU-only but must request one.

# Download Jolmo spectral-density JSONs → process → plot.
#
# Submit:
#   sbatch /home/catheri4/optimizer-study-copy/job_scripts/sbatch_download_process_plot_spectrum.sh

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
GS=gs://cmu-gpucloud-catheri4/Optim-60M-tuning/SharpnessEvaluation
LOCAL="${REPO}/results/SharpnessEvaluation"
OUT="${REPO}/results/hessian_spectrum"

mkdir -p "${REPO}/logs/spectrum_dl_plot" "${LOCAL}" "${OUT}"

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
python -m new_utils.hessian_spectrum \
  --local-dir "${LOCAL}" \
  --out-dir "${OUT}"

echo "================================================================"
echo "Done."
echo "  JSONs:  ${LOCAL}/"
echo "  Table:  ${OUT}/final_results.json"
echo "  Plots:  ${OUT}/spectral_density.png"
echo "          ${OUT}/entropy_erank.png"
echo "================================================================"
