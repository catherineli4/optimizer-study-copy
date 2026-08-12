#!/bin/bash
#SBATCH --job-name=muon-ckpt-diag
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/muon_ckpt_diag/%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/muon_ckpt_diag/%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=general
#SBATCH --gres=gpu:1

# Place each saved checkpoint of the muon chinchilla-4 run on its own in-loop
# wandb eval curve. The run's final-unsharded weights score 3.8077 on C4_val while
# the trainer's own eval at step 7600 read 3.7196 — i.e. the saved model looks like
# it predates the WSD decay. Re-unsharding several steps says whether only `final`
# is wrong (checkpointer) or every muon checkpoint is (state gathering).
#
# In-loop C4_val for reference: step6000 3.8111, step7000 ~3.793, step7500 3.7246,
# step7600 3.7196.
#
# Submit:
#   sbatch /home/catheri4/optimizer-study-copy/job_scripts/sbatch_diagnose_muon_checkpoint.sh

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
GS=gs://cmu-gpucloud-catheri4/Optim-100M-tuning/JolmoModel
VAL_GS=gs://cmu-gpucloud-jspringe/outputs/diversity.pretraining.v2/ValidationDataset/Tokenized/C4_val.bin
RUN=MuonExpt3-0.1B-chinchilla-4-muon-muonlr7.0e-3-adamwlr1.0e-2-wsd
WORK=/tmp/muon_ckpt_diag

mkdir -p "${REPO}/logs/muon_ckpt_diag" "${WORK}" "${REPO}/results"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study

gsutil ls "${GS}/${RUN}/" >/dev/null || {
  echo "gsutil auth failed — run 'gcloud auth login' on the login node first." >&2
  exit 1
}

gsutil -m cp -n "${VAL_GS}" "${WORK}/C4_val.bin"

# What step does the final checkpoint claim to be?
gsutil cp "${GS}/${RUN}/final/train/rank0.pt" "${WORK}/final_rank0.pt" 2>/dev/null || true
if [ -f "${WORK}/final_rank0.pt" ]; then
  python - <<'PY'
import torch
s = torch.load("/tmp/muon_ckpt_diag/final_rank0.pt", map_location="cpu", weights_only=False)
for k in ("global_step", "global_train_tokens_seen", "epoch"):
    if k in s:
        print(f"final/train recorded {k} = {s[k]}")
PY
fi

ARGS=()
for step in final step7500 step7000 step6000; do
  out="${WORK}/${step}-unsharded"
  if [ ! -f "${out}/model.pt" ]; then
    python "${REPO}/JOLMo/src/scripts/unshard.py" \
      "${GS}/${RUN}/${step}" "${out}" \
      --no-optim --overwrite --pre-download --work-dir "${WORK}/scratch-${step}"
    gsutil cp "${GS}/${RUN}/${step}/config.json" "${out}/config.json"
  fi
  ARGS+=(--checkpoint "${step}=${out}")
done

# The already-published artifact, for comparison against a fresh unshard of final/.
pub="${WORK}/published-final-unsharded"
mkdir -p "${pub}"
gsutil -m cp -n "${GS}/${RUN}/final-unsharded/model.pt" "${GS}/${RUN}/final-unsharded/config.json" "${pub}/"
ARGS+=(--checkpoint "published=${pub}")

python "${REPO}/scripts/eval_unsharded_c4.py" \
  "${ARGS[@]}" \
  --val-path "${WORK}/C4_val.bin" \
  --chunk-size 4096 --batch-size 4 \
  --output "${REPO}/results/muon_ckpt_diag.json"
