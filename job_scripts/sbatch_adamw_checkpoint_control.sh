#!/bin/bash
#SBATCH --job-name=adamw-ckpt-ctrl
#SBATCH --output=/home/catheri4/optimizer-study-copy/logs/muon_ckpt_diag/ctrl-%j.out
#SBATCH --error=/home/catheri4/optimizer-study-copy/logs/muon_ckpt_diag/ctrl-%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=general
#SBATCH --gres=gpu:1

# Control for the muon checkpoint anomaly. The muon run's saved checkpoints move
# 9.8% (Frobenius, mean over Muon-group matrices) between step7500 and final even
# though the LR decays to 0 there, and they score ~0.08 CE worse than the trainer's
# own in-loop eval. Does the AdamW run at the same steps behave differently?
#
# In-loop C4_val for the adamw run: step7000 3.8720, step7500 3.7619, step7600 3.7445.
#
# Submit:
#   sbatch /home/catheri4/optimizer-study-copy/job_scripts/sbatch_adamw_checkpoint_control.sh

set -euo pipefail

REPO=/home/catheri4/optimizer-study-copy
GS=gs://cmu-gpucloud-catheri4/Optim-100M-tuning/JolmoModel
VAL_GS=gs://cmu-gpucloud-jspringe/outputs/diversity.pretraining.v2/ValidationDataset/Tokenized/C4_val.bin
RUN=MuonExpt3-0.1B-chinchilla-4-adamw-lr1.0e-2-wsd
MUON=MuonExpt3-0.1B-chinchilla-4-muon-muonlr7.0e-3-adamwlr1.0e-2-wsd
WORK=/tmp/adamw_ckpt_ctrl

mkdir -p "${REPO}/logs/muon_ckpt_diag" "${WORK}" "${REPO}/results"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study

gsutil -m cp -n "${VAL_GS}" "${WORK}/C4_val.bin"

ARGS=()
for step in final step7500 step7000; do
  out="${WORK}/${step}-unsharded"
  if [ ! -f "${out}/model.pt" ]; then
    python "${REPO}/JOLMo/src/scripts/unshard.py" \
      "${GS}/${RUN}/${step}" "${out}" \
      --no-optim --overwrite --pre-download --work-dir "${WORK}/scratch-${step}"
    gsutil cp "${GS}/${RUN}/${step}/config.json" "${out}/config.json"
  fi
  ARGS+=(--checkpoint "${step}=${out}")
done

echo "================================================================"
echo " AdamW saved checkpoints vs its own in-loop curve"
echo "================================================================"
python "${REPO}/scripts/eval_unsharded_c4.py" \
  "${ARGS[@]}" \
  --val-path "${WORK}/C4_val.bin" \
  --chunk-size 4096 --batch-size 4 \
  --output "${REPO}/results/adamw_ckpt_ctrl.json"

for pair in "step7000 step7500" "step7500 final"; do
  set -- $pair
  echo "================================================================"
  echo "  AdamW weight movement: $1 -> $2"
  echo "================================================================"
  python "${REPO}/scripts/diff_checkpoint_params.py" \
    --a "${WORK}/$1-unsharded" --b "${WORK}/$2-unsharded" --top 5
done

# What global_step does each muon checkpoint's trainer state claim?
for step in final step7500; do
  gsutil -q cp "${GS}/${MUON}/${step}/train/rank0.pt" "${WORK}/${step}-rank0.pt"
  python - "${WORK}/${step}-rank0.pt" "${step}" <<'PY'
import sys, torch
path, label = sys.argv[1], sys.argv[2]
s = torch.load(path, map_location="cpu", weights_only=False)
print(f"[{label}] top-level keys: {list(s)[:12]}")
for k in ("global_step", "global_train_tokens_seen", "epoch"):
    if isinstance(s, dict) and k in s:
        print(f"[{label}] {k} = {s[k]}")
PY
done
