# optimizer-study-copy — DCLM pretrain → CPT → perturbation

A slimmed copy of `optimizer-study`, scoped to **one experiment line**: pretrain
small Transformers (Muon vs AdamW) on **DCLM**, **continually pre-train (CPT) /
finetune** those models on downstream datasets, optionally **perturb** the
pretrained weights, and evaluate everything by **validation loss**
(`ModelEvaluation`). All the other families from the original repo (bioS
finetune, epoch-sweep, interpolation, weight-distance, diversity, confidence,
QA-group, typo, logit-margin, checkpoint-metrics) have been removed.

The training engine is the vendored OLMo-core fork in [`JOLMo/`](JOLMo/);
experiment *definitions* live in [`launch_jolmo/`](launch_jolmo/).

---

## 1. What the experiments are

```
pretrain-*  ─┬─►  cpt-*        (finetune the DCLM models on a dataset, e.g. musicpile)
             └─►  perturb-*    (add Gaussian noise to the DCLM weights)

eval-pretrain-*  /  eval-cpt-*  /  eval-perturb-*   →  validation-loss JSON
```

| Family | What it produces | Built from |
| --- | --- | --- |
| **Pretrain** | base `JolmoModel`s trained on DCLM, one per (chinchilla × optimizer × LR) | `pretraining_matrix.py` (`PT_LR`, `CHINCHILLAS`, `OPTIMIZERS`) |
| **CPT** | `CPTModel`s = each DCLM model continue-trained on a CPT dataset, sweeping CPT LR | `cpt.py` (`CPT_DATASETS`, `CPT_TOKENS`, `CPT_LR_SWEEP`) |
| **Perturb** | `PerturbedModel`s = DCLM weights + scaled Gaussian noise (std = γ·‖W‖_F), one per γ | `perturb.py` (`DEFAULT_GAMMAS`) |
| **Evals** | `ModelEvaluation` JSON = per-dataset **validation loss**, via `JOLMo/src/scripts/validate.py` | one per model in the family |

Artifacts are **content-addressed and idempotent**: an artifact whose output
already exists on GCS is skipped. Downstream artifacts (CPT, perturb, evals)
read the upstream model's uploaded `final-unsharded/model.pt`, so a stage's
dependencies are built first automatically when you launch it.

---

## 2. Setup (once per node)

```bash
# 1. Environment with the `experiments` framework + olmo_core (from JOLMo)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate optim-study

# 2. The training engine must be importable as `olmo_core`
pip install -e JOLMo[all]      # if not already installed in the env

# 3. GCS auth on the compute node (checkpoints + data live on gs://)
gcloud auth login              # account: catheri4@andrew.cmu.edu
# WANDB_API_KEY must be set for training/eval jobs (W&B is on by default)
```

Notes
- Run everything **from the repo root** (`optimizer-study-copy/`) using the
  `-m` module form. `python launch_jolmo/launcher.py …` fails with
  `ModuleNotFoundError: launch_jolmo`.
- Data is **not** downloaded by hand — each job pulls its DCLM/CPT shards from
  GCS to node-local scratch at run time (see §6).

---

## 3. Launching experiments

Stages are run by name through the `launch` subcommand of
[`launch_jolmo/launcher.py`](launch_jolmo/launcher.py):

```bash
cd /home/catheri4/optimizer-study-copy

# Submit a stage as SLURM jobs (dependencies built first, in tier order):
python -m launch_jolmo.launcher launch pretrain-muon-wsd
python -m launch_jolmo.launcher launch cpt          # builds pretrain deps if missing

# Preview WHAT would launch — no submission:
python -m launch_jolmo.launcher drylaunch cpt

# Run the stage directly on THIS node (no SLURM) — emits a self-contained
# bash script (env + download + worker + upload) for every missing artifact:
python -m launch_jolmo.launcher print eval-cpt | bash

# Several stages at once:
python -m launch_jolmo.launcher launch pretrain-muon-wsd cpt-muon eval-cpt
```

Useful flags: `--rerun` (ignore the `exists` check), `--head N` / `--tail N`
(first/last N artifacts), `--slurm KEY=VALUE` (override resources, e.g.
`--slurm time=12:00:00 gpus=4`), `--throttle N` (cap concurrent array tasks).
Other subcommands: `relaunch`, `cancel`, `cat` (tail a job log), `history`.

### Model-size profile (`OPTIM_SIZE`)

The launcher runs a size *profile* selected by the `OPTIM_SIZE` env var (default
`60M`). A profile (see [`launch_jolmo/sizes.py`](launch_jolmo/sizes.py)) bundles
the `experiments` project (its `project.json` `remote_path` = the GCS prefix),
the `model_type` (size tag in every run name), and the chinchilla budgets that
exist for that size. `launcher.py` and `pretraining_matrix.py` read the same
profile so they can't drift.

| `OPTIM_SIZE` | Project / GCS prefix | `model_type` | Chinchillas |
| --- | --- | --- | --- |
| `60M` (default) | `Optim-60M-tuning` | `0.06B` | 1–128 |
| `100M` | `Optim-100M-tuning` | `0.1B` | 1–16 |

```bash
# CPT + eval the imported optimal-LR 100M base models (chinchilla 1–16):
OPTIM_SIZE=100M python -m launch_jolmo.launcher drylaunch cpt   # preview
OPTIM_SIZE=100M python -m launch_jolmo.launcher launch cpt eval-cpt
```

The 100M base `JolmoModel`s are the optimal-LR checkpoints imported from jgai's
0.1B sweep by [`new_utils/import_100m_models.py`](new_utils/import_100m_models.py)
(converted to `final-unsharded/{model.pt,config.json}`); they already exist on
GCS, so CPT/eval treat them as built dependencies and never retrain them.

### Local pretraining without SLURM
[`run_local.py`](launch_jolmo/run_local.py) runs `JolmoModel` pretraining via
`torchrun` on the current node (project namespace `60m-muonxadamw`):

```bash
python -m launch_jolmo.run_local list           # enumerate models w/ indices
python -m launch_jolmo.run_local launch 0 1 2    # run models 0,1,2 in parallel
python -m launch_jolmo.run_local queue all --parallel 2
python -m launch_jolmo.run_local drylaunch 0     # print the generated config only
```

---

## 4. Stage catalog

All registered in [`launch_jolmo/launcher.py`](launch_jolmo/launcher.py)
(20 stages). Suffix `-adamw` / `-muon` selects the optimizer.

**Pretrain (DCLM)** — base `JolmoModel`s, one per (chinchilla × optimizer × LR):
| Stage | Does |
| --- | --- |
| `pretrain-adamw-wsd` / `pretrain-muon-wsd` | tuned-LR pretrain, WSD scheduler (the main sets) |
| `pretrain-adamw-cosine` / `pretrain-muon-cosine` | same, cosine scheduler |
| `pretrain-all-wsd` | **every** sweep LR (not just the optimal cell) — for LR tuning |

**CPT (finetune the DCLM models on a dataset)** — `CPTModel`s:
| Stage | Does |
| --- | --- |
| `cpt` | the default CPT set (= `cpt-muon`) |
| `cpt-adamw` | AdamW DCLM models, continue-trained with AdamW, LR sweep |
| `cpt-muon` | Muon DCLM models, continue-trained with Muon (adamw-component LR = 0.25×muon_lr) |
| `cpt-muon-adamw-ft` | Muon-pretrained models finetuned with **AdamW** (cross-optimizer) |
| `cpt-adamw-muon-ft` | AdamW-pretrained models finetuned with **Muon** (cross-optimizer) |

**Perturb** — `PerturbedModel`s over a γ grid (std = γ·‖W‖_F):
| Stage | Does |
| --- | --- |
| `perturb-adamw` / `perturb-muon` | perturb the DCLM-pretrained models, one model per γ in `DEFAULT_GAMMAS` |

**Interpolate** — `InterpolatedModel`s = `α·pretrained + (1−α)·finetuned`:
| Stage | Does |
| --- | --- |
| `interpolate` | for each specified finetuned (CPT) model, interpolate it with its pretrained base over `INTERP_ALPHAS` (0.2/0.4/0.6/0.8), one model per (ft model × α) |

**Evaluation** — `ModelEvaluation` (per-dataset **validation loss**):
| Stage | Scores |
| --- | --- |
| `eval-pretrain-adamw` / `eval-pretrain-muon` | the tuned pretrain models — on the diversity-v2 val sets **and held-out DCLM** |
| `eval-pretrain-all` | every LR-sweep pretrain model — diversity-v2 val sets **and held-out DCLM** |
| `eval-cpt` | the CPT models |
| `eval-cpt-muon-adamw-ft` / `eval-cpt-adamw-muon-ft` | the cross-optimizer CPT models |
| `eval-perturb-adamw` / `eval-perturb-muon` | the perturbed models |
| `eval-interpolate` | the interpolated models (forgetting loss + finetuning loss for the Pareto curve) |

The pretrain evals also report a **held-out DCLM** loss (label `DCLM_heldout`)
alongside the diversity-v2 sets, in the same `…-eval.json`. The held-out set is
`part-059/00004.npy` — the last shard of the last DCLM part. The sweep trains
only on the first ~2 parts (chinchilla-32 ≈ 2 parts), so this shard is never
seen. Each eval pulls only that one shard and scores `DCLM_HELDOUT_INSTANCES`
(8192) sequences; the diversity sets stay uncapped, so their numbers are
unchanged. Knobs live in `pretraining_matrix.py` (`DCLM_HELDOUT_*`); the cap is
plumbed through `ModelEvaluation.extra_val_chunks` / `extra_val_max_instances`.

Typical end-to-end run for one optimizer:
```bash
python -m launch_jolmo.launcher launch pretrain-muon-wsd
python -m launch_jolmo.launcher launch eval-pretrain-muon
python -m launch_jolmo.launcher launch cpt-muon        eval-cpt
python -m launch_jolmo.launcher launch perturb-muon    eval-perturb-muon
```

After the pretrain evals finish, plot each eval metric as a function of
Chinchilla (AdamW = blue, Muon = orange), written to `results/pretrain-eval/`:
```bash
python -m launch_jolmo.launcher launch eval-pretrain-adamw eval-pretrain-muon
python -m new_utils.plot_pretrain_eval     # → results/pretrain-eval/pretrain-eval-<metric>.png
```

**Divergence** — per-token KL/JSD vs. reference OLMo 2 on DCLM heldout:
| Stage | Does |
| --- | --- |
| `divergence-adamw` / `divergence-muon` / `divergence-all` | vs. **OLMo 2 32B** (`allenai/OLMo-2-0325-32B`, A100-80GB) |
| `divergence-13b-adamw` / `divergence-13b-muon` / `divergence-13b-all` | vs. **OLMo 2 13B** (`allenai/OLMo-2-1124-13B`, A100-40GB, **batch=8**) |
| `divergence-7b-adamw` / `divergence-7b-muon` / `divergence-7b-all` | vs. **OLMo 2 7B** base (`allenai/OLMo-2-1124-7B`, A100-40GB, **batch=8**) |
| `divergence-bases` | dependency resolution only (checkpoints already on GCS) |

Artifacts: `DivergenceEvaluation/{run}-vs-{reference-tag}-divergence.npz`.

After divergence stages finish, plot via the colm-moss-latex pipeline
(download → process → plot; one AdamW vs Muon overlay per Chinchilla):
```bash
cd colm-moss-latex/scripts
python download_divergence.py
python process_divergence.py
python plot_divergence.py --reference OLMo-2-0325-32B
python plot_divergence.py --reference OLMo-2-1124-13B
python plot_divergence.py --reference OLMo-2-1124-7B
python plot_divergence.py --reference OLMo-2-0325-32B --xlim 8 0 0.05 --xlim 16 0 0.1
```
Or run the full pipeline: `./run_pipeline.sh` (includes divergence stages).

---

## 5. Configuration knobs

### Pretraining — [`launch_jolmo/pretraining_matrix.py`](launch_jolmo/pretraining_matrix.py)
| Knob | Meaning |
| --- | --- |
| `MODEL_TYPE` | model size; options in `training.MODEL_ARCHS` (`0.03B`…`2.5B`). Size-dependent constants follow automatically. |
| `BASE_TOKENS` | chinchilla-1 token budget; **derived** from `MODEL_TYPE` (≈ `CHINCHILLA_MULT × params`). Pin an exact value in `BASE_TOKENS_OVERRIDE`. |
| `CHINCHILLAS` | list of token multipliers swept, e.g. `[1, 2, 4, …, 128]`. |
| `OPTIMIZERS` | which of `adamw` / `muon` to include. |
| `PT_LR_BY_MODEL` | tuned LR table **keyed by model size** → `PT_LR` for the active `MODEL_TYPE`. A missing size/cell falls back to `PT_LR_SWEEP`. |
| `SCHEDULER` | `wsd` (default) or `cosine`. |
| `GLOBAL_BATCH_SIZE` | tokens per optimizer step (1M for pretrain). |
| `UPLOAD_MODELS` | `False` → keep checkpoints local only (downstream CPT/perturb won't find them). |

### CPT — [`launch_jolmo/cpt.py`](launch_jolmo/cpt.py)
| Knob | Meaning |
| --- | --- |
| `CPT_DATASETS` | which datasets to finetune on (active: `musicpile`; `tulu`/`starcoder`/`alpaca`/`gsm8k`/… are commented out). Registry of paths is `CPT_DATASETS` in `training.py`. |
| `CPT_TOKENS` | tokens per CPT run (20M). |
| `CPT_LR_SWEEP` | CPT LR grid per optimizer. |
| `CPT_SCHEDULER` | `cosine`. |

### Perturbation — [`launch_jolmo/perturb.py`](launch_jolmo/perturb.py)
| Knob | Meaning |
| --- | --- |
| `DEFAULT_GAMMAS` | the γ grid (noise scale, std = γ·‖W‖_F), `1e-5 … 1e-3`. |

### Interpolation — [`launch_jolmo/interpolate.py`](launch_jolmo/interpolate.py)
| Knob | Meaning |
| --- | --- |
| `DEFAULT_ALPHAS` | the α grid for `α·pretrained + (1−α)·finetuned` (default `0.2/0.4/0.6/0.8`; α=0 is the finetuned model, α=1 the pretrained). |
| `interpolation_finetuned_models` (in `pretraining_matrix.py`) | the "specified finetuned models" set to interpolate (default: the active CPT set `cpt_models`). |

---

## 6. Data

Both DCLM and the CPT datasets already live on GCS and are pulled automatically
by each job (cached, `skip_existing`). You don't normally download anything.

- **DCLM pretrain** — `DCLM_NPY_GS` base prefix; the corpus is **60 `part-NNN`
  dirs × 5 shards (~19.85 B tokens/part, raw `uint32`) ≈ 1.19 T unique tokens**.
  Each run downloads **only as many whole parts as its token budget needs** via
  `_dclm_chunks_for_tokens()` (a chinchilla-1 run pulls 1 part ≈ 74 GiB, not the
  full ~4.7 TB). Cap = all 60 parts; the loader cycles if a run still exceeds them.
- **CPT / finetune** — `CPT_DATASETS` registry in `training.py` →
  `gs://…/OLMo/<dataset>/{train,val}/input_ids.npy`. `CPTModel.construct`
  downloads the chosen dataset at job time.

Manual staging scripts (only needed to add data **not** already on GCS — they
carry hardcoded paths from the original setup and need editing first):
`new_utils/download_dclm.py`, `new_utils/download_data.py`,
`new_utils/download_alpaca.py`, `new_utils/tokenize_data.py`, plus the
`new_utils/slurm_download_*.sh` / `job_scripts/download_data.sh` wrappers.

---

## 7. Outputs & plotting

- Checkpoints / models → `gs://cmu-gpucloud-catheri4/Optim-60M-tuning/<ArtifactClass>/<run_name>/`
  with `checkpoints/`, `final-unsharded/model.pt`, the generated YAML, `wandb/`.
- Eval JSON → `…/ModelEvaluation/<run_name>-eval.json` (per-dataset val loss).
- W&B is on by default (`WANDB_API_KEY` required); disable per-run via
  `wandb.enabled: false` in the generated YAML.

Plotters (read JSON straight from `gs://`, no GPU) in [`new_utils/`](new_utils/):
| Script | Plots |
| --- | --- |
| `plot_pretrain_eval.py` | Chinchilla → pretrain eval loss, one fig per metric (AdamW blue, Muon orange) → `results/pretrain-eval/` |
| `lr_sweep_dclm.py` | `eval`: submit DCLM held-out evals for every trained LR per Chinchilla; `plot`: DCLM loss → pretrain LR, one fig per Chinchilla → `results/pretrain-eval/lr-sweep/` |
| `plot_cpt.py` | tokens(B) → C4 perplexity, line per CPT-LR, Muon vs AdamW |
| `plot_perturb_loss.py` | perturbation γ → pretrain val loss, line per model |
| `plot_perturb_metrics.py` | γ → loss (raw + degradation), symlog x |
| `collect_eval_results.py` | `ModelEvaluation` JSON → CSV (filter by optimizer/dataset/CPT) |

Convention: **muon = orange, adamw = green** (auto-keyed off the run name).

---

## 8. Layout

```
launch_jolmo/        experiment definitions
  pretraining_matrix.py   DCLM pretrain sweep + CPT + perturbation + evals (the main file)
  cpt.py                  build_cpt_models / build_cpt_model_evaluations
  perturb.py              build_perturbed_models / build_perturbed_model_evaluations
  interpolate.py          build_interpolated_models / build_interpolated_model_evaluations
  training.py             JolmoModel, CPTModel, PerturbedModel, ModelEvaluation + YAML builders + MODEL_ARCHS
  launcher.py             stage registration + SLURM CLI  (THE entry point)
  run_local.py            local (no-SLURM) pretraining runner
  data.py / utils.py      Chunk + path helpers
  qa_eval_callback.py / weighted_train_module.py   engine callbacks referenced by the training config
JOLMo/               vendored OLMo-core training engine (validate.py, launch_from_yaml.py)
jolmo_configs/       hand-edited YAML templates
new_configs/         auto-generated YAML (pretrain/, cpt/) — don't hand-edit
new_utils/           plotting + data download/tokenize helpers
scripts/             generic engine utilities (HF conversion, unshard, dataloader, …)
job_scripts/         sbatch launchers (cpt_script.sh, pretrain_script.sh, download_data.sh)
```
