"""Muon vs AdamW pretraining comparison, parameterized by model size (MODEL_TYPE).

Change MODEL_TYPE (see training.MODEL_ARCHS for the options) and the size-dependent
constants follow automatically: BASE_TOKENS (the Chinchilla-1 token budget) is derived
from the model size, and the tuned LR table is selected per model via PT_LR_BY_MODEL.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import math
from typing import List, Dict, Any
from experiments import ArtifactSet

from launch_jolmo.data import Chunk
from launch_jolmo.training import JolmoModel, ModelEvaluation
from launch_jolmo.cpt import build_cpt_models, build_cpt_model_evaluations
from launch_jolmo.perturb import build_perturbed_models, build_perturbed_model_evaluations


# ---------------------------------------------------------------------------
# GCS paths
# ---------------------------------------------------------------------------

DIVERSITY_V2_GS = "gs://cmu-gpucloud-jspringe/outputs/diversity.pretraining.v2"
# Base prefix of the DCLM shards; the dataset is split into 60 `part-NNN` dirs,
# each holding 5 shards (00000.npy … 00004.npy). The `_part-NNN` suffix is added
# when building all_chunks below.
DCLM_NPY_GS = "gs://cmu-gpucloud-jspringe/shared/datasets/OLMo/dclm/train/preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer"

# ---------------------------------------------------------------------------
# Model & schedule
# ---------------------------------------------------------------------------

TOKENIZER = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_TYPE = "0.06B" #0.1B, 0.3B
SCHEDULER = "wsd"
NUM_PROCESSES = 1
CHINCHILLAS: List[int] = [1, 2, 4, 8, 16, 32]   # ← list: add multiples e.g. [1, 2, 4]

GLOBAL_BATCH_SIZE = 1_048_576 // 16 #1M for pretrain, 256k for cpt
RANK_MICROBATCH_SIZE = GLOBAL_BATCH_SIZE // NUM_PROCESSES // 8
# Eval logits are (batch, seq_len, vocab_size) — keep small to avoid OOM.
# 4 seqs * 4096 tokens * 100352 vocab * 2 bytes ≈ 3.2 GiB, safely within the ~22 GiB free.
EVAL_RANK_MICROBATCH_SIZE = GLOBAL_BATCH_SIZE // NUM_PROCESSES // 64
VALIDATION_EVAL_INTERVAL = 95


# ---------------------------------------------------------------------------
# Model-size-dependent token budget
#
# The Chinchilla-1 token budget scales with model size (~CHINCHILLA_MULT tokens
# per parameter), so BASE_TOKENS is derived from MODEL_TYPE rather than hard-coded.
# Pin an exact, measured budget for a size in BASE_TOKENS_OVERRIDE; any size not
# listed falls back to CHINCHILLA_MULT × (params parsed from the size tag), rounded
# down to a whole number of global batches.
# ---------------------------------------------------------------------------

CHINCHILLA_MULT = 20    # Chinchilla-optimal tokens per parameter

BASE_TOKENS_OVERRIDE: Dict[str, int] = {
    "0.06B": 1_200_619_520,    # 20 × 60,030,976 measured params
}


def _approx_params(model_type: str) -> int:
    """Approximate parameter count parsed from the size tag, e.g. '0.06B' -> 60_000_000."""
    return int(float(model_type.rstrip("B")) * 1e9)


def _base_tokens_for(model_type: str) -> int:
    """Chinchilla-1 token budget for the given model size."""
    if model_type in BASE_TOKENS_OVERRIDE:
        return BASE_TOKENS_OVERRIDE[model_type]
    raw = CHINCHILLA_MULT * _approx_params(model_type)
    return (raw // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE


BASE_TOKENS = _base_tokens_for(MODEL_TYPE)    # tokens at chinchilla-1


def _tokens_for(chinchilla: int) -> Dict[str, Any]:
    """Return the schedule params that depend on chinchilla multiplier."""
    n_tokens = BASE_TOKENS * chinchilla
    total_steps = n_tokens // GLOBAL_BATCH_SIZE
    return {
        "n_tokens": n_tokens,
        "warmup_steps": total_steps // 10,
    }


# ---------------------------------------------------------------------------
# Optimal LR tables — keyed by MODEL_TYPE  (batch: 1M tokens)
#
# PT_LR_BY_MODEL[model_type][scheduler][opt][chinchilla] is the tuned LR:
#   AdamW -> learning_rate
#   Muon  -> (muon_lr, adamw_component_lr)
#
# Selecting MODEL_TYPE picks that size's table (PT_LR below). A model size with no
# entry, or a missing/None cell, falls back to the manual sweep PT_LR_SWEEP — useful
# when the optimal LR is not yet known. Add a table per size as you tune it.
# ---------------------------------------------------------------------------

PT_LR_BY_MODEL: Dict[str, Dict] = {
    "0.06B": {
        "wsd": {
            "adamw": {
                1: 1.4e-2,
                2: 1e-2,
                4: 7e-3,
                8: 1e-2,
                16: 7e-3,
                32: 7e-3,
                64: 1e-2,
                128: 1e-2
            },
            "muon": {
                # (muon_lr, adamw_component_lr)
                1: (1.4e-2,1.4e-2),
                2: (1.4e-2,1e-2),
                4: (1.4e-2,7e-3),
                8: (7e-3,1e-2),
                16: (5e-3,7e-3),
                32: (5e-3,7e-3),
                64: (5e-3,1e-2),
                128: (5e-3, 1e-2),

            },
        },
        "cosine": {
            "adamw": {
                1: 4e-2,   # unknown — will sweep PT_LR_SWEEP["adamw"]
                2: 2e-2,
                4: 2e-2,
                8: 1e-2,
            },
            "muon": {
                # (muon_lr, adamw_component_lr)
                1: (1e-2, 4e-2),   # unknown — will sweep PT_LR_SWEEP["muon"]
                2: (1e-2, 2e-2),
                4: (2e-2, 2e-2),
                8: (2e-2, 1e-2),
            },
        },
    },
    # Add a tuned table for each new MODEL_TYPE here; a size left out falls back
    # entirely to PT_LR_SWEEP.
}

# Tuned-LR table for the active model size (empty -> always use PT_LR_SWEEP).
PT_LR = PT_LR_BY_MODEL.get(MODEL_TYPE, {})

# Fallback sweep used when PT_LR[scheduler][opt][chinchilla] is None
PT_LR_SWEEP = {
    "adamw": [1e-3],
    "muon":  [(5e-3, 1e-3)],
}

# ---------------------------------------------------------------------------
# Which optimizers to include in this run — comment out to disable
# ---------------------------------------------------------------------------

OPTIMIZERS: List[str] = [
    "adamw",
    "muon",
]


# ---------------------------------------------------------------------------
# Upload toggle
#
# When False, trained checkpoints are NOT uploaded to GCS (gcloud). The final
# checkpoint is still written + unsharded locally, but nothing is pushed to the
# cloud bucket — useful for throwaway / debug runs. NOTE: downstream artifacts
# (perturbation, CPT, weight-distance, etc.) re-load the uploaded
# final-unsharded/model.pt, so they will NOT find a model from a no-upload run.
# ---------------------------------------------------------------------------

UPLOAD_MODELS: bool = True


# ---------------------------------------------------------------------------
# Training data  (toggle between Option A / B)
# ---------------------------------------------------------------------------

# Option A — Preprocessed DCLM .npy shards (active)
# Layout: 60 part-NNN dirs × 5 shards each (~19.85B tokens/part, raw uint32);
# all 60 = 300 shards ≈ 1.19T unique tokens. Because every train shard is
# downloaded to node-local scratch before training (~74 GiB/part), we include
# only as many WHOLE parts as a run's token budget needs — so a chinchilla-1
# run pulls 1 part (~74 GiB) instead of the full ~4.7 TB. Each model overrides
# train_chunks to its OWN budget (see _make_models); the module-level all_chunks
# default below is sized to the LARGEST chinchilla in the sweep.
DCLM_SHARDS_PER_PART = 5
DCLM_MAX_PARTS = 60
DCLM_TOKENS_PER_PART = 19_852_222_793   # measured: part-000 (5 shards, uint32)


def _dclm_chunks_for_tokens(n_tokens: int) -> tuple:
    """Return just enough DCLM shards (whole parts) to cover ``n_tokens`` unique
    tokens, capped at the full 60-part corpus. Avoids downloading the entire
    ~4.7 TB dataset for small runs (the loader cycles if a run still exceeds the
    included parts)."""
    parts = min(DCLM_MAX_PARTS, max(1, math.ceil(n_tokens / DCLM_TOKENS_PER_PART)))
    return tuple(
        Chunk(uri=f"{DCLM_NPY_GS}_part-{p:03d}/{i:05d}.npy")
        for p in range(parts)
        for i in range(DCLM_SHARDS_PER_PART)
    )


# Default = enough parts for the largest chinchilla in the sweep (per-model
# train_chunks below trim each run down to exactly its own budget).
all_chunks = _dclm_chunks_for_tokens(BASE_TOKENS * max(CHINCHILLAS))

# Option B — Original diversity-v2 .bin shards
# (Comment out Option A and uncomment below to switch back.)
# CHUNK_COUNTS = {"DCLM": 128}
# chunks_by_dataset = {
#     name: tuple(
#         Chunk(uri=f"{DIVERSITY_V2_GS}/PretrainingDataset/Tokenized/Vanilla/{name}/{name}-{i:04d}.bin")
#         for i in range(count)
#     )
#     for name, count in CHUNK_COUNTS.items()
# }
# all_chunks = tuple(c for chunks in chunks_by_dataset.values() for c in chunks)


# ---------------------------------------------------------------------------
# Validation data
# ---------------------------------------------------------------------------

diversity_val_chunks = tuple(
    (n, Chunk(uri=f"{DIVERSITY_V2_GS}/ValidationDataset/Tokenized/{n}.bin"))
    for n in ("C4_val", "Reddit_val", "Wiki_val", "Books_val")
)

# ---------------------------------------------------------------------------
# Shared model parameters
# ---------------------------------------------------------------------------

SHARED_MODEL_PARAMS = {
    "model_type": MODEL_TYPE,
    "tokenizer": TOKENIZER,
    "sequence_length": 1024, #4096 for pretrain,
    # Optimizer
    "weight_decay": 0.1,
    "betas": (0.9, 0.98),
    "max_grad_norm": 1.0,
    # Schedule (chinchilla-dependent keys — n_tokens, warmup_steps — added per model)
    "scheduler": SCHEDULER,
    "global_batch_size": GLOBAL_BATCH_SIZE,
    "rank_microbatch_size": RANK_MICROBATCH_SIZE,
    "eval_rank_microbatch_size": EVAL_RANK_MICROBATCH_SIZE,
    "validation_eval_interval": VALIDATION_EVAL_INTERVAL,
    # Parallelism & compilation
    "compile_model": True,
    "parallelism": "ddp",
    "num_processes": NUM_PROCESSES,
    "dp_param_dtype": "bfloat16",
    "dp_reduce_dtype": "float32",
    # Data
    "train_chunks": all_chunks,
    "validation_chunks": diversity_val_chunks,
    # Checkpointing & export
    "save_interval": 1000,
    "ephemeral_save_interval": 500,
    "unshard_checkpoint": True,
    "convert_to_hf": False,
    "upload": UPLOAD_MODELS,
    # Experiment
    "experiment_name": "Optim-60M-tuning",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lr_tag(lr: float) -> str:
    return f"{lr:.1e}".replace("e-0", "e-")


def _make_models(optimizers: List[str], scheduler: str = SCHEDULER) -> ArtifactSet:
    """Build one JolmoModel per (chinchilla × optimizer × LR) combo for the given scheduler.

    If PT_LR[scheduler][opt][chinchilla] is set, uses that single optimal LR.
    If it is None, falls back to sweeping PT_LR_SWEEP[opt].
    """
    models = []
    sched_table = PT_LR.get(scheduler, {})
    for chinchilla in CHINCHILLAS:
        schedule = _tokens_for(chinchilla)
        # Download only as many DCLM parts as THIS run's token budget needs.
        train_chunks = _dclm_chunks_for_tokens(schedule["n_tokens"])
        for opt in optimizers:
            best = sched_table.get(opt, {}).get(chinchilla)

            if opt == "adamw":
                lrs = [best] if best is not None else PT_LR_SWEEP.get("adamw", [])
                for lr in lrs:
                    tag = f"lr{_lr_tag(lr)}"
                    models.append(JolmoModel(
                        model_name=f"MuonExpt3-{MODEL_TYPE}-chinchilla-{chinchilla}-adamw-{tag}-{scheduler}",
                        **{**SHARED_MODEL_PARAMS, **schedule, "scheduler": scheduler, "train_chunks": train_chunks},
                        optimizer="adamw",
                        learning_rate=lr,
                    ))

            elif opt == "muon":
                pairs = [best] if best is not None else PT_LR_SWEEP.get("muon", [])
                for muon_lr, adamw_lr in pairs:
                    tag = f"muonlr{_lr_tag(muon_lr)}-adamwlr{_lr_tag(adamw_lr)}"
                    models.append(JolmoModel(
                        model_name=f"MuonExpt3-{MODEL_TYPE}-chinchilla-{chinchilla}-muon-{tag}-{scheduler}",
                        **{**SHARED_MODEL_PARAMS, **schedule, "scheduler": scheduler, "train_chunks": train_chunks},
                        optimizer="muon",
                        muon_lr=muon_lr,
                        learning_rate=adamw_lr,
                    ))

    return ArtifactSet(models)


def _make_all_lr_models(optimizers: List[str], scheduler: str = SCHEDULER) -> ArtifactSet:
    """Like _make_models but always uses ALL sweep LRs for every chinchilla."""
    models = []
    for chinchilla in CHINCHILLAS:
        schedule = _tokens_for(chinchilla)
        # Download only as many DCLM parts as THIS run's token budget needs.
        train_chunks = _dclm_chunks_for_tokens(schedule["n_tokens"])
        for opt in optimizers:
            if opt == "adamw":
                for lr in PT_LR_SWEEP.get("adamw", []):
                    tag = f"lr{_lr_tag(lr)}"
                    models.append(JolmoModel(
                        model_name=f"MuonExpt3-{MODEL_TYPE}-chinchilla-{chinchilla}-adamw-{tag}-{scheduler}",
                        **{**SHARED_MODEL_PARAMS, **schedule, "scheduler": scheduler, "train_chunks": train_chunks},
                        optimizer="adamw",
                        learning_rate=lr,
                    ))
            elif opt == "muon":
                for muon_lr, adamw_lr in PT_LR_SWEEP.get("muon", []):
                    tag = f"muonlr{_lr_tag(muon_lr)}-adamwlr{_lr_tag(adamw_lr)}"
                    models.append(JolmoModel(
                        model_name=f"MuonExpt3-{MODEL_TYPE}-chinchilla-{chinchilla}-muon-{tag}-{scheduler}",
                        **{**SHARED_MODEL_PARAMS, **schedule, "scheduler": scheduler, "train_chunks": train_chunks},
                        optimizer="muon",
                        muon_lr=muon_lr,
                        learning_rate=adamw_lr,
                    ))
    return ArtifactSet(models)


# ---------------------------------------------------------------------------
# Model sets  (per optimizer × per scheduler)
# ---------------------------------------------------------------------------

pretrain_adamw_wsd     = _make_models(["adamw"], "wsd")    if "adamw" in OPTIMIZERS else ArtifactSet([])
pretrain_adamw_cosine  = _make_models(["adamw"], "cosine") if "adamw" in OPTIMIZERS else ArtifactSet([])
pretrain_muon_wsd      = _make_models(["muon"],  "wsd")    if "muon"  in OPTIMIZERS else ArtifactSet([])
pretrain_muon_cosine   = _make_models(["muon"],  "cosine") if "muon"  in OPTIMIZERS else ArtifactSet([])

# All LR sweep models (includes every LR, not just the optimal one)
pretrain_all_wsd = _make_all_lr_models(OPTIMIZERS, "wsd")

# Convenience aliases
pretrain_adamw_models = pretrain_adamw_wsd
pretrain_muon_models  = pretrain_muon_wsd  + pretrain_muon_cosine


# ---------------------------------------------------------------------------
# CPT models  (via launch_jolmo/cpt.py)
# — always applied to all defined pretrained models
# ---------------------------------------------------------------------------

cpt_adamw_models = build_cpt_models(pretrain_adamw_wsd)
cpt_muon_models = build_cpt_models(pretrain_muon_wsd, muon_adamw_multiplier=0.25)
# Muon-pretrained models finetuned with AdamW CPT optimizer
cpt_muon_pretrain_adamw_ft = build_cpt_models(pretrain_muon_wsd, cpt_optimizers=["adamw"])
# AdamW-pretrained models finetuned with Muon CPT optimizer
cpt_adamw_pretrain_muon_ft = build_cpt_models(pretrain_adamw_wsd, cpt_optimizers=["muon"], muon_adamw_multiplier=0.25)
cpt_models = cpt_muon_models


# ---------------------------------------------------------------------------
# Perturbed models  (via launch_jolmo/perturb.py)
# Gaussian weight perturbation (std = γ/‖W‖_F) of the DCLM-pretrained models.
# ---------------------------------------------------------------------------

perturbed_adamw_models = build_perturbed_models(pretrain_adamw_wsd)
perturbed_muon_models  = build_perturbed_models(pretrain_muon_wsd)


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------

pretrain_adamw_wsd_evals   = ArtifactSet([ModelEvaluation(model=m) for m in pretrain_adamw_wsd])
pretrain_adamw_cosine_evals = ArtifactSet([ModelEvaluation(model=m) for m in pretrain_adamw_cosine])
pretrain_muon_wsd_evals    = ArtifactSet([ModelEvaluation(model=m) for m in pretrain_muon_wsd])
pretrain_muon_cosine_evals = ArtifactSet([ModelEvaluation(model=m) for m in pretrain_muon_cosine])

# Evals for all LR sweep models (all LRs, not just optimal)
pretrain_all_wsd_evals     = ArtifactSet([ModelEvaluation(model=m) for m in pretrain_all_wsd])

# Backward-compat aliases
pretrain_adamw_evals = pretrain_adamw_wsd_evals
pretrain_muon_evals  = pretrain_muon_wsd_evals

cpt_evals                  = build_cpt_model_evaluations(cpt_models)
cpt_muon_pretrain_adamw_ft_evals = build_cpt_model_evaluations(cpt_muon_pretrain_adamw_ft)
cpt_adamw_pretrain_muon_ft_evals = build_cpt_model_evaluations(cpt_adamw_pretrain_muon_ft)

# Perturbed-model loss evals (ModelEvaluation: pretrain val loss only).
perturbed_adamw_evals      = build_perturbed_model_evaluations(perturbed_adamw_models)
perturbed_muon_evals       = build_perturbed_model_evaluations(perturbed_muon_models)
