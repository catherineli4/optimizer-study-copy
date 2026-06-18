# ---------------------------------------------------------------------------
# Project and imports
# ---------------------------------------------------------------------------

from experiments import Project, SlurmExecutor

Project.init("Optim-60M-tuning")

from launch_jolmo.pretraining_matrix import (
    pretrain_adamw_wsd,
    pretrain_adamw_cosine,
    pretrain_muon_wsd,
    pretrain_muon_cosine,
    pretrain_all_wsd,  # all LR sweep models (needed as dependency for eval-pretrain-all)
    cpt_adamw_models,
    cpt_muon_models,
    cpt_muon_pretrain_adamw_ft,
    cpt_adamw_pretrain_muon_ft,
    cpt_models,
    perturbed_adamw_models,
    perturbed_muon_models,
    pretrain_adamw_evals,
    pretrain_muon_evals,
    pretrain_all_wsd_evals,
    cpt_evals,
    cpt_muon_pretrain_adamw_ft_evals,
    cpt_adamw_pretrain_muon_ft_evals,
    perturbed_adamw_evals,
    perturbed_muon_evals,
)


# ---------------------------------------------------------------------------
# Cluster setup
# ---------------------------------------------------------------------------

if Project.config.cluster == "orchard":
    setup_command = "; ".join(
        [
            "source /home/jspringe/.bashrc",
            "source /home/jspringe/.secrets",
            "source /home/jspringe/env/train/bin/activate",
        ]
    )
elif Project.config.cluster == "babel":
    setup_command = "; ".join(
        [
            "source ~/miniconda3/etc/profile.d/conda.sh",
            "conda activate optim-study",
        ]
    )
else:
    raise ValueError(f"Unknown cluster: {Project.config.cluster}")


# ---------------------------------------------------------------------------
# Stage registration
# ---------------------------------------------------------------------------

executor = SlurmExecutor(setup_command=setup_command)

# --- Pretraining (DCLM) ---
executor.stage("pretrain-adamw-wsd",    pretrain_adamw_wsd)
executor.stage("pretrain-adamw-cosine", pretrain_adamw_cosine)
executor.stage("pretrain-muon-wsd",     pretrain_muon_wsd)
executor.stage("pretrain-muon-cosine",  pretrain_muon_cosine)
executor.stage("pretrain-all-wsd",      pretrain_all_wsd)

# --- CPT (finetune the DCLM-pretrained models on datasets) ---
executor.stage("cpt",                  cpt_models)
executor.stage("cpt-adamw",            cpt_adamw_models)
executor.stage("cpt-muon",             cpt_muon_models)
executor.stage("cpt-muon-adamw-ft",    cpt_muon_pretrain_adamw_ft)
executor.stage("cpt-adamw-muon-ft",    cpt_adamw_pretrain_muon_ft)

# --- Gaussian weight perturbation of the DCLM-pretrained models ---
executor.stage("perturb-adamw",        perturbed_adamw_models)
executor.stage("perturb-muon",         perturbed_muon_models)

# --- Evaluation stages (ModelEvaluation: validation loss) ---
executor.stage("eval-pretrain-adamw", pretrain_adamw_evals)
executor.stage("eval-pretrain-muon",  pretrain_muon_evals)
executor.stage("eval-pretrain-all",   pretrain_all_wsd_evals)
executor.stage("eval-cpt",            cpt_evals)
executor.stage("eval-cpt-muon-adamw-ft", cpt_muon_pretrain_adamw_ft_evals)
executor.stage("eval-cpt-adamw-muon-ft", cpt_adamw_pretrain_muon_ft_evals)
executor.stage("eval-perturb-adamw",  perturbed_adamw_evals)
executor.stage("eval-perturb-muon",   perturbed_muon_evals)

# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor.auto_cli()
