"""Perturbed model artifacts for the optimizer-study pretraining experiments."""

from experiments import ArtifactSet

from launch_jolmo.training import PerturbedModel, ModelEvaluation


def build_perturbed_model_evaluations(perturbed_models: ArtifactSet) -> ArtifactSet:
    """One ModelEvaluation per perturbed model (pretrain val loss only)."""
    return perturbed_models.map(lambda model: ModelEvaluation(model=model))


DEFAULT_GAMMAS = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]


def build_perturbed_models(base_models: ArtifactSet, gammas=None) -> ArtifactSet:
    """One PerturbedModel per (base model × gamma) combination.

    gamma controls the per-parameter noise scale:  std = gamma / ||W||_F
    Values default to those used in catastrophic-forgetting/launch/perturb.py;
    pass `gammas` to override (e.g. a much-smaller-σ grid for a finer sweep).
    """
    gs = list(DEFAULT_GAMMAS if gammas is None else gammas)
    return base_models.map_flatten(
        lambda model: ArtifactSet.from_product(
            cls=PerturbedModel,
            params=dict(source_model=model, gamma=gs),
        )
    )
