"""Interpolated-model artifacts for the optimizer-study experiments.

For each specified finetuned (CPT) model, build the convex combination of the
finetuned weights with its own pretrained base:

    W = alpha * W_pretrained + (1 - alpha) * W_finetuned

one InterpolatedModel per (finetuned model x alpha), plus a ModelEvaluation
that also scores the held-out DCLM shard (the forgetting axis).
"""

from typing import Iterable, Optional

from experiments import ArtifactSet

from launch_jolmo.training import (
    CPTModel,
    InterpolatedModel,
    JolmoModel,
    ModelEvaluation,
)


# Interior alpha grid for the interpolation (strictly between the two models).
INTERIOR_ALPHAS = [0.2, 0.4, 0.6, 0.8]
# Endpoint at alpha=1 = the PRETRAINED model. It needs its own InterpolatedModel
# eval because the pretrain eval stage does NOT score the CPT dataset (no y-axis
# fine-tuning loss), so we evaluate the pretrained weights on the CPT val set here
# to get the alpha=1 Pareto endpoint. (alpha=0 = the finetuned model is omitted —
# its CPT eval already has both axes and the plotter reuses it.)
PRETRAINED_ENDPOINT_ALPHA = 1.0
DEFAULT_ALPHAS = INTERIOR_ALPHAS + [PRETRAINED_ENDPOINT_ALPHA]


def build_interpolated_models(
    finetuned_models: ArtifactSet,
    alphas: Optional[Iterable[float]] = None,
) -> ArtifactSet:
    """One InterpolatedModel per (finetuned model x alpha).

    The pretrained base is read off each finetuned model: a CPTModel exposes it
    as ``.pretrained_model``; a JolmoModel that was itself continued-trained
    exposes it as ``.base_model``.

    The default grid includes ``alpha=1.0`` (the pretrained endpoint) so the
    pretrained model is scored on the CPT dataset for the alpha=1 Pareto point.
    ``alpha=0.0`` (the finetuned endpoint) is intentionally NOT built — the
    finetuned model's own CPT eval already provides that point, which the
    plotter reuses to avoid a redundant re-evaluation.
    """
    al = list(DEFAULT_ALPHAS if alphas is None else alphas)

    def _base_of(ft):
        if isinstance(ft, CPTModel):
            return ft.pretrained_model
        if isinstance(ft, JolmoModel):
            return ft.base_model
        raise TypeError(f"Cannot resolve pretrained base from {type(ft)}")

    def _make(ft) -> ArtifactSet:
        base = _base_of(ft)
        if base is None:
            return ArtifactSet([])
        return ArtifactSet([
            InterpolatedModel(pretrained_model=base, finetuned_model=ft, alpha=a)
            for a in al
        ])

    return finetuned_models.map_flatten(_make)


def build_interpolated_model_evaluations(
    interpolated_models: ArtifactSet,
    extra_val_chunks=(),
    extra_val_max_instances=None,
) -> ArtifactSet:
    """One ModelEvaluation per interpolated model.

    `extra_val_chunks` / `extra_val_max_instances` mirror the cpt/perturb evals:
    pass the held-out DCLM shard so each interpolated model is also scored on the
    DCLM split (label ``DCLM_heldout``) — the forgetting axis of the Pareto plot.
    """
    return interpolated_models.map(
        lambda model: ModelEvaluation(
            model=model,
            extra_val_chunks=tuple(extra_val_chunks),
            extra_val_max_instances=extra_val_max_instances,
        )
    )
