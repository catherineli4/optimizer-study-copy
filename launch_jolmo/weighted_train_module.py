"""Power-law per-example loss weighting for JOLMo training.

Analog of associative_memory's power-law data distribution (p_i ∝ i^-α), but
applied to the LLM *loss*: each training example i contributes to the loss with
weight w_i ∝ i^-α, where i is the example's global dataset index (1..N) and N is
the (capped) number of unique training instances.

Weights are normalized so the **mean weight over the full dataset is 1**, which
keeps the overall loss/gradient magnitude comparable to the unweighted run (so
the tuned LRs stay roughly valid).

Implementation: a thin subclass of olmo_core's ``TransformerTrainModule`` that
overrides ``train_batch`` to reduce the per-token loss with per-example weights
(read from ``batch["index"]``, which the data loader attaches to every instance).
The config below reuses the stock ``build()`` and re-blesses the resulting
module, so all the standard parallelism/optimizer wiring is inherited untouched.

This is intended for **token-capped** runs (e.g. the epoch-sweep): N must be the
fixed unique-instance count so the rank i = index is well defined and bounded.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import torch

from olmo_core.config import Config  # noqa: F401  (ensures Config machinery imported)
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module.transformer.config import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule
from olmo_core.utils import move_to_device


def power_law_norm(n_examples: int, alpha: float) -> float:
    """C such that mean_i [ C * (i+1)^-alpha ] == 1 over i = 0..N-1 (rank = i+1)."""
    ranks = torch.arange(1, n_examples + 1, dtype=torch.float64)
    raw_sum = float(ranks.pow(-alpha).sum())
    return n_examples / raw_sum


def _group_ranks(index: torch.Tensor, n_examples: int, n_groups: int) -> torch.Tensor:
    """1-based group rank for each 0-based dataset index: floor(index*G/N)+1.

    Bins the N examples into G equal-sized contiguous index groups (with N % G != 0
    the last group(s) absorb the remainder). Group rank ranges 1..G.
    """
    g = torch.div(index.to(torch.float64) * n_groups, n_examples, rounding_mode="floor")
    return g + 1.0


def power_law_group_norm(n_examples: int, alpha: float, n_groups: int) -> float:
    """C such that mean_i [ C * (group(i)+1)^-alpha ] == 1 over i = 0..N-1, where
    group(i) = floor(i*G/N). Group analogue of :func:`power_law_norm` — every
    example in group g shares the weight C*(g+1)^-alpha, mean-normalized to 1."""
    ranks = _group_ranks(torch.arange(n_examples, dtype=torch.float64), n_examples, n_groups)
    raw_sum = float(ranks.pow(-alpha).sum())
    return n_examples / raw_sum


class PowerLawWeightedTransformerTrainModule(TransformerTrainModule):
    """``TransformerTrainModule`` that weights each example's loss by w_i ∝ (i+1)^-α.

    Set via re-blessing in ``PowerLawWeightedTransformerTrainModuleConfig.build``;
    do not instantiate directly. Requires ``batch["index"]`` (global dataset
    index per instance) and z-loss disabled.
    """

    # Attributes injected by the config's build():
    _lw_alpha: float
    _lw_n_examples: int
    _lw_norm: float
    _lw_n_groups: Optional[int] = None

    def _instance_weights(self, index: torch.Tensor) -> torch.Tensor:
        """w = C * rank^-alpha (index is 0-based).

        rank = index+1 by default (per-example power law), or the 1-based group
        rank floor(index*G/N)+1 when ``_lw_n_groups`` (G) is set (per-group power
        law — every example in a group shares one weight). C (``_lw_norm``)
        mean-normalizes the chosen weighting to 1 over the N examples.
        """
        if self._lw_n_groups:
            ranks = _group_ranks(index, self._lw_n_examples, self._lw_n_groups)
        else:
            ranks = index.to(torch.float64) + 1.0
        return self._lw_norm * ranks.pow(-self._lw_alpha)

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self._set_model_mode("train")

        # The dry-run mock batch (data_loader.get_mock_batch) has only input_ids —
        # no "index" — so fall back to uniform weights there. Real training batches
        # always carry "index" (attached per instance by the data loader); require
        # it for an actual optimization step.
        has_index = "index" in batch
        if not has_index and not dry_run:
            raise RuntimeError(
                "PowerLawWeightedTransformerTrainModule requires batch['index'] for a real "
                "training step; the data loader attaches it automatically — check the data config."
            )
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        labels = batch["labels"]
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        def _weights_for(b: Dict[str, Any], size: int) -> torch.Tensor:
            if "index" in b:
                return self._instance_weights(b["index"])
            return torch.ones(size, dtype=torch.float64)

        # Per-instance weights and non-ignored token counts over the FULL batch.
        # The loss normalizer is Σ_b w_b · n_b (with mean-weight 1 this ≈ the
        # unweighted run's total-token divisor, keeping gradient scale matched).
        weights_full = move_to_device(_weights_for(batch, labels.shape[0]), self.device)  # (B,)
        toks_full = move_to_device(
            (labels != self.label_ignore_index).sum(dim=1).to(torch.float64), self.device
        )  # (B,)
        div_factor = (weights_full * toks_full).sum().clamp_min(1.0)

        ce_batch_loss = move_to_device(torch.tensor(0.0, dtype=torch.float64), self.device)

        seq_len = batch["input_ids"].shape[1]
        if self.rank_microbatch_size < seq_len:
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative "
                f"to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, mb_labels, model_kwargs = self._prepare_batch(micro_batch)

                # Per-token loss (loss_reduction="none" → shape (B_mb, T)).
                # Use index 1 ("loss"): it carries the gradient. Index 2
                # ("ce_loss") is a .detach()ed copy for logging only — backward
                # on it raises "does not require grad". With z-loss disabled,
                # loss == ce_loss numerically.
                out = self.model_forward(
                    input_ids,
                    labels=mb_labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="none",
                    return_logits=False,
                    **model_kwargs,
                )
                loss_per_token = out[1]  # (logits, loss, ce_loss, z_loss); loss per token, grad-carrying

                mb_size = input_ids.shape[0]
                ce_per_inst = loss_per_token.view(mb_size, -1).sum(dim=1)  # (B_mb,) summed over tokens
                mb_w = _weights_for(micro_batch, mb_size).to(ce_per_inst.dtype)
                mb_w = move_to_device(mb_w, ce_per_inst.device)
                loss = (ce_per_inst * mb_w).sum() / div_factor.to(ce_per_inst.dtype)

                ce_batch_loss += get_local_tensor(loss.detach()).to(torch.float64)
                loss.backward()

        del batch

        self.model.post_batch(dry_run=dry_run)
        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        self.record_ce_loss(ce_batch_loss.to(torch.float32), ReduceType.mean)

        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(metric_name, metric_val, reduction, namespace="train")


@dataclass
class PowerLawWeightedTransformerTrainModuleConfig(TransformerTrainModuleConfig):
    """Builds a :class:`PowerLawWeightedTransformerTrainModule`.

    Adds three fields to the stock transformer train-module config:
      * ``loss_weight_alpha`` — the power-law exponent α (weight ∝ rank^-α).
      * ``loss_weight_num_examples`` — N, the number of unique training instances
        (used to normalize so the mean weight is 1). For a token-capped FSL run
        this is dataset_requested_tokens // sequence_length; for a padded
        one-doc-per-instance run it is the document count.
      * ``loss_weight_num_groups`` — G, optional. If set, the N examples are
        binned into G equal-sized index groups and every example in a group is
        weighted by (group_rank)^-α (rank 1..G) instead of (index+1)^-α. None ⇒
        per-example weighting (original behavior).
    """

    loss_weight_alpha: float = 1.0
    loss_weight_num_examples: int = 0
    loss_weight_num_groups: Optional[int] = None

    def build(self, model, device: Optional[torch.device] = None):
        if self.loss_weight_num_examples <= 0:
            raise OLMoConfigurationError(
                "loss_weight_num_examples (N) must be > 0 for power-law loss weighting"
            )
        if self.z_loss_multiplier is not None:
            raise OLMoConfigurationError(
                "power-law loss weighting does not support z-loss (z_loss_multiplier must be None)"
            )
        # Build via a *base* config containing only the parent's fields. The
        # parent build() forwards every dataclass field as a kwarg to
        # TransformerTrainModule.__init__, so our two extra fields must not be on
        # the config it sees (they'd raise "unexpected keyword argument").
        base_cfg = TransformerTrainModuleConfig(
            **{f.name: getattr(self, f.name) for f in fields(TransformerTrainModuleConfig)}
        )
        module = base_cfg.build(model, device=device)
        if not isinstance(module, TransformerTrainModule):
            raise OLMoConfigurationError(
                "power-law loss weighting requires the non-pipeline TransformerTrainModule "
                f"(got {type(module).__name__}); pipeline parallelism is unsupported."
            )
        # Re-bless the stock module to the weighted subclass (no __slots__, so safe)
        # and attach the weighting parameters.
        module.__class__ = PowerLawWeightedTransformerTrainModule
        module._lw_alpha = float(self.loss_weight_alpha)
        module._lw_n_examples = int(self.loss_weight_num_examples)
        module._lw_n_groups = (
            int(self.loss_weight_num_groups) if self.loss_weight_num_groups else None
        )
        if module._lw_n_groups:
            module._lw_norm = power_law_group_norm(
                self.loss_weight_num_examples, self.loss_weight_alpha, module._lw_n_groups)
        else:
            module._lw_norm = power_law_norm(
                self.loss_weight_num_examples, self.loss_weight_alpha)
        return module
