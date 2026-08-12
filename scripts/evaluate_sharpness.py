"""Evaluate sharpness metrics (max eigenvalue, trace, directional) for JOLMo models.

Ported from ``catastrophic-forgetting/scripts/evaluate_sharpness.py``. Data /
metrics surface is compatible with ``SharpnessEvaluation`` /
``ForgettingSharpnessEvaluation`` in ``launch_jolmo.training``.

JOLMo adaptations vs. catastrophic-forgetting:
  * loads ``final-unsharded/{model.pt,config.json}`` via olmo_core Transformer
  * forces attention backend ``torch`` (FlashAttn lacks reliable double backward)
  * runs in float32 so HVP / Lanczos / Hutch++ are stable
  * default memmap dtype ``uint32`` (DCLM / diversity-v2 tokenized bins)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

# Repo root on PYTHONPATH (Slurm jobs set this; support local invocation too).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.hessian_sharpness import (  # noqa: E402
    DirectionalSharpnessEvaluator,
    MaxEigenvalueSharpnessEvaluator,
    OptimizerGradientTransform,
    SpectralDensityEvaluator,
    SumEigenvaluesSharpnessEvaluator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

METRIC_ALIASES = {
    "sum_eigenvalue": "sum_eigenvalues",
}
VALID_METRICS = {
    "max_eigenvalue",
    "sum_eigenvalues",
    "directional",
    "spectral_density",
}


def _is_excluded_param(name: str) -> bool:
    lname = name.lower()
    return (
        "embed" in lname
        or "lm_head" in lname
        or "wte" in lname
        or "wpe" in lname
    )


# ---------------------------------------------------------------------------
# Dataset adapter: memmap tokens packaged as evaluator-compatible dataset.
# ---------------------------------------------------------------------------


class MemmapDataset:
    """Tokenised memmap data exposed via ``as_evaluation_data()``."""

    dataset_type = "cpt"

    def __init__(self, data_path, chunk_size, mask_path=None, max_chunks=None, dtype="uint32"):
        data = np.memmap(data_path, dtype=np.dtype(dtype), mode="r")

        mask = None
        if mask_path:
            file_size = os.path.getsize(mask_path)
            data_len = len(data)
            if file_size == data_len * 2:
                mask = np.memmap(mask_path, dtype=np.uint16, mode="r")
            elif file_size == data_len:
                mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
            else:
                log.warning("Mask file size mismatch; ignoring mask.")

        num_chunks = len(data) // chunk_size
        if max_chunks is not None:
            num_chunks = min(num_chunks, max_chunks)

        self._input_ids = []
        self._attention_mask = []
        self._labels = []
        for i in range(num_chunks):
            s, e = i * chunk_size, (i + 1) * chunk_size
            chunk = np.array(data[s:e]).astype(np.int64).tolist()
            self._input_ids.append(chunk)
            self._attention_mask.append([1] * chunk_size)
            if mask is not None:
                mc = np.array(mask[s:e]).astype(bool)
                self._labels.append(
                    [chunk[j] if mc[j] else -100 for j in range(chunk_size)]
                )
            else:
                self._labels.append(list(chunk))

    def as_evaluation_data(self):
        return {
            "input_ids": self._input_ids,
            "attention_mask": self._attention_mask,
            "labels": self._labels,
        }


class _StubTokenizer:
    """Evaluator classes only touch ``pad_token_id`` / ``eos_token_id``."""

    def __init__(self, pad_token_id=0, eos_token_id=0):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


def _force_torch_attention(exp_cfg: dict) -> None:
    """Mutate TransformerConfig dict so attention uses torch SDPA (HVP-safe)."""
    model = exp_cfg.get("model")
    if not isinstance(model, dict):
        return
    block = model.get("block")
    if not isinstance(block, dict):
        return
    attn = block.get("attention")
    if isinstance(attn, dict):
        attn["backend"] = "torch"
        attn.pop("use_flash", None)


def _find_model_state_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    for c in ("model.pt", "model.pth", "model.safetensors", "model.bin"):
        candidate = os.path.join(path, c)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find unsharded model file under {path!r}")


def _find_config_json_near(path: str) -> str:
    for c in (
        os.path.join(path, "config.json"),
        os.path.join(path, "final", "config.json"),
        os.path.join(os.path.dirname(path.rstrip("/")), "final", "config.json"),
        os.path.join(os.path.dirname(path.rstrip("/")), "config.json"),
    ):
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Could not find config.json near {path!r}")


def _load_olmo_sharpness(model_dir_or_file: str, device: torch.device) -> torch.nn.Module:
    """Load JOLMo checkpoint in float32 with torch attention for double backward."""
    from olmo_core.nn.transformer import Transformer, TransformerConfig

    state_path = _find_model_state_path(model_dir_or_file)
    cfg_path = _find_config_json_near(os.path.dirname(state_path))
    with open(cfg_path, "r", encoding="utf-8") as f:
        exp_cfg = json.load(f)
    if "model" not in exp_cfg:
        raise RuntimeError(f"Invalid config at {cfg_path!r}: missing 'model' section.")
    _force_torch_attention(exp_cfg)
    model_cfg = TransformerConfig.from_dict(exp_cfg["model"])
    model: Transformer = model_cfg.build(init_device="cpu")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(True)
    return model.train(False)  # eval mode, but grads still enabled on params


def _load_hf_sharpness(model_path: str, device: torch.device, quantize=None) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if quantize is not None:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=quantize == 8,
            load_in_4bit=quantize == 4,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if quantize is None else "auto",
        attn_implementation="eager",
        quantization_config=quantization_config,
        device_map="auto" if quantize is not None else None,
    )
    if quantize is None:
        model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    return model.train(False)


def _load_model(model_path, device, hf_model, quantize):
    device_t = torch.device(device)
    if hf_model:
        return _load_hf_sharpness(model_path, device_t, quantize=quantize)
    return _load_olmo_sharpness(model_path, device_t)


def _build_direction_dict(model, direction_path, gradient_type):
    """Map a flat direction vector onto a per-parameter dict."""
    direction_vector = torch.load(direction_path, map_location="cpu", weights_only=True)

    all_params = list(model.named_parameters())
    full_numel = sum(p.numel() for _, p in all_params)
    nonembed_numel = sum(p.numel() for n, p in all_params if not _is_excluded_param(n))

    direction_dict = {}
    if gradient_type == "nonembedding":
        if direction_vector.numel() not in (nonembed_numel, full_numel):
            raise ValueError(
                f"direction vector length {direction_vector.numel()} does not match "
                f"nonembedding={nonembed_numel} or full={full_numel}"
            )
        vector_is_full = direction_vector.numel() == full_numel
        src_idx = 0
        for name, param in all_params:
            numel = param.numel()
            if vector_is_full:
                sl = direction_vector[src_idx : src_idx + numel]
                src_idx += numel
                if _is_excluded_param(name):
                    direction_dict[name] = torch.zeros_like(param)
                else:
                    direction_dict[name] = sl.view_as(param).to(
                        device=param.device, dtype=param.dtype
                    )
            else:
                if _is_excluded_param(name):
                    direction_dict[name] = torch.zeros_like(param)
                else:
                    sl = direction_vector[src_idx : src_idx + numel]
                    src_idx += numel
                    direction_dict[name] = sl.view_as(param).to(
                        device=param.device, dtype=param.dtype
                    )
    else:
        if direction_vector.numel() != full_numel:
            raise ValueError(
                f"direction vector length {direction_vector.numel()} != "
                f"full param count {full_numel}"
            )
        src_idx = 0
        for name, param in all_params:
            numel = param.numel()
            sl = direction_vector[src_idx : src_idx + numel]
            src_idx += numel
            direction_dict[name] = sl.view_as(param).to(
                device=param.device, dtype=param.dtype
            )
    return direction_dict


def _load_state_dict_any(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    if path.is_dir():
        state_path = _find_model_state_path(str(path))
        return torch.load(state_path, map_location="cpu", weights_only=True)
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _compute_delta(pretrained_model, ft_checkpoint_path):
    """Return Delta = theta_FT - theta_PT as a flat CPU tensor and ||Delta||."""
    ft_state = _load_state_dict_any(ft_checkpoint_path)
    pt_state = {k: v.cpu() for k, v in pretrained_model.state_dict().items()}
    keys = [k for k in pt_state if k in ft_state]
    delta_vec = parameters_to_vector(
        [(ft_state[k].float() - pt_state[k].float()) for k in keys]
    )
    return delta_vec, delta_vec.norm().item()


def _find_optim_path(model_path: str, optim_path: Optional[str] = None) -> Optional[str]:
    if optim_path:
        if not os.path.isfile(optim_path):
            raise FileNotFoundError(f"optim_path not found: {optim_path}")
        return optim_path
    for c in (
        os.path.join(model_path, "optim.pt") if os.path.isdir(model_path) else None,
        os.path.join(os.path.dirname(model_path.rstrip("/")), "optim.pt"),
    ):
        if c and os.path.isfile(c):
            return c
    return None


def _build_optimizer_transform(model, optim_path: str, device: torch.device):
    log.info(f"Loading optimizer state from {optim_path}")
    optim_flat = torch.load(optim_path, map_location="cpu", weights_only=False)
    if not isinstance(optim_flat, dict):
        raise TypeError(f"Expected flat optim.pt dict, got {type(optim_flat)}")
    named = list(model.named_parameters())
    transform = OptimizerGradientTransform(named, optim_flat, device=device)
    n_muon = sum(1 for s in transform._slices if s["is_muon"])
    n_adamw = len(transform._slices) - n_muon
    log.info(
        f"Optimizer transform ready: {n_muon} Muon (Newton–Schulz) + "
        f"{n_adamw} AdamW (1/√v̂) parameter tensors"
    )
    return transform


def _run_metric(metric, model, tokenizer, dataset, args, direction_path, transform=None):
    if metric == "max_eigenvalue":
        evaluator = MaxEigenvalueSharpnessEvaluator(
            dataset, args.batch_size, args.max_length, transform=transform
        )
    elif metric == "sum_eigenvalues":
        evaluator = SumEigenvaluesSharpnessEvaluator(
            dataset, args.batch_size, args.max_length, args.num_samples
        )
    elif metric == "directional":
        direction_dict = _build_direction_dict(model, direction_path, args.gradient_type)
        evaluator = DirectionalSharpnessEvaluator(
            dataset, args.batch_size, args.max_length, direction_dict
        )
    elif metric == "spectral_density":
        evaluator = SpectralDensityEvaluator(
            dataset,
            args.batch_size,
            args.max_length,
            lanczos_steps=args.lanczos_steps,
            num_probes=args.num_probes,
            probe_seed=args.probe_seed,
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return evaluator.evaluate(model, tokenizer)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sharpness metrics for JOLMo / HF causal LMs"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_model", action="store_true")
    parser.add_argument("--quantize", type=int, default=None, choices=[4, 8])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="uint32")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated: max_eigenvalue, sum_eigenvalue(s), directional, "
            "spectral_density"
        ),
    )
    parser.add_argument(
        "--lanczos_steps",
        type=int,
        default=100,
        help="Lanczos steps per probe for spectral_density (m).",
    )
    parser.add_argument(
        "--num_probes",
        type=int,
        default=10,
        help="Random probe vectors for spectral_density (n_v).",
    )
    parser.add_argument(
        "--probe_seed",
        type=int,
        default=1234,
        help="Seed for spectral_density probe vectors (recorded in output).",
    )
    parser.add_argument(
        "--sharpness_type",
        type=str,
        default=None,
        help="Legacy single-metric alias for --metrics",
    )
    parser.add_argument(
        "--gradient_type",
        type=str,
        default="full",
        choices=["full", "nonembedding"],
    )
    parser.add_argument("--direction_path", type=str, default=None)
    parser.add_argument(
        "--compute_direction_from",
        type=str,
        default=None,
        help="FT checkpoint; Delta = theta_FT - theta_PT (overrides --direction_path)",
    )
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--optim_path",
        type=str,
        default=None,
        help=(
            "Unsharded optim.pt for preconditioned max eigenvalue. "
            "When set (or found next to --model_path), max_eigenvalue also "
            "reports λ_max of T∘H with AdamW / Muon gradient transforms."
        ),
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Optional checkpoint tag recorded in the output JSON (e.g. step1000, final).",
    )

    args = parser.parse_args()

    raw_metrics = args.metrics or args.sharpness_type
    if not raw_metrics:
        parser.error("--metrics (or --sharpness_type) is required")
    metrics = [
        METRIC_ALIASES.get(m.strip(), m.strip())
        for m in raw_metrics.split(",")
        if m.strip()
    ]
    for m in metrics:
        if m not in VALID_METRICS:
            parser.error(f"Unknown metric '{m}'. Valid: {sorted(VALID_METRICS)}")
    if "directional" in metrics and args.direction_path is None and args.compute_direction_from is None:
        parser.error("--direction_path or --compute_direction_from is required for directional")

    log.info(f"Loading model from {args.model_path}")
    model = _load_model(args.model_path, args.device, args.hf_model, args.quantize)
    tokenizer = _StubTokenizer(pad_token_id=args.pad_token_id)

    log.info(f"Loading memmap data from {args.data_path}")
    dataset = MemmapDataset(
        args.data_path,
        args.chunk_size,
        mask_path=args.mask_path,
        max_chunks=args.max_chunks,
        dtype=args.dtype,
    )
    log.info(f"Loaded {len(dataset._input_ids)} chunks of length {args.chunk_size}")

    direction_path = args.direction_path
    delta_norm = None
    tmp_direction = None
    if args.compute_direction_from is not None:
        log.info("Computing direction vector: Delta = theta_FT - theta_PT")
        delta_vec, delta_norm = _compute_delta(model, args.compute_direction_from)
        log.info(f"||Delta|| = {delta_norm:.6f}")
        tmp_direction = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(delta_vec, tmp_direction.name)
        direction_path = tmp_direction.name
        del delta_vec

    transform = None
    optim_path = None
    if "max_eigenvalue" in metrics:
        optim_path = _find_optim_path(args.model_path, args.optim_path)
        if optim_path is None and args.optim_path is not None:
            raise FileNotFoundError(f"optim_path not found: {args.optim_path}")
        if optim_path is not None:
            device_t = torch.device(args.device)
            transform = _build_optimizer_transform(model, optim_path, device_t)
        else:
            log.warning(
                "No optim.pt found; computing raw max_eigenvalue only "
                "(pass --optim_path for preconditioned)."
            )

    results = {}
    if args.checkpoint_name is not None:
        results["checkpoint"] = args.checkpoint_name
        if args.checkpoint_name.startswith("step") and args.checkpoint_name[4:].isdigit():
            results["step"] = int(args.checkpoint_name[4:])
        elif args.checkpoint_name == "final":
            results["step"] = "final"
    if optim_path is not None:
        results["optim_path"] = optim_path
        results["preconditioner"] = "optimizer_gradient_transform"

    for metric in metrics:
        log.info(f"Computing {metric}...")
        results.update(
            _run_metric(
                metric, model, tokenizer, dataset, args, direction_path, transform=transform
            )
        )

    if delta_norm is not None:
        results["delta_norm"] = delta_norm
        if "directional_sharpness" in results:
            results["forgetting_estimate"] = 0.5 * results["directional_sharpness"]
    if tmp_direction is not None:
        os.unlink(tmp_direction.name)

    log.info("Results: " + json.dumps(results, indent=2))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
