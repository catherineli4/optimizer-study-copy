#!/usr/bin/env python3
"""Per-token cross-entropy (NLL) on memmap validation data.

At every next-token position, records ``nll = -log P_model(token | context)``.
Output is a compressed .npz with flat float32 ``nll`` arrays concatenating all
datasets, plus one ``nll`` array per dataset label, and a ``.summary.json`` sidecar.

Config YAML shape:

    model: {type: olmo|hf, path: <checkpoint_dir_or_hf_id>}
    chunk_size: <int>
    batch_size: <int>
    device: cuda|cpu|null
    validation_datasets:
      - name: C4_val
        paths: [<memmap.bin>]
        max_instances: null   # optional cap

Example:
    python3 scripts/ce_loss_eval.py config.yaml --output out-ce-loss.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

EXPECTED_VOCAB_DIM = 100_352


def _load_eval_helpers():
    from scripts.divergence_eval import (
        detect_device,
        iter_batches_memmap,
        load_hf,
        load_olmo,
        summarize,
    )

    return detect_device, iter_batches_memmap, load_hf, load_olmo, summarize


def run_self_tests() -> None:
    """Sanity-check NLL against hand-computed values."""
    log_p = F.log_softmax(torch.tensor([[0.0, 0.0]]), dim=-1)
    targets = torch.tensor([0])
    nll = -log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    expect = -np.log(0.5)
    assert abs(nll.item() - expect) < 1e-5, (nll.item(), expect)

    log_p2 = F.log_softmax(torch.tensor([[2.0, 0.0, 1.0]]), dim=-1)
    targets2 = torch.tensor([0])
    nll2 = -log_p2.gather(-1, targets2.unsqueeze(-1)).squeeze(-1)
    p0 = float(torch.softmax(torch.tensor([2.0, 0.0, 1.0]), dim=-1)[0])
    assert abs(nll2.item() + np.log(p0)) < 1e-5
    print("[self-test] CE/NLL checks passed.", file=sys.stderr)


def _logits(model: torch.nn.Module, model_type: str, ids: torch.Tensor) -> torch.Tensor:
    if model_type == "hf":
        return model(input_ids=ids, use_cache=False).logits
    return model(input_ids=ids)


def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """NLL at each position for the true next token. logits [B,L,V], targets [B,L]."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def collect_ce_loss(
    model,
    model_type: str,
    datasets,
    device: torch.device,
    batch_size: int,
    chunk_size: int,
    iter_batches_memmap,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for ds in datasets:
        pieces: List[np.ndarray] = []
        max_inst = ds.get("max_instances")
        total_tokens = 0
        for batch_idx, np_batch in enumerate(
            iter_batches_memmap(ds["paths"], chunk_size, batch_size, max_inst)
        ):
            ids = torch.from_numpy(np_batch.astype(np.int64)).to(device)
            logits = _logits(model, model_type, ids)
            nll = nll_from_logits(logits[:, :-1, :], ids[:, 1:])
            nll_np = nll.reshape(-1).cpu().numpy().astype(np.float32)
            pieces.append(nll_np)
            total_tokens += int(nll_np.size)
            print(
                f"[eval] {ds['name']}  batch {batch_idx:4d}"
                f"  seqs={np_batch.shape[0]}"
                f"  nll_mean={float(nll_np.mean()):.4f}"
                f"  tokens_so_far={total_tokens}",
                file=sys.stderr,
                flush=True,
            )
        cat = np.concatenate(pieces) if pieces else np.zeros((0,), dtype=np.float32)
        out[ds["name"]] = {"nll": cat}
    return out


def assert_vocab_dim(model, model_type: str, device: torch.device) -> int:
    probe = torch.zeros((1, 2), dtype=torch.long, device=device)
    dim = int(_logits(model, model_type, probe).shape[-1])
    if dim != EXPECTED_VOCAB_DIM:
        print(
            f"[gate] WARNING: model emits {dim} logits, expected {EXPECTED_VOCAB_DIM}.",
            file=sys.stderr,
        )
    else:
        print(f"[gate] model emits {dim} logits.", file=sys.stderr)
    return dim


def load_model(model_cfg: dict, device: torch.device, *, load_hf, load_olmo):
    model_type = model_cfg["type"].lower()
    if model_type not in ("hf", "olmo"):
        raise ValueError(f"Unknown model.type {model_type!r}, expected 'hf' or 'olmo'")
    loader = load_hf if model_type == "hf" else load_olmo
    return loader(model_cfg["path"], device), model_type


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", help="Path to YAML config")
    parser.add_argument("--output", type=str, help="Output .npz path")
    parser.add_argument("--self-test", action="store_true", help="Run unit tests and exit.")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return
    if not args.config or not args.output:
        raise SystemExit("config and --output are required (or pass --self-test).")

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    detect_device, iter_batches_memmap, load_hf, load_olmo, summarize = _load_eval_helpers()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chunk_size = int(cfg["chunk_size"])
    batch_size = int(cfg.get("batch_size", 4))
    device = detect_device(cfg.get("device"))
    datasets = cfg["validation_datasets"]

    run_self_tests()

    model, model_type = load_model(
        cfg["model"], device, load_hf=load_hf, load_olmo=load_olmo
    )
    logit_dim = assert_vocab_dim(model, model_type, device)

    per_label = collect_ce_loss(
        model, model_type, datasets, device, batch_size, chunk_size, iter_batches_memmap
    )

    flat_pieces: List[np.ndarray] = []
    label_arrays: Dict[str, np.ndarray] = {}
    for label, metrics in per_label.items():
        flat_pieces.append(metrics["nll"])
        label_arrays[label] = metrics["nll"]
    flat_nll = np.concatenate(flat_pieces) if flat_pieces else np.zeros((0,), dtype=np.float32)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.output, nll=flat_nll, **label_arrays)

    summary = {
        "model_type": model_type,
        "model_path": cfg["model"]["path"],
        "logit_dim": int(logit_dim),
        "n_tokens": int(flat_nll.size),
        "nll": summarize(flat_nll),
        "by_label": {k: int(v.size) for k, v in label_arrays.items()},
    }
    with open(os.path.splitext(args.output)[0] + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
