#!/usr/bin/env python3
"""Per-token metrics binned by adamw↔muon logit angle.

For each next-token position compute ``θ = arccos(cos(ℓ_a, ℓ_m))`` (degrees)
and a suite of metrics already used elsewhere in the LLM pipeline:

  * classification margin  ℓ[y] − max_{v≠y} ℓ[v]   (adamw & muon)
  * top-1 − top-2 confidence margin               (adamw & muon)
  * NLL / CE on the true token                     (adamw & muon)
  * KL(Q‖P) vs OLMo-2 1B                           (adamw & muon)
  * KL between adamw and muon                      (both directions + JSD)
  * token-type frequency (count of that next-token id on this eval walk)

Bins are every 10°: [0,10), [10,20), …  Tokens per bin get histograms at
plot time; this script also dumps ``examples_per_bin`` decoded contexts.

Config YAML (from :class:`LogitAngleBinEvaluation`)::

    adamw_model / muon_model / reference / …
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml


DEFAULT_TOKENIZER = "allenai/OLMo-2-0425-1B-Instruct"
ANGLE_BIN_EDGES = list(range(0, 91, 10))  # 0,10,...,90


def _load_helpers():
    from scripts.divergence_eval import (
        detect_device,
        divergences_from_logprobs,
        iter_batches_memmap,
        load_hf,
        load_olmo,
        load_reference,
    )

    return (
        detect_device,
        divergences_from_logprobs,
        iter_batches_memmap,
        load_hf,
        load_olmo,
        load_reference,
    )


def _logits(model, model_type: str, ids: torch.Tensor) -> torch.Tensor:
    if model_type == "hf":
        return model(input_ids=ids, use_cache=False).logits
    return model(input_ids=ids)


def _load_model(cfg, device, load_hf, load_olmo):
    t, p = cfg["type"], cfg["path"]
    print(f"Loading {t} from {p} on {device}", file=sys.stderr)
    return (load_hf(p, device) if t == "hf" else load_olmo(p, device)), t


def classification_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ℓ[y] − max_{v≠y} ℓ[v]. logits [B,L,V], targets [B,L] → [B,L]."""
    B, L, V = logits.shape
    flat = logits.reshape(-1, V)
    tgt = targets.reshape(-1)
    true = flat.gather(1, tgt.unsqueeze(1)).squeeze(1)
    flat_masked = flat.clone()
    flat_masked.scatter_(1, tgt.unsqueeze(1), float("-inf"))
    runner = flat_masked.max(dim=1).values
    return (true - runner).reshape(B, L)


def top12_margin(logits: torch.Tensor) -> torch.Tensor:
    """top1 − top2 over vocab. logits [B,L,V] → [B,L]."""
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=-1)
    return -log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def cosine_angle_deg(la: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
    """θ in degrees between per-token logit vectors. [B,L,V] → [B,L]."""
    la_f = la.float().reshape(-1, la.shape[-1])
    lm_f = lm.float().reshape(-1, lm.shape[-1])
    na = la_f.norm(dim=-1).clamp_min(1e-12)
    nm = lm_f.norm(dim=-1).clamp_min(1e-12)
    cos = ((la_f * lm_f).sum(dim=-1) / (na * nm)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos)).reshape(la.shape[:2])


def angle_bin_index(angle_deg: np.ndarray, edges: List[int]) -> np.ndarray:
    """Bin index into [edges[i], edges[i+1]); last bin catches ≥ edges[-2]."""
    # digitize with right=False: edges e0..eK → bins 1..K for values in [e_{i-1}, e_i)
    idx = np.digitize(angle_deg, edges[1:-1], right=False)
    return idx.astype(np.int32)


def bin_label(i: int, edges: List[int]) -> str:
    if i >= len(edges) - 1:
        return f"{edges[-1]}+"
    return f"{edges[i]}-{edges[i + 1]}"


@torch.inference_mode()
def collect(
    adamw, adamw_type,
    muon, muon_type,
    reference, ref_device,
    datasets,
    student_device,
    batch_size, chunk_size,
    iter_batches_memmap, divergences_from_logprobs,
    examples_per_bin: int,
    context_tokens: int,
    tokenizer,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[dict]], Dict[str, Any]]:
    pieces: Dict[str, List[np.ndarray]] = {
        k: []
        for k in (
            "angle_deg",
            "token_id",
            "seq_index",
            "token_position",
            "nll_adamw",
            "nll_muon",
            "margin_adamw",
            "margin_muon",
            "top12_adamw",
            "top12_muon",
            "kl_fwd_adamw",
            "kl_fwd_muon",
            "kl_rev_adamw",
            "kl_rev_muon",
            "jsd_adamw",
            "jsd_muon",
            "kl_a_to_m",
            "kl_m_to_a",
            "jsd_am",
        )
    }
    # Reservoirs of candidate examples per angle bin (keep more, subsample later).
    n_bins = len(ANGLE_BIN_EDGES) - 1
    reservoirs: List[List[dict]] = [[] for _ in range(n_bins)]
    keep_cap = max(examples_per_bin * 5, examples_per_bin)

    seq_offset = 0
    n_tokens = 0
    vocab_counts: Optional[np.ndarray] = None

    for ds in datasets:
        max_inst = ds.get("max_instances")
        for batch_idx, np_batch in enumerate(
            iter_batches_memmap(ds["paths"], chunk_size, batch_size, max_inst)
        ):
            ids = torch.from_numpy(np_batch.astype(np.int64)).to(student_device)
            targets = ids[:, 1:]
            la = _logits(adamw, adamw_type, ids)[:, :-1, :].float()
            lm = _logits(muon, muon_type, ids)[:, :-1, :].float()

            ids_r = ids.to(ref_device) if ref_device != student_device else ids
            lq = _logits(reference, "hf", ids_r)[:, :-1, :].float()
            if ref_device != student_device:
                lq = lq.to(student_device)
            log_q = F.log_softmax(lq, dim=-1)
            del lq

            B, L, V = la.shape
            if vocab_counts is None:
                vocab_counts = np.zeros(V, dtype=np.int64)

            angle = cosine_angle_deg(la, lm)
            nll_a = nll_from_logits(la, targets)
            nll_m = nll_from_logits(lm, targets)
            mar_a = classification_margin(la, targets)
            mar_m = classification_margin(lm, targets)
            t12_a = top12_margin(la)
            t12_m = top12_margin(lm)

            log_pa = F.log_softmax(la, dim=-1)
            log_pm = F.log_softmax(lm, dim=-1)
            klf_a, klr_a, jsd_a = divergences_from_logprobs(log_pa, log_q)
            klf_m, klr_m, jsd_m = divergences_from_logprobs(log_pm, log_q)
            kl_am, kl_ma, jsd_am = divergences_from_logprobs(log_pm, log_pa)
            # divergences_from_logprobs(log_p, log_q): kl_forward = KL(Q||P)
            # so (log_pm, log_pa) → KL(Pa || Pm) = kl_a_to_m naming: student P=pm, Q=pa
            # We want KL(adamw || muon) = KL(Pa || Pm): call with log_p=log_pm, log_q=log_pa
            # → kl_forward = KL(Pa||Pm). Good as kl_a_to_m.
            # KL(muon||adamw): log_p=log_pa, log_q=log_pm → kl_m_to_a.

            tok = targets.detach().cpu().numpy().astype(np.int32)
            ang = angle.detach().cpu().numpy().astype(np.float32)
            np.add.at(vocab_counts, tok.ravel(), 1)

            def _np(t):
                return t.detach().cpu().numpy().astype(np.float32)

            seq_idx = (
                np.arange(B, dtype=np.int32)[:, None] + seq_offset
            ).repeat(L, axis=1)
            pos = np.broadcast_to(
                np.arange(L, dtype=np.int32)[None, :], (B, L)
            ).copy()

            pieces["angle_deg"].append(ang.ravel())
            pieces["token_id"].append(tok.ravel())
            pieces["seq_index"].append(seq_idx.ravel())
            pieces["token_position"].append(pos.ravel())
            pieces["nll_adamw"].append(_np(nll_a).ravel())
            pieces["nll_muon"].append(_np(nll_m).ravel())
            pieces["margin_adamw"].append(_np(mar_a).ravel())
            pieces["margin_muon"].append(_np(mar_m).ravel())
            pieces["top12_adamw"].append(_np(t12_a).ravel())
            pieces["top12_muon"].append(_np(t12_m).ravel())
            pieces["kl_fwd_adamw"].append(_np(klf_a).ravel())
            pieces["kl_fwd_muon"].append(_np(klf_m).ravel())
            pieces["kl_rev_adamw"].append(_np(klr_a).ravel())
            pieces["kl_rev_muon"].append(_np(klr_m).ravel())
            pieces["jsd_adamw"].append(_np(jsd_a).ravel())
            pieces["jsd_muon"].append(_np(jsd_m).ravel())
            pieces["kl_a_to_m"].append(_np(kl_am).ravel())
            pieces["kl_m_to_a"].append(_np(kl_ma).ravel())
            pieces["jsd_am"].append(_np(jsd_am).ravel())

            # Example reservoirs (CPU numpy batch for context)
            bin_idx = angle_bin_index(ang.ravel(), ANGLE_BIN_EDGES)
            flat_tok = tok.ravel()
            flat_ang = ang.ravel()
            flat_seq = seq_idx.ravel()
            flat_pos = pos.ravel()
            flat_nll_a = _np(nll_a).ravel()
            flat_nll_m = _np(nll_m).ravel()
            flat_kl_a = _np(klf_a).ravel()
            flat_kl_m = _np(klf_m).ravel()
            flat_mar_a = _np(mar_a).ravel()
            flat_mar_m = _np(mar_m).ravel()

            # Sample up to a few candidates per batch for sparse high-angle bins.
            # Take all positions in rare bins; randomly subsample dense bins.
            rng = np.random.default_rng(seq_offset + batch_idx)
            for bi in range(n_bins):
                locs = np.where(bin_idx == bi)[0]
                if locs.size == 0:
                    continue
                need = keep_cap - len(reservoirs[bi])
                if need <= 0:
                    # reservoir replacement
                    for loc in locs:
                        j = int(rng.integers(0, keep_cap + 1))
                        if j < keep_cap:
                            b_i, t_i = divmod(int(loc), L)
                            ctx_start = max(0, int(flat_pos[loc]) + 1 - context_tokens)
                            # position in ids (input) for context ending at target
                            ctx_ids = np_batch[b_i, ctx_start : int(flat_pos[loc]) + 2]
                            reservoirs[bi][j] = {
                                "angle_deg": float(flat_ang[loc]),
                                "token_id": int(flat_tok[loc]),
                                "token_text": tokenizer.decode(
                                    [int(flat_tok[loc])], skip_special_tokens=False
                                ),
                                "seq_index": int(flat_seq[loc]),
                                "token_position": int(flat_pos[loc]),
                                "context": tokenizer.decode(
                                    ctx_ids.tolist(), skip_special_tokens=False
                                ),
                                "nll_adamw": float(flat_nll_a[loc]),
                                "nll_muon": float(flat_nll_m[loc]),
                                "kl_fwd_adamw": float(flat_kl_a[loc]),
                                "kl_fwd_muon": float(flat_kl_m[loc]),
                                "margin_adamw": float(flat_mar_a[loc]),
                                "margin_muon": float(flat_mar_m[loc]),
                            }
                    continue
                take = locs if locs.size <= need else rng.choice(locs, size=need, replace=False)
                for loc in np.atleast_1d(take):
                    loc = int(loc)
                    b_i, t_i = divmod(loc, L)
                    ctx_start = max(0, int(flat_pos[loc]) + 1 - context_tokens)
                    ctx_ids = np_batch[b_i, ctx_start : int(flat_pos[loc]) + 2]
                    reservoirs[bi].append({
                        "angle_deg": float(flat_ang[loc]),
                        "token_id": int(flat_tok[loc]),
                        "token_text": tokenizer.decode(
                            [int(flat_tok[loc])], skip_special_tokens=False
                        ),
                        "seq_index": int(flat_seq[loc]),
                        "token_position": int(flat_pos[loc]),
                        "context": tokenizer.decode(
                            ctx_ids.tolist(), skip_special_tokens=False
                        ),
                        "nll_adamw": float(flat_nll_a[loc]),
                        "nll_muon": float(flat_nll_m[loc]),
                        "kl_fwd_adamw": float(flat_kl_a[loc]),
                        "kl_fwd_muon": float(flat_kl_m[loc]),
                        "margin_adamw": float(flat_mar_a[loc]),
                        "margin_muon": float(flat_mar_m[loc]),
                    })

            n_tokens += int(ang.size)
            seq_offset += B
            print(
                f"[eval] {ds['name']} batch {batch_idx:4d} "
                f"seqs={B} tokens_so_far={n_tokens} "
                f"mean_angle={float(ang.mean()):.2f}°",
                file=sys.stderr,
                flush=True,
            )
            del la, lm, log_q, log_pa, log_pm

    arrays = {k: np.concatenate(v) for k, v in pieces.items()}
    assert vocab_counts is not None
    arrays["token_freq"] = vocab_counts[arrays["token_id"]].astype(np.int64)
    arrays["angle_bin"] = angle_bin_index(arrays["angle_deg"], ANGLE_BIN_EDGES)

    examples: Dict[str, List[dict]] = {}
    for bi in range(n_bins):
        label = bin_label(bi, ANGLE_BIN_EDGES)
        pool = reservoirs[bi]
        if len(pool) > examples_per_bin:
            rng = np.random.default_rng(bi + 123)
            pick = rng.choice(len(pool), size=examples_per_bin, replace=False)
            pool = [pool[i] for i in pick]
        examples[label] = pool

    meta = {
        "num_tokens": int(n_tokens),
        "angle_bin_edges": ANGLE_BIN_EDGES,
        "bin_counts": {
            bin_label(i, ANGLE_BIN_EDGES): int((arrays["angle_bin"] == i).sum())
            for i in range(n_bins)
        },
        "vocab_size": int(vocab_counts.size),
    }
    return arrays, examples, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("--output-npz", type=str, required=True)
    parser.add_argument("--output-examples", type=str, required=True)
    parser.add_argument("--output-summary", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    (
        detect_device,
        divergences_from_logprobs,
        iter_batches_memmap,
        load_hf,
        load_olmo,
        load_reference,
    ) = _load_helpers()

    device = detect_device(cfg.get("device"))
    adamw, adamw_type = _load_model(cfg["adamw_model"], device, load_hf, load_olmo)
    muon, muon_type = _load_model(cfg["muon_model"], device, load_hf, load_olmo)

    ref_cfg = cfg["reference"]
    reference, ref_device = load_reference(
        ref_cfg["path"],
        device,
        prefer=cfg.get("reference_device", "cuda"),
        use_cache_if_complete=bool(ref_cfg.get("use_cache_if_complete", True)),
    )

    from transformers import AutoTokenizer

    tok_name = cfg.get("tokenizer", DEFAULT_TOKENIZER)
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)

    datasets = cfg["validation_datasets"]
    global_max = cfg.get("max_instances")
    if global_max is not None:
        for ds in datasets:
            ds.setdefault("max_instances", int(global_max))

    arrays, examples, meta = collect(
        adamw, adamw_type,
        muon, muon_type,
        reference, ref_device,
        datasets,
        device,
        int(cfg["batch_size"]),
        int(cfg["chunk_size"]),
        iter_batches_memmap,
        divergences_from_logprobs,
        int(cfg.get("examples_per_bin", 20)),
        int(cfg.get("context_tokens", 64)),
        tokenizer,
    )

    meta.update({
        "chinchilla": cfg.get("chinchilla"),
        "adamw_run": cfg.get("adamw_run"),
        "muon_run": cfg.get("muon_run"),
        "reference_path": ref_cfg["path"],
        "tokenizer": tok_name,
        "chunk_size": int(cfg["chunk_size"]),
        "batch_size": int(cfg["batch_size"]),
        "max_instances": global_max,
        "mean_angle_deg": float(arrays["angle_deg"].mean()),
        "mean_kl_fwd_adamw": float(arrays["kl_fwd_adamw"].mean()),
        "mean_kl_fwd_muon": float(arrays["kl_fwd_muon"].mean()),
        "mean_margin_adamw": float(arrays["margin_adamw"].mean()),
        "mean_margin_muon": float(arrays["margin_muon"].mean()),
    })

    # Per-bin metric summaries
    bin_summaries = {}
    for bi, edge_lo in enumerate(ANGLE_BIN_EDGES[:-1]):
        label = bin_label(bi, ANGLE_BIN_EDGES)
        m = arrays["angle_bin"] == bi
        if not m.any():
            bin_summaries[label] = {"count": 0}
            continue
        bin_summaries[label] = {
            "count": int(m.sum()),
            "mean_angle_deg": float(arrays["angle_deg"][m].mean()),
            "mean_nll_adamw": float(arrays["nll_adamw"][m].mean()),
            "mean_nll_muon": float(arrays["nll_muon"][m].mean()),
            "mean_margin_adamw": float(arrays["margin_adamw"][m].mean()),
            "mean_margin_muon": float(arrays["margin_muon"][m].mean()),
            "mean_kl_fwd_adamw": float(arrays["kl_fwd_adamw"][m].mean()),
            "mean_kl_fwd_muon": float(arrays["kl_fwd_muon"][m].mean()),
            "mean_kl_a_to_m": float(arrays["kl_a_to_m"][m].mean()),
            "mean_token_freq": float(arrays["token_freq"][m].mean()),
            "median_token_freq": float(np.median(arrays["token_freq"][m])),
        }
    meta["bin_summaries"] = bin_summaries

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    with open(args.output_examples, "w", encoding="utf-8") as f:
        json.dump({"chinchilla": cfg.get("chinchilla"), "examples": examples}, f, indent=2)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Human-readable examples dump
    ex_txt = args.output_examples.replace(".json", ".txt")
    with open(ex_txt, "w", encoding="utf-8") as f:
        f.write(f"chinchilla={cfg.get('chinchilla')}\n")
        for label, xs in examples.items():
            f.write(f"\n{'=' * 72}\nANGLE BIN {label}°  (n_examples={len(xs)})\n{'=' * 72}\n")
            for i, ex in enumerate(xs, 1):
                f.write(
                    f"\n--- example {i}/{len(xs)}  θ={ex['angle_deg']:.2f}°  "
                    f"tok={ex['token_id']!r} {ex['token_text']!r}  "
                    f"seq={ex['seq_index']} pos={ex['token_position']}\n"
                    f"nll_a={ex['nll_adamw']:.3f} nll_m={ex['nll_muon']:.3f}  "
                    f"kl_a={ex['kl_fwd_adamw']:.3f} kl_m={ex['kl_fwd_muon']:.3f}  "
                    f"margin_a={ex['margin_adamw']:.3f} margin_m={ex['margin_muon']:.3f}\n"
                    f"CONTEXT:\n{ex['context']}\n"
                )

    print(
        f"Wrote {args.output_npz}  tokens={meta['num_tokens']}  "
        f"mean_angle={meta['mean_angle_deg']:.2f}°  bins={meta['bin_counts']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
