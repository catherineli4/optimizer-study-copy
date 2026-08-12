#!/usr/bin/env python3
"""Per-token distributional divergence eval vs. a reference model (OLMo 2 32B).

For every next-token-prediction position in a validation set, compare the
predicted distribution of MY student model against the reference (ground-truth)
model `allenai/OLMo-2-0325-32B`, and record per-token divergences. The PRIMARY
metric is the forward KL with the reference as ground truth,

    KL(Q || P) = sum_v Q(v) * (log Q(v) - log P(v)),   Q = reference, P = student,

computed explicitly from log-probabilities. Reverse KL and Jensen-Shannon
divergence are also recorded. Outputs per-token rows (parquet/csv), summary
stats, a histogram (log-scaled) + CDF of forward KL, and a JSON run summary.

This is a standalone HF-style eval (both models loaded via
`AutoModelForCausalLM`). It requires that the student shares the OLMo 2
tokenizer and vocab dim (100,352) -- the script asserts this loudly before
computing anything, because cross-vocab KL is meaningless.

Example (smoke test on wikitext-2, a handful of sequences):

    python scripts/eval_divergence.py --student-model allenai/OLMo-2-0425-1B \
        --max-seqs 4 --batch-size 2 --max-length 512 --output-dir results/divergence/smoke

Run the built-in unit tests (no model download):

    python scripts/eval_divergence.py --self-test

Requirements: transformers>=4.48 (OLMo 2 support), torch, numpy, matplotlib,
tqdm; pandas+pyarrow for parquet output (falls back to csv/npy); datasets only
if you rely on the wikitext fallback (i.e. omit --val-data).
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

EXPECTED_VOCAB_DIM = 100_352  # OLMo 2 pads its vocab to a multiple of 128.
PROBE_STRINGS = [
    "The quick brown fox jumps over the lazy dog.",
    "import numpy as np\n\ndef f(x):\n    return x ** 2",
    "Hello, world! 1234567890 — café, naïve, résumé.",
    "",
    "a",
]
EPS = 1e-12  # log-space floor for the JSD mixture; KL itself stays in log space.


# --------------------------------------------------------------------------- #
# Core divergence math (operates purely on log-probabilities)                 #
# --------------------------------------------------------------------------- #
def divergences_from_logprobs(
    log_p: torch.Tensor, log_q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given student log-probs P and reference log-probs Q over the last dim,
    return (kl_forward, kl_reverse, jsd) reduced over the vocab axis.

      kl_forward = KL(Q || P) = sum_v Q(v) (log Q(v) - log P(v))   [PRIMARY]
      kl_reverse = KL(P || Q) = sum_v P(v) (log P(v) - log Q(v))
      jsd        = 0.5 KL(P || M) + 0.5 KL(Q || M),  M = 0.5 (P + Q)

    Inputs are normalized log-probabilities (e.g. from log_softmax). Math is
    done in fp32. Tiny negative values from fp error are clamped to 0.
    """
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    q = log_q.exp()

    kl_forward = (q * (log_q - log_p)).sum(dim=-1)
    kl_reverse = (p * (log_p - log_q)).sum(dim=-1)

    m = 0.5 * (p + q)
    log_m = m.clamp_min(EPS).log()
    jsd = 0.5 * (p * (log_p - log_m)).sum(dim=-1) + 0.5 * (q * (log_q - log_m)).sum(dim=-1)

    # KL/JSD are >= 0 in exact arithmetic; clamp fp noise.
    return (
        kl_forward.clamp_min(0.0),
        kl_reverse.clamp_min(0.0),
        jsd.clamp_min(0.0),
    )


# --------------------------------------------------------------------------- #
# Self-tests / sanity checks                                                  #
# --------------------------------------------------------------------------- #
def run_self_tests() -> None:
    """Prove the divergence math and the KL argument direction are correct."""
    torch.manual_seed(0)

    # 1. KL/JSD of a distribution with itself == 0.
    logits = torch.randn(3, 7, 11)
    lp = F.log_softmax(logits, dim=-1)
    klf, klr, jsd = divergences_from_logprobs(lp, lp)
    assert klf.abs().max() < 1e-5, f"KL(self) not ~0: {klf.abs().max()}"
    assert jsd.abs().max() < 1e-5, f"JSD(self) not ~0: {jsd.abs().max()}"

    # 2. All divergences are non-negative for random distinct distributions.
    lq = F.log_softmax(torch.randn(3, 7, 11), dim=-1)
    klf, klr, jsd = divergences_from_logprobs(lp, lq)
    assert (klf >= 0).all() and (klr >= 0).all() and (jsd >= 0).all()

    # 3. Closed-form check against a hand-computed two-point KL.
    #    Q = [0.5, 0.5], P = [0.25, 0.75].
    log_q = torch.log(torch.tensor([[0.5, 0.5]]))
    log_p = torch.log(torch.tensor([[0.25, 0.75]]))
    klf, klr, _ = divergences_from_logprobs(log_p, log_q)
    # KL(Q||P) = .5*log(.5/.25) + .5*log(.5/.75)
    expect_fwd = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
    expect_rev = 0.25 * np.log(0.25 / 0.5) + 0.75 * np.log(0.75 / 0.5)
    assert abs(klf.item() - expect_fwd) < 1e-6, (klf.item(), expect_fwd)
    assert abs(klr.item() - expect_rev) < 1e-6, (klr.item(), expect_rev)

    # 4. Argument-direction guard: our explicit forward KL must equal
    #    F.kl_div(input=log_p, target=q) (PyTorch computes sum target*(log target - input)).
    q = log_q.exp()
    fkl_div = F.kl_div(log_p, q, reduction="none").sum(dim=-1)
    assert abs(fkl_div.item() - klf.item()) < 1e-6, (
        f"F.kl_div direction mismatch: {fkl_div.item()} vs {klf.item()}"
    )

    print("[self-test] all divergence/direction checks passed.")


# --------------------------------------------------------------------------- #
# Tokenizer / model setup with the correctness gate                           #
# --------------------------------------------------------------------------- #
def assert_tokenizers_match(tok_ref, tok_student) -> None:
    """Fail loudly unless both tokenizers are identical on vocab size and on
    the token ids produced for several probe strings."""
    if tok_ref.vocab_size != tok_student.vocab_size:
        raise SystemExit(
            "FATAL: tokenizer vocab_size mismatch -- KL is meaningless across "
            f"different vocabularies. ref={tok_ref.vocab_size} "
            f"student={tok_student.vocab_size}"
        )
    for s in PROBE_STRINGS:
        ids_ref = tok_ref(s, add_special_tokens=False)["input_ids"]
        ids_student = tok_student(s, add_special_tokens=False)["input_ids"]
        if ids_ref != ids_student:
            raise SystemExit(
                "FATAL: tokenizers encode probe string differently -- the two "
                f"models do NOT share a tokenizer.\n  probe={s!r}\n  "
                f"ref={ids_ref}\n  student={ids_student}"
            )
    print(
        f"[gate] tokenizers match: vocab_size={tok_ref.vocab_size}, "
        f"{len(PROBE_STRINGS)} probe strings encode identically."
    )


def assert_logit_dims_match(model_ref, model_student) -> int:
    """Both models must emit the same number of logits; warn if it isn't the
    expected OLMo 2 vocab dim. Returns the shared logit dim."""
    dim_ref = model_ref.get_output_embeddings().weight.shape[0]
    dim_student = model_student.get_output_embeddings().weight.shape[0]
    if dim_ref != dim_student:
        raise SystemExit(
            "FATAL: models output different numbers of logits -- the "
            f"comparison is invalid across mismatched vocab dims. "
            f"ref={dim_ref} student={dim_student}"
        )
    if dim_ref != EXPECTED_VOCAB_DIM:
        print(
            f"[gate] WARNING: shared logit dim {dim_ref} != expected "
            f"{EXPECTED_VOCAB_DIM}; proceeding since both models agree.",
            file=sys.stderr,
        )
    print(f"[gate] both models emit {dim_ref} logits (padding indices included identically).")
    return dim_ref


def detect_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_str: Optional[str], device: torch.device) -> torch.dtype:
    if dtype_str:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32  # cpu/mps default to fp32 for numerical stability


def load_model(path: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation="sdpa"
    )
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Validation data                                                             #
# --------------------------------------------------------------------------- #
def load_documents(val_data: Optional[str]) -> List[str]:
    """Return a list of raw-text documents. Reads .txt (one doc per line) or
    .jsonl (one record per line; uses the 'text' field, else the raw string).
    If no path is given, falls back to a slice of wikitext-2 via `datasets`."""
    if val_data is None:
        from datasets import load_dataset

        print("[data] no --val-data; falling back to wikitext-2 validation split.")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        return [t for t in ds["text"] if t and t.strip()]

    docs: List[str] = []
    if val_data.endswith(".jsonl"):
        with open(val_data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
                except json.JSONDecodeError:
                    text = line
                if text and text.strip():
                    docs.append(text)
    else:  # treat as .txt: one document per line
        with open(val_data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(line.rstrip("\n"))
    print(f"[data] loaded {len(docs)} documents from {val_data}")
    return docs


def tokenize_documents(docs: List[str], tokenizer, max_length: int) -> List[List[int]]:
    """Tokenize each document independently, truncating to max_length. Drops
    sequences shorter than 2 tokens (no prediction position)."""
    seqs: List[List[int]] = []
    for doc in docs:
        ids = tokenizer(doc, add_special_tokens=False, truncation=True, max_length=max_length)[
            "input_ids"
        ]
        if len(ids) >= 2:
            seqs.append(ids)
    return seqs


def iter_batches(seqs: List[List[int]], pad_id: int, batch_size: int):
    """Yield (input_ids, attention_mask, seq_indices) padded to the batch max
    length. seq_indices give each row's index in the original sequence list."""
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        maxlen = max(len(s) for s in chunk)
        input_ids = np.full((len(chunk), maxlen), pad_id, dtype=np.int64)
        attn = np.zeros((len(chunk), maxlen), dtype=np.int64)
        for i, s in enumerate(chunk):
            input_ids[i, : len(s)] = s
            attn[i, : len(s)] = 1
        yield (
            torch.from_numpy(input_ids),
            torch.from_numpy(attn),
            list(range(start, start + len(chunk))),
        )


# --------------------------------------------------------------------------- #
# Main eval loop                                                              #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate(
    model_student,
    model_ref,
    seqs: List[List[int]],
    device: torch.device,
    pad_id: int,
    batch_size: int,
    max_tokens: Optional[int],
    smoke_print: int = 0,
):
    """Run both models and collect one row per valid prediction position.

    Returns a dict of parallel numpy arrays."""
    cols = {
        "seq_index": [],
        "position": [],
        "next_token_id": [],
        "kl_forward": [],
        "kl_reverse": [],
        "jsd": [],
        "student_nll": [],
        "reference_nll": [],
    }
    printed = 0
    total = 0

    from tqdm import tqdm

    n_batches = (len(seqs) + batch_size - 1) // batch_size
    for input_ids, attn, seq_idx in tqdm(
        iter_batches(seqs, pad_id, batch_size), total=n_batches, desc="divergence"
    ):
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        logits_s = model_student(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        logits_q = model_ref(input_ids=input_ids, attention_mask=attn, use_cache=False).logits

        # Position t predicts token t+1; valid prediction positions are 0..L-2.
        log_p = F.log_softmax(logits_s[:, :-1, :], dim=-1)
        log_q = F.log_softmax(logits_q[:, :-1, :], dim=-1)
        klf, klr, jsd = divergences_from_logprobs(log_p, log_q)  # [B, L-1]

        targets = input_ids[:, 1:]  # token predicted at each position
        # NLL of the true next token under each model (context for later).
        nll_s = -log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        nll_q = -log_q.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Valid where BOTH the predicting position and the predicted position
        # are real (not padding): attn[:, :-1] & attn[:, 1:].
        valid = (attn[:, :-1] * attn[:, 1:]).bool()  # [B, L-1]

        b_idx, t_idx = torch.nonzero(valid, as_tuple=True)
        if b_idx.numel() == 0:
            continue

        klf_v = klf[b_idx, t_idx].cpu().numpy()
        klr_v = klr[b_idx, t_idx].cpu().numpy()
        jsd_v = jsd[b_idx, t_idx].cpu().numpy()
        tok_v = targets[b_idx, t_idx].cpu().numpy()
        nll_s_v = nll_s[b_idx, t_idx].float().cpu().numpy()
        nll_q_v = nll_q[b_idx, t_idx].float().cpu().numpy()
        seq_v = np.array([seq_idx[b] for b in b_idx.cpu().numpy()], dtype=np.int64)
        pos_v = t_idx.cpu().numpy()

        cols["seq_index"].append(seq_v)
        cols["position"].append(pos_v)
        cols["next_token_id"].append(tok_v)
        cols["kl_forward"].append(klf_v)
        cols["kl_reverse"].append(klr_v)
        cols["jsd"].append(jsd_v)
        cols["student_nll"].append(nll_s_v)
        cols["reference_nll"].append(nll_q_v)

        if printed < smoke_print:
            for k in range(min(smoke_print - printed, len(klf_v))):
                print(
                    f"[smoke] seq={seq_v[k]} pos={pos_v[k]} tok={tok_v[k]} "
                    f"KL_fwd={klf_v[k]:.4f} KL_rev={klr_v[k]:.4f} "
                    f"JSD={jsd_v[k]:.4f} nll_student={nll_s_v[k]:.4f} "
                    f"nll_ref={nll_q_v[k]:.4f}"
                )
            printed += min(smoke_print - printed, len(klf_v))

        total += int(b_idx.numel())
        if max_tokens is not None and total >= max_tokens:
            break

    out = {k: (np.concatenate(v) if v else np.zeros((0,), dtype=np.float32)) for k, v in cols.items()}
    if max_tokens is not None and out["kl_forward"].size > max_tokens:
        out = {k: v[:max_tokens] for k, v in out.items()}
    return out


# --------------------------------------------------------------------------- #
# Output: summary + plots                                                     #
# --------------------------------------------------------------------------- #
def summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def save_rows(out: dict, output_dir: str) -> str:
    """Save per-token rows. Prefer parquet; fall back to csv, then npz."""
    base = os.path.join(output_dir, "per_token_divergence")
    try:
        import pandas as pd  # noqa

        df = pd.DataFrame(out)
        path = base + ".parquet"
        df.to_parquet(path, index=False)  # needs pyarrow/fastparquet
        return path
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[save] parquet unavailable ({e}); trying csv/npz.", file=sys.stderr)
    try:
        import pandas as pd

        path = base + ".csv"
        pd.DataFrame(out).to_csv(path, index=False)
        return path
    except Exception as e:  # pragma: no cover
        print(f"[save] csv unavailable ({e}); saving npz.", file=sys.stderr)
        path = base + ".npz"
        np.savez_compressed(path, **out)
        return path


def make_plots(kl: np.ndarray, output_dir: str) -> Tuple[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = 1e-10
    log_kl = np.log10(kl + eps)
    mean_log = np.log10(kl.mean() + eps)
    median_log = np.log10(np.median(kl) + eps)

    # Histogram on a log10 x-axis.
    hist_path = os.path.join(output_dir, "kl_forward_hist.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(log_kl, bins=120, color="tab:blue", alpha=0.8)
    ax.axvline(mean_log, color="tab:red", ls="--", label=f"mean={kl.mean():.3g}")
    ax.axvline(median_log, color="tab:green", ls="--", label=f"median={np.median(kl):.3g}")
    ax.set_xlabel("log10(forward KL(Q||P) + 1e-10)")
    ax.set_ylabel("count")
    ax.set_title("Per-token forward KL (reference OLMo 2 32B vs student)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_path, dpi=130)
    plt.close(fig)

    # CDF on a log x-axis.
    cdf_path = os.path.join(output_dir, "kl_forward_cdf.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.sort(kl)
    ys = np.arange(1, xs.size + 1) / xs.size
    ax.plot(np.maximum(xs, eps), ys, color="tab:blue")
    ax.set_xscale("log")
    ax.axvline(kl.mean(), color="tab:red", ls="--", label=f"mean={kl.mean():.3g}")
    ax.axvline(np.median(kl), color="tab:green", ls="--", label=f"median={np.median(kl):.3g}")
    ax.set_xlabel("forward KL(Q||P) (log scale)")
    ax.set_ylabel("cumulative fraction of tokens")
    ax.set_title("CDF of per-token forward KL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cdf_path, dpi=130)
    plt.close(fig)

    return hist_path, cdf_path


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--self-test", action="store_true", help="Run unit tests and exit (no models).")
    p.add_argument("--student-model", type=str, help="Path or HF id of MY model (required).")
    p.add_argument("--reference-model", type=str, default="allenai/OLMo-2-0325-32B",
                   help="Ground-truth reference model (default: %(default)s).")
    p.add_argument("--val-data", type=str, default=None,
                   help="Path to .txt (one doc/line) or .jsonl (one record/line). "
                        "If omitted, falls back to wikitext-2 validation.")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-seqs", type=int, default=None, help="Cap number of sequences.")
    p.add_argument("--max-tokens", type=int, default=None, help="Cap number of recorded tokens.")
    p.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (auto-detect if unset).")
    p.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp16", "fp32"])
    p.add_argument("--output-dir", type=str, default="results/divergence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-print", type=int, default=0,
                   help="Print this many per-token values to eyeball (e.g. 5).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.self_test:
        run_self_tests()
        return

    if not args.student_model:
        raise SystemExit("--student-model is required (or pass --self-test).")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = detect_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"[setup] device={device} dtype={dtype}")

    from transformers import AutoTokenizer

    # --- correctness gate: tokenizers ---
    tok_ref = AutoTokenizer.from_pretrained(args.reference_model)
    tok_student = AutoTokenizer.from_pretrained(args.student_model)
    print(f"[gate] vocab sizes: reference={tok_ref.vocab_size} student={tok_student.vocab_size}")
    assert_tokenizers_match(tok_ref, tok_student)
    tokenizer = tok_ref
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # --- load models + correctness gate: logit dims ---
    print(f"[setup] loading reference model {args.reference_model}")
    model_ref = load_model(args.reference_model, device, dtype)
    print(f"[setup] loading student model {args.student_model}")
    model_student = load_model(args.student_model, device, dtype)
    logit_dim = assert_logit_dims_match(model_ref, model_student)

    # --- run built-in self-tests so every real run is validated too ---
    run_self_tests()

    # --- data ---
    docs = load_documents(args.val_data)
    seqs = tokenize_documents(docs, tokenizer, args.max_length)
    if args.max_seqs is not None:
        seqs = seqs[: args.max_seqs]
    print(f"[data] {len(seqs)} sequences (>=2 tokens) after tokenization/caps.")
    if not seqs:
        raise SystemExit("No usable sequences (need >=2 tokens each).")

    # --- evaluate ---
    out = evaluate(
        model_student, model_ref, seqs, device, pad_id,
        args.batch_size, args.max_tokens, smoke_print=args.smoke_print,
    )
    kl = out["kl_forward"]
    print(f"[eval] recorded {kl.size} per-token divergence values.")
    if kl.size == 0:
        raise SystemExit("No divergence values recorded.")

    # --- save rows ---
    rows_path = save_rows(out, args.output_dir)
    print(f"[save] per-token rows -> {rows_path}")

    # --- summary stats (primary metric: forward KL) ---
    summary = {
        "student_model": args.student_model,
        "reference_model": args.reference_model,
        "val_data": args.val_data or "wikitext-2-raw-v1:validation",
        "logit_dim": int(logit_dim),
        "n_sequences": len(seqs),
        "n_tokens": int(kl.size),
        "kl_forward": summarize(out["kl_forward"]),
        "kl_reverse": summarize(out["kl_reverse"]),
        "jsd": summarize(out["jsd"]),
        "student_nll_mean": float(out["student_nll"].mean()),
        "reference_nll_mean": float(out["reference_nll"].mean()),
        "rows_path": rows_path,
    }
    print("\n=== forward KL(Q||P) summary (primary metric) ===")
    print(json.dumps(summary["kl_forward"], indent=2))

    # --- plots ---
    hist_path, cdf_path = make_plots(kl, args.output_dir)
    summary["hist_path"] = hist_path
    summary["cdf_path"] = cdf_path
    print(f"[plot] histogram -> {hist_path}")
    print(f"[plot] CDF       -> {cdf_path}")

    # --- JSON run summary ---
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] run summary -> {summary_path}")


if __name__ == "__main__":
    main()
