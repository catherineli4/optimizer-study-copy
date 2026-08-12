#!/usr/bin/env python3
"""Per-token distributional divergence vs. a reference model (OLMo 2 32B).

Memmap-based, two-model sibling of a confidence/perplexity eval. It loads a
student (the JOLMo checkpoint under eval, OLMo-core or HF) and a reference /
ground-truth model (``allenai/OLMo-2-0325-32B``, always HF), and records, at
every next-token prediction position of the validation set, the divergence
between their predicted distributions.

PRIMARY metric: forward KL with the reference as ground truth,
``KL(Q || P) = sum_v Q(v) (log Q(v) - log P(v))``, Q = reference, P = student,
computed explicitly from log-probabilities. Reverse KL and Jensen-Shannon
divergence are also recorded.

Config YAML shape:

    student:   {type: olmo|hf, path: <checkpoint_dir>}
    reference: {path: allenai/OLMo-2-0325-32B}   # loaded via AutoModelForCausalLM
    reference_device: cuda|cpu   # optional; cuda on A100-80GB, auto-fallback to CPU
    chunk_size: <int>          # sequence length
    batch_size: <int>          # sequences per forward pass (keep modest: the
                               # [B, L, V] fp32 softmax tensors are large)
    device: <"cuda"|"cpu"|null>
    validation_datasets:
      - name: <label>
        paths: [<memmap.npy/.bin>, ...]

Output is a compressed .npz with flat float32 arrays ``kl_forward`` (primary),
``kl_reverse``, ``jsd`` concatenating all datasets, plus one ``kl_forward``
array per dataset label (and a ``.summary.json`` sidecar). Plot the histogram
with the colm-moss-latex pipeline (``scripts/plot_divergence.py``).

Correctness gate: both models MUST emit the same number of logits (OLMo 2's
128-padded vocab dim, 100352). KL across mismatched vocabularies is meaningless,
so a mismatch is a hard stop. Self-tests (``--self-test``, also run before every
real eval) prove KL(self)~0, non-negativity, a closed-form two-point KL, and the
F.kl_div argument direction.

Example:
    python3 scripts/divergence_eval.py config.yaml --output out-divergence.npz
"""

import argparse
import fcntl
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM

from olmo_core.nn.transformer import Transformer, TransformerConfig

EXPECTED_VOCAB_DIM = 100_352  # OLMo 2 pads its vocab to a multiple of 128.
EPS = 1e-12  # log-space floor for the JSD mixture.


# --------------------------------------------------------------------------- #
# Core divergence math (operates purely on log-probabilities)                 #
# --------------------------------------------------------------------------- #
def divergences_from_logprobs(log_p: torch.Tensor, log_q: torch.Tensor):
    """Given student log-probs P and reference log-probs Q over the last dim,
    return (kl_forward, kl_reverse, jsd) reduced over the vocab axis.

      kl_forward = KL(Q || P) = sum_v Q(v) (log Q(v) - log P(v))   [PRIMARY]
      kl_reverse = KL(P || Q)
      jsd        = 0.5 KL(P || M) + 0.5 KL(Q || M),  M = 0.5 (P + Q)

    Math is done in fp32; tiny negative fp values are clamped to 0.
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

    return kl_forward.clamp_min(0.0), kl_reverse.clamp_min(0.0), jsd.clamp_min(0.0)


def run_self_tests() -> None:
    """Prove the divergence math and the KL argument direction are correct."""
    torch.manual_seed(0)
    lp = F.log_softmax(torch.randn(3, 7, 11), dim=-1)
    klf, klr, jsd = divergences_from_logprobs(lp, lp)
    assert klf.abs().max() < 1e-5 and jsd.abs().max() < 1e-5, "KL/JSD(self) not ~0"

    lq = F.log_softmax(torch.randn(3, 7, 11), dim=-1)
    klf, klr, jsd = divergences_from_logprobs(lp, lq)
    assert (klf >= 0).all() and (klr >= 0).all() and (jsd >= 0).all()

    log_q = torch.log(torch.tensor([[0.5, 0.5]]))
    log_p = torch.log(torch.tensor([[0.25, 0.75]]))
    klf, klr, _ = divergences_from_logprobs(log_p, log_q)
    expect_fwd = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
    expect_rev = 0.25 * np.log(0.25 / 0.5) + 0.75 * np.log(0.75 / 0.5)
    assert abs(klf.item() - expect_fwd) < 1e-6 and abs(klr.item() - expect_rev) < 1e-6

    # Argument-direction guard: explicit forward KL == F.kl_div(log_p, q).
    fkl_div = F.kl_div(log_p, log_q.exp(), reduction="none").sum(dim=-1)
    assert abs(fkl_div.item() - klf.item()) < 1e-6, "F.kl_div direction mismatch"
    print("[self-test] all divergence/direction checks passed.", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Data + model loading (memmap, OLMo-core / HF)                               #
# --------------------------------------------------------------------------- #
def iter_batches_memmap(
    paths: List[str],
    chunk_size: int,
    batch_size: int,
    max_instances: Optional[int] = None,
):
    """Iterate fixed-size chunks from memmap files.

    ``max_instances`` caps the number of ``chunk_size``-token sequences yielded
    (used for the held-out DCLM shard, matching validate.py).
    """
    batch = []
    emitted = 0
    for p in paths:
        arr = np.memmap(p, mode="r", dtype=np.uint32)
        n_full = (arr.shape[0] // chunk_size) * chunk_size
        for start in range(0, n_full, chunk_size):
            if max_instances is not None and emitted >= max_instances:
                if batch:
                    yield np.stack(batch, axis=0)
                return
            batch.append(np.asarray(arr[start : start + chunk_size]))
            emitted += 1
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0)
                batch = []
    if batch:
        yield np.stack(batch, axis=0)


def find_model_state_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    for c in ("model.pt", "model.pth", "model.safetensors", "model.bin"):
        candidate = os.path.join(path, c)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find unsharded model file under {path!r}")


def find_config_json_near(path: str) -> str:
    for c in (
        os.path.join(path, "config.json"),
        os.path.join(path, "final", "config.json"),
        os.path.join(os.path.dirname(path.rstrip("/")), "final", "config.json"),
        os.path.join(os.path.dirname(path.rstrip("/")), "config.json"),
    ):
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Could not locate 'config.json' near {path!r}")


def load_olmo(model_dir_or_file: str, device: torch.device) -> torch.nn.Module:
    state_path = find_model_state_path(model_dir_or_file)
    cfg_path = find_config_json_near(os.path.dirname(state_path))
    with open(cfg_path, "r", encoding="utf-8") as f:
        exp_cfg = json.load(f)
    if "model" not in exp_cfg:
        raise RuntimeError(f"Invalid config at {cfg_path!r}: missing 'model' section.")
    model_cfg = TransformerConfig.from_dict(exp_cfg["model"])
    model: Transformer = model_cfg.build(init_device="cpu")
    model.load_state_dict(torch.load(state_path, map_location="cpu"), strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()


# Weight shards transformers must resolve (partial hub caches often miss these).
DEFAULT_HF_WEIGHT_SHARDS: dict[str, tuple[str, ...]] = {
    "allenai/OLMo-2-0425-1B": (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ),
}


def _is_hf_hub_id(model_path: str) -> bool:
    return "/" in model_path and not os.path.isdir(model_path)


def _hf_hub_cache_dir() -> str | None:
    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        return cache_dir.rstrip("/")
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return None


def _hub_repo_cache_path(model_id: str) -> str | None:
    if not _is_hf_hub_id(model_id):
        return None
    cache_dir = _hf_hub_cache_dir()
    if not cache_dir:
        return None
    return os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))


def _shard_present(model_id: str, filename: str) -> bool:
    """True when a weight shard is already in the HF hub cache."""
    from huggingface_hub import try_to_load_from_cache

    cache_dir = _hf_hub_cache_dir()
    if not cache_dir:
        return False
    path = try_to_load_from_cache(model_id, filename, cache_dir=cache_dir)
    if not isinstance(path, str):
        return False
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _hf_weight_shards_present(model_id: str, shards: tuple[str, ...]) -> bool:
    if not shards:
        return False
    return all(_shard_present(model_id, name) for name in shards)


def _missing_weight_shards(model_id: str, shards: tuple[str, ...]) -> list[str]:
    return [name for name in shards if not _shard_present(model_id, name)]


def _hf_download_lock_path(model_id: str) -> str:
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE") or "/tmp"
    return os.path.join(hub, f".lock-download-{model_id.replace('/', '--')}")


# ---------------------------------------------------------------------------
# GCS model cache — avoids HF Hub rate limits on subsequent cluster nodes
# ---------------------------------------------------------------------------
_GCS_HF_MODEL_CACHE = "gs://cmu-gpucloud-catheri4/HFModelCache"


def _gcs_model_base(model_id: str) -> str:
    return f"{_GCS_HF_MODEL_CACHE}/{model_id.replace('/', '--')}"


def _flat_model_dir(model_id: str) -> str:
    """Node-local flat directory (plain files, not HF hub symlink format)."""
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(hf_home, "flat-cache", model_id.replace("/", "--"))


def _flat_shards_present(model_id: str, shards: tuple[str, ...]) -> bool:
    d = _flat_model_dir(model_id)
    return all(
        os.path.isfile(os.path.join(d, s)) and os.path.getsize(os.path.join(d, s)) > 0
        for s in shards
    )


def _run_gsutil(*args: str, timeout: int = 300) -> bool:
    import subprocess
    try:
        r = subprocess.run(["gsutil"] + list(args), capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def _gcs_shards_present(model_id: str, shards: tuple[str, ...]) -> bool:
    """Check if all shards are already in the GCS cache (fast stat, no download)."""
    base = _gcs_model_base(model_id)
    return _run_gsutil("stat", *[f"{base}/{s}" for s in shards], timeout=30)


def _populate_flat_dir_from_hub(model_id: str) -> bool:
    """Copy HF hub snapshot files (resolving symlinks) into the flat model dir."""
    import shutil as _shutil
    repo_cache = _hub_repo_cache_path(model_id)
    if not repo_cache:
        return False
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return False
    snap_ids = [d for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snap_ids:
        return False
    snap_dir = os.path.join(snapshots_dir, snap_ids[0])
    flat_dir = _flat_model_dir(model_id)
    os.makedirs(flat_dir, exist_ok=True)
    for name in os.listdir(snap_dir):
        src = os.path.realpath(os.path.join(snap_dir, name))
        if not os.path.isfile(src):
            continue
        dst = os.path.join(flat_dir, name)
        if not os.path.exists(dst):
            _shutil.copy2(src, dst)
    return True


def _upload_flat_to_gcs(model_id: str) -> None:
    """Upload flat model dir to GCS (best-effort; non-fatal)."""
    flat_dir = _flat_model_dir(model_id)
    if not os.path.isdir(flat_dir):
        return
    base = _gcs_model_base(model_id)
    files = [os.path.join(flat_dir, n) for n in os.listdir(flat_dir)
             if os.path.isfile(os.path.join(flat_dir, n))]
    if not files:
        return
    print(f"[hf] uploading {model_id!r} flat cache → GCS {base}", file=sys.stderr)
    ok = _run_gsutil("-m", "cp", *files, base + "/", timeout=900)
    if ok:
        print(f"[hf] GCS upload complete for {model_id!r}", file=sys.stderr)
    else:
        print(f"[hf] GCS upload failed for {model_id!r} (non-fatal)", file=sys.stderr)


def _download_from_gcs_to_flat(model_id: str) -> bool:
    """Download all model files from GCS flat cache to the local flat dir."""
    import subprocess
    base = _gcs_model_base(model_id)
    flat_dir = _flat_model_dir(model_id)
    os.makedirs(flat_dir, exist_ok=True)
    print(f"[hf] downloading {model_id!r} from GCS cache {base}", file=sys.stderr)
    r = subprocess.run(
        ["gsutil", "-m", "cp", f"{base}/*", flat_dir + "/"],
        capture_output=True, timeout=900,
    )
    if r.returncode == 0:
        print(f"[hf] GCS download complete for {model_id!r}", file=sys.stderr)
        return True
    print(
        f"[hf] GCS download failed for {model_id!r}: "
        f"{r.stderr.decode(errors='replace')[:200]}",
        file=sys.stderr,
    )
    return False


def _hf_hub_download_with_retry(
    model_id: str, filename: str, cache_dir: str, *, max_attempts: int = 8
) -> None:
    from huggingface_hub import hf_hub_download

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            hf_hub_download(model_id, filename, cache_dir=cache_dir)
            return
        except Exception as err:
            if not _is_429(err):
                raise
            last_err = err
            wait_s = _retry_wait_s(err, attempt)
            print(
                f"[hf] rate limited downloading {filename!r} "
                f"(attempt {attempt + 1}/{max_attempts}), sleeping {wait_s:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait_s)
    raise RuntimeError(
        f"HF download failed for {model_id!r}/{filename!r} "
        f"after {max_attempts} rate-limit retries"
    ) from last_err


def _is_429(err: BaseException) -> bool:
    """True if err (or its __cause__ chain) is an HF Hub 429."""
    import re as _re
    cause: BaseException | None = err
    while cause is not None:
        if "429" in str(cause):
            return True
        status = getattr(getattr(cause, "response", None), "status_code", None)
        if status == 429:
            return True
        cause = cause.__cause__
    return False


def _retry_wait_s(err: BaseException, attempt: int) -> float:
    import re as _re
    wait_s = min(600.0, 45.0 * (2 ** attempt) + random.uniform(0, 30))
    m = _re.search(r"Retry after (\d+)", str(err))
    if m:
        wait_s = max(wait_s, float(m.group(1)) + 5.0)
    return wait_s


def _snapshot_download_with_retry(model_id: str, kwargs: dict[str, Any], *, max_attempts: int = 8) -> None:
    from huggingface_hub import snapshot_download

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            snapshot_download(model_id, **kwargs)
            return
        except Exception as err:
            # snapshot_download wraps 429 HfHubHTTPError into LocalEntryNotFoundError
            # via its offline-mode fallback; inspect the full cause chain.
            if not _is_429(err):
                raise
            last_err = err
            wait_s = _retry_wait_s(err, attempt)
            print(
                f"[hf] rate limited (attempt {attempt + 1}/{max_attempts}), "
                f"sleeping {wait_s:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait_s)
    raise RuntimeError(
        f"HF download failed for {model_id!r} after {max_attempts} rate-limit retries"
    ) from last_err


def _ensure_hf_weight_shards(model_id: str, shards: tuple[str, ...]) -> None:
    """Download missing weight shards; skip hub entirely when both are cached."""
    missing = _missing_weight_shards(model_id, shards)
    if not missing:
        print(
            f"[hf] weight shards already cached for {model_id!r}, skipping download",
            file=sys.stderr,
        )
        return

    lock_path = _hf_download_lock_path(model_id)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            missing = _missing_weight_shards(model_id, shards)
            if not missing:
                print(
                    f"[hf] weight shards ready (peer on this node finished) "
                    f"for {model_id!r}, skipping download",
                    file=sys.stderr,
                )
                return

            jitter_s = random.uniform(0, 120)
            print(f"[hf] jitter {jitter_s:.0f}s before hub download", file=sys.stderr)
            time.sleep(jitter_s)
            missing = _missing_weight_shards(model_id, shards)
            if not missing:
                print(
                    f"[hf] weight shards ready after jitter for {model_id!r}, "
                    "skipping download",
                    file=sys.stderr,
                )
                return

            cache_dir = _hf_hub_cache_dir()
            kwargs: dict[str, Any] = {}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir

            # Wipe any partial/stale repo cache so snapshot_download doesn't
            # return early believing the local snapshot matches the remote head
            # while actual blob files are missing.
            repo_cache = _hub_repo_cache_path(model_id)
            if repo_cache and os.path.isdir(repo_cache):
                import shutil
                shutil.rmtree(repo_cache, ignore_errors=True)
                print(f"[hf] removed stale/partial cache at {repo_cache}", file=sys.stderr)

            print(
                f"[hf] downloading {model_id!r} ({', '.join(missing)})",
                file=sys.stderr,
            )
            _snapshot_download_with_retry(model_id, kwargs)

            still_missing = _missing_weight_shards(model_id, shards)
            if still_missing:
                raise RuntimeError(
                    f"HF download incomplete for {model_id!r}; "
                    f"missing shard(s): {still_missing}"
                )
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def load_hf(model_path: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", attn_implementation="sdpa"
    ).to(device)
    return model.eval()


# bf16 weights + headroom for student + short-lived activations.
REF_MIN_VRAM_GB_BY_SIZE = {
    "32b": 72,
    "13b": 36,
    "7b": 20,
}
REF_MIN_VRAM_GB_DEFAULT = 72


def _ref_min_vram_gb(model_path: str) -> float:
    lower = model_path.lower()
    for tag, gb in REF_MIN_VRAM_GB_BY_SIZE.items():
        if tag in lower:
            return gb
    return REF_MIN_VRAM_GB_DEFAULT


def _gpu_vram_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)


def load_reference(
    model_path: str,
    student_device: torch.device,
    *,
    prefer: Optional[str] = None,
    use_cache_if_complete: bool = False,
    weight_shards: Optional[tuple[str, ...]] = None,
) -> tuple[torch.nn.Module, torch.device]:
    """Load the HF reference on GPU when VRAM allows (A100-80GB), else CPU."""
    if prefer == "cpu":
        ref_device = torch.device("cpu")
    elif prefer == "cuda" and student_device.type == "cuda":
        ref_device = student_device
    elif student_device.type == "cuda":
        min_vram = _ref_min_vram_gb(model_path)
        vram_gb = _gpu_vram_gb(student_device)
        if vram_gb >= min_vram:
            ref_device = student_device
        else:
            ref_device = torch.device("cpu")
            print(
                f"[load] reference on CPU (GPU has {vram_gb:.0f}GB, "
                f"need >={min_vram:.0f}GB for {model_path!r} bf16).",
                file=sys.stderr,
            )
    else:
        ref_device = student_device

    if ref_device.type == "cuda":
        print(
            f"[load] reference on GPU ({_gpu_vram_gb(ref_device):.0f}GB VRAM).",
            file=sys.stderr,
        )
    elif prefer != "cpu":
        print("[load] reference on CPU.", file=sys.stderr)

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if use_cache_if_complete and _is_hf_hub_id(model_path):
        shards = weight_shards or DEFAULT_HF_WEIGHT_SHARDS.get(model_path, ())
        if shards:
            if _hf_weight_shards_present(model_path, shards):
                print(
                    f"[hf] using cached weight shards for {model_path!r} "
                    "(local_files_only, no hub download)",
                    file=sys.stderr,
                )
            else:
                _ensure_hf_weight_shards(model_path, shards)
            load_kwargs["local_files_only"] = True

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    return model.to(ref_device).eval(), ref_device


def detect_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logits(model: torch.nn.Module, model_type: str, ids: torch.Tensor) -> torch.Tensor:
    if model_type == "hf":
        return model(input_ids=ids, use_cache=False).logits
    return model(input_ids=ids)


def assert_logit_dims_match(
    student, student_type, reference, student_device: torch.device, ref_device: torch.device
) -> int:
    """Run a tiny probe through both models; require identical logit dims."""
    probe_s = torch.zeros((1, 2), dtype=torch.long, device=student_device)
    probe_r = torch.zeros((1, 2), dtype=torch.long, device=ref_device)
    with torch.inference_mode():
        dim_student = int(_logits(student, student_type, probe_s).shape[-1])
        dim_ref = int(_logits(reference, "hf", probe_r).shape[-1])
    if dim_student != dim_ref:
        raise SystemExit(
            "FATAL: student and reference emit different numbers of logits -- "
            f"comparison invalid across mismatched vocab dims. "
            f"student={dim_student} reference={dim_ref}"
        )
    if dim_ref != EXPECTED_VOCAB_DIM:
        print(
            f"[gate] WARNING: shared logit dim {dim_ref} != expected "
            f"{EXPECTED_VOCAB_DIM}; proceeding since both models agree.",
            file=sys.stderr,
        )
    print(f"[gate] both models emit {dim_ref} logits.", file=sys.stderr)
    return dim_ref


# --------------------------------------------------------------------------- #
# Main eval loop                                                              #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def collect_divergences(
    student,
    student_type,
    reference,
    ref_device: torch.device,
    datasets,
    student_device: torch.device,
    batch_size,
    chunk_size,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for ds in datasets:
        klf_pieces, klr_pieces, jsd_pieces = [], [], []
        max_inst = ds.get("max_instances")
        total_tokens = 0
        for batch_idx, np_batch in enumerate(iter_batches_memmap(
            ds["paths"], chunk_size, batch_size, max_inst
        )):
            ids = torch.from_numpy(np_batch.astype(np.int64)).to(student_device)
            logits_s = _logits(student, student_type, ids)
            log_p = F.log_softmax(logits_s[:, :-1, :], dim=-1).float().cpu()
            del logits_s
            if ref_device.type == "cuda":
                torch.cuda.empty_cache()
            logits_q = _logits(reference, "hf", ids.to(ref_device))
            log_q = F.log_softmax(logits_q[:, :-1, :], dim=-1).float().cpu()
            del logits_q
            klf, klr, jsd = divergences_from_logprobs(log_p, log_q)
            klf_np = klf.reshape(-1).cpu().numpy()
            klf_pieces.append(klf_np)
            klr_pieces.append(klr.reshape(-1).cpu().numpy())
            jsd_pieces.append(jsd.reshape(-1).cpu().numpy())
            total_tokens += int(klf_np.size)
            print(
                f"[eval] {ds['name']}  batch {batch_idx:4d}"
                f"  seqs={np_batch.shape[0]}"
                f"  kl_fwd_mean={float(klf_np.mean()):.4f}"
                f"  tokens_so_far={total_tokens}",
                file=sys.stderr,
                flush=True,
            )
        cat = lambda ps: np.concatenate(ps) if ps else np.zeros((0,), dtype=np.float32)
        out[ds["name"]] = {
            "kl_forward": cat(klf_pieces),
            "kl_reverse": cat(klr_pieces),
            "jsd": cat(jsd_pieces),
        }
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    student_cfg = cfg["student"]
    student_type = student_cfg["type"].lower()
    if student_type not in ("hf", "olmo"):
        raise ValueError(f"Unknown student.type {student_type!r}, expected 'hf' or 'olmo'")
    reference_path = cfg["reference"]["path"]
    ref_cfg = cfg["reference"]
    use_cache_if_complete = bool(ref_cfg.get("use_cache_if_complete", False))
    weight_shards_cfg = ref_cfg.get("weight_shards")
    weight_shards = tuple(weight_shards_cfg) if weight_shards_cfg else None
    chunk_size = int(cfg["chunk_size"])
    batch_size = int(cfg.get("batch_size", 4))
    device = detect_device(cfg.get("device"))
    datasets = cfg["validation_datasets"]

    run_self_tests()  # validate the math before touching any model

    student = (load_hf if student_type == "hf" else load_olmo)(student_cfg["path"], device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reference, ref_device = load_reference(
        reference_path,
        device,
        prefer=cfg.get("reference_device"),
        use_cache_if_complete=use_cache_if_complete,
        weight_shards=weight_shards,
    )
    logit_dim = assert_logit_dims_match(
        student, student_type, reference, device, ref_device
    )

    per_label = collect_divergences(
        student, student_type, reference, ref_device, datasets, device, batch_size, chunk_size
    )

    flat = {k: [] for k in ("kl_forward", "kl_reverse", "jsd")}
    label_arrays: Dict[str, np.ndarray] = {}
    for label, metrics in per_label.items():
        for k in flat:
            flat[k].append(metrics[k])
        label_arrays[label] = metrics["kl_forward"]
    flat_arrays = {
        k: (np.concatenate(v) if v else np.zeros((0,), dtype=np.float32))
        for k, v in flat.items()
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.output, **flat_arrays, **label_arrays)

    summary = {
        "student_type": student_type,
        "reference": reference_path,
        "logit_dim": int(logit_dim),
        "n_tokens": int(flat_arrays["kl_forward"].size),
        "kl_forward": summarize(flat_arrays["kl_forward"]),
        "kl_reverse": summarize(flat_arrays["kl_reverse"]),
        "jsd": summarize(flat_arrays["jsd"]),
        "by_label": {k: int(v.size) for k, v in label_arrays.items()},
    }
    with open(os.path.splitext(args.output)[0] + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
