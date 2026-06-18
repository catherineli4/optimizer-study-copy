import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, Tuple, List

from experiments import Project, Artifact, Task
from olmo_core.data import TokenizerConfig

from launch_jolmo.data import Chunk
from launch_jolmo.utils import (
    check_exists_remote,
    local_path,
    local_cache_path,
    remote_path,
)


# ---------------------------------------------------------------------------
# Model architecture registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    hidden_size: int


MODEL_ARCHS: Dict[str, _ModelConfig] = {
    # Approximate parameter scales for small-model sweeps.
    "0.03B":  _ModelConfig(d_model=128 * 1, n_layers=3 * 1, n_heads=2 * 1, hidden_size=512 * 1),
    "0.06B": _ModelConfig(d_model=128 * 2, n_layers=3 * 2, n_heads=2 * 2, hidden_size=512 * 2),
    "0.1B": _ModelConfig(d_model=128 * 3, n_layers=3 * 3, n_heads=2 * 3, hidden_size=512 * 3),
    "0.3B": _ModelConfig(d_model=128 * 6, n_layers=3 * 6, n_heads=2 * 6, hidden_size=512 * 6),
    "0.6B": _ModelConfig(d_model=128 * 8, n_layers=3 * 8, n_heads=2 * 8, hidden_size=512 * 8),
    "1B":   _ModelConfig(d_model=128 * 10, n_layers=3 * 10, n_heads=2 * 10, hidden_size=512 * 10),
    "2.5B": _ModelConfig(d_model=128 * 14, n_layers=3 * 14, n_heads=2 * 14, hidden_size=512 * 14),
}


# ---------------------------------------------------------------------------
# Module-level helpers for YAML config generation
# ---------------------------------------------------------------------------

def _tokenizer_config(tokenizer_id: str) -> TokenizerConfig:
    """Resolve a tokenizer identifier into a fully populated TokenizerConfig."""
    try:
        cfg = TokenizerConfig(identifier=tokenizer_id)
    except TypeError:
        cfg = TokenizerConfig.from_hf(tokenizer_id)
    else:
        if cfg.vocab_size is None or cfg.eos_token_id is None or cfg.pad_token_id is None:
            cfg = TokenizerConfig.from_hf(tokenizer_id)
    return cfg


def _build_tokenizer_yaml(tokenizer_id: str) -> Dict[str, Any]:
    cfg = _tokenizer_config(tokenizer_id)
    d: Dict[str, Any] = {
        "_CLASS_": "olmo_core.data.TokenizerConfig",
        "identifier": tokenizer_id,
        "vocab_size": cfg.vocab_size,
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
    }
    if cfg.bos_token_id is not None:
        d["bos_token_id"] = cfg.bos_token_id
    return d


def _build_model_spec(model_type: str, vocab_size: int) -> Dict[str, Any]:
    arch = MODEL_ARCHS[model_type]
    return {
        "_CLASS_": "olmo_core.nn.transformer.config.TransformerConfig",
        "d_model": arch.d_model,
        "vocab_size": vocab_size,
        "n_layers": arch.n_layers,
        "block": {
            "_CLASS_": "olmo_core.nn.transformer.config.TransformerBlockConfig",
            "name": "reordered_norm",
            "attention": {
                "_CLASS_": "olmo_core.nn.attention.AttentionConfig",
                "n_heads": arch.n_heads,
                "n_kv_heads": None,
                "bias": False,
                "qk_norm": {
                    "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig",
                    "name": "rms",
                    "eps": 1.0e-06,
                    "bias": False,
                },
                "rope": {
                    "_CLASS_": "olmo_core.nn.rope.RoPEConfig",
                    "name": "default",
                    "theta": 500_000,
                },
                "backend": "flash_2",
                "dtype": "float32",
            },
            "feed_forward": {
                "_CLASS_": "olmo_core.nn.feed_forward.FeedForwardConfig",
                "hidden_size": arch.hidden_size,
                "bias": False,
                "dtype": "float32",
            },
            "layer_norm": {
                "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig",
                "name": "rms",
                "eps": 1.0e-06,
                "bias": False,
            },
        },
        "lm_head": {
            "_CLASS_": "olmo_core.nn.lm_head.LMHeadConfig",
            "name": "default",
            "layer_norm": {
                "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig",
                "name": "rms",
                "eps": 1.0e-06,
                "bias": False,
            },
            "bias": False,
            "dtype": "float32",
        },
        "dtype": "float32",
    }


def _optimizer_class_name(optimizer: str) -> str:
    name = optimizer.lower()
    if name == "adamw":
        return "olmo_core.optim.AdamWConfig"
    if name == "adam":
        return "olmo_core.optim.AdamConfig"
    if name == "muon":
        return "olmo_core.optim.MuonConfig"
    raise ValueError(f"Unsupported optimizer '{optimizer}'. Supported: ['adamw', 'adam', 'muon']")


def _build_optimizer_spec(
    optimizer: str,
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    *,
    muon_lr: float = 0.02,
    muon_weight_decay: float = 0.1,
) -> Dict[str, Any]:
    cls = _optimizer_class_name(optimizer)

    embedding_override = {
        "_CLASS_": "olmo_core.optim.OptimGroupOverride",
        "params": ["embeddings.weight"],
        "opts": {"weight_decay": 0.0},
    }

    if cls == "olmo_core.optim.MuonConfig":
        return {
            "_CLASS_": cls,
            "lr": muon_lr,
            "weight_decay": muon_weight_decay,
            "adamw_lr": lr,
            "adamw_betas": list(betas),
            "adamw_weight_decay": weight_decay,
            "group_overrides": [embedding_override],
        }

    spec: Dict[str, Any] = {
        "_CLASS_": cls,
        "lr": lr,
        "betas": list(betas),
        "weight_decay": weight_decay,
        "group_overrides": [embedding_override],
    }
    if cls in {"olmo_core.optim.AdamWConfig", "olmo_core.optim.AdamConfig"}:
        spec["fused"] = True
    return spec


def _build_scheduler_spec(scheduler: str, warmup_steps: int) -> Dict[str, Any]:
    name = scheduler.lower()
    if name == "cosine":
        return {"_CLASS_": "olmo_core.optim.CosWithWarmup", "warmup_steps": warmup_steps}
    if name == "constant":
        return {"_CLASS_": "olmo_core.optim.ConstantScheduler"}
    if name in {"inv_sqrt", "inverse_sqrt", "invsqrt"}:
        return {
            "_CLASS_": "olmo_core.optim.InvSqrtWithWarmup",
            "warmup": warmup_steps,
            "alpha_f": 0.1,
        }
    if name == "wsd":
        return {
            "_CLASS_": "olmo_core.optim.WSD",
            "warmup": warmup_steps,
            "decay_min_lr": 0.0,
        }
    raise ValueError(
        "Unsupported scheduler "
        f"'{scheduler}'. Supported: ['cosine', 'constant', 'inv_sqrt', 'wsd']"
    )


def _gs_parent(uri: str) -> str:
    """Return the parent directory of a URI (everything before the last '/')."""
    return uri.rsplit("/", 1)[0]


def _resolve_chunk_path(chunk: Chunk, dataset_cache_dir: str) -> str:
    """Return the local path where a Chunk's data will be cached."""
    if chunk.is_global:
        # Use the last two path components of the GCS parent directory so that
        # datasets sharing a leaf name (e.g. ".../bigcode_starcoderdata/train/"
        # and ".../allenai_tulu-3-sft-mixture/train/") resolve to distinct local
        # paths and don't silently overwrite each other.
        gs_parent = _gs_parent(chunk.uri)
        components = gs_parent.replace("gs://", "").split("/")
        parent_name = os.path.join(*components[-2:]) if len(components) >= 2 else components[-1]
        return os.path.join(dataset_cache_dir, parent_name, os.path.basename(chunk.uri))
    return os.path.join(dataset_cache_dir, chunk.uri)


def _upload_to_gs_with_retry(
    builder: Task,
    local_path: str,
    gs_path: str,
    directory: bool = False,
    contents: bool = True,
    max_attempts: int = 5,
    base_delay: int = 30,
    guard_path: Optional[str] = None,
) -> None:
    """Upload to GCS with exponential-backoff retry to handle 429/503/stall errors.

    When many job-array tasks finish simultaneously they all upload at once,
    which frequently triggers GCS's per-object mutation rate limit (HTTP 429).
    The built-in gsutil retry only covers transient network errors; 429s require
    explicit backoff at the shell level.

    gsutil rsync can also stall indefinitely when it resumes a stale resumable-upload
    tracker from a prior interrupted attempt (the GCS session is dead but gsutil
    waits forever).  We wrap each attempt in `timeout` so a hung gsutil is killed
    and the next retry starts fresh.

    ``guard_path``: when set, the whole upload is wrapped in a test for that path,
    so a worker that legitimately produced no output (e.g. it skipped a missing
    input checkpoint) doesn't abort the combined ``print | bash`` script under
    ``set -e`` — the upload simply no-ops.
    """
    if directory:
        source = local_path.rstrip("/") + "/"
        dest = gs_path.rstrip("/") + "/"
        gsutil_cmd = f'gsutil -m rsync -r "{source}" "{dest}"'
    else:
        gsutil_cmd = f'gsutil cp "{local_path}" "{gs_path}"'

    # Wrap each gsutil invocation in a 5-minute timeout.  If gsutil stalls
    # (e.g. resuming a dead upload session) it will be killed and the next
    # retry attempt fires after the configured delay.
    timeout_cmd = f'timeout 300 {gsutil_cmd}'

    # Use ||‑chained retries instead of a while loop with shell variables.
    # The SLURM executor echoes every command (with $ unexpanded) before running
    # it; any $var reference in that echo triggers set -u "unbound variable" if
    # the variable hasn't been set yet.  A chain of || subshells contains no
    # unset variable references and is safe under set -euo pipefail.
    delays = [base_delay * (2 ** i) for i in range(max_attempts - 1)]
    attempts = [timeout_cmd] + [f'(sleep {d}; {timeout_cmd})' for d in delays]
    retry_cmd = ' || '.join(attempts)
    if guard_path is not None:
        retry_cmd = (
            f'if [ -e "{guard_path}" ]; then {retry_cmd}; '
            f'else echo "SKIP upload: {guard_path} absent (worker produced no output)"; fi'
        )
    builder.run_command(retry_cmd)


def _download_chunk_dirs(
    builder: Task,
    chunks: List[Chunk],
    dataset_cache_dir: str,
) -> None:
    """Download GCS parent directories for all chunks, skipping dirs already present locally.

    Downloads at directory granularity (directory=True) rather than per-file so that
    the executor's built-in failure cleanup fires: on any gsutil error the whole local
    directory is removed (rm -rf), preventing partial downloads from being mistaken for
    complete ones on subsequent runs.

    Path collision between datasets that share a leaf directory name (e.g.
    ".../bigcode_starcoderdata/train/" and ".../allenai_tulu-3-sft-mixture/train/")
    is avoided by _resolve_chunk_path, which uses the last two GCS path components
    as the local subdirectory (e.g. "bigcode_starcoderdata/train/").
    """
    seen_gs_dirs: set = set()
    for chunk in chunks:
        if chunk.uri.startswith("/"):
            # Local absolute path: symlink into cache dir (create its parent
            # first — _resolve_chunk_path nests under the last two path
            # components, which gsutil would otherwise create on download).
            local = _resolve_chunk_path(chunk, dataset_cache_dir)
            builder.ensure_directory(os.path.dirname(local))
            builder.run_command(f'[ -e "{local}" ] || ln -s "{chunk.uri}" "{local}"')
            continue

        gs_file = chunk.uri if chunk.uri.startswith("gs://") else remote_path(chunk.uri)
        gs_dir = _gs_parent(gs_file)

        if gs_dir in seen_gs_dirs:
            continue
        seen_gs_dirs.add(gs_dir)

        local_file = _resolve_chunk_path(chunk, dataset_cache_dir)
        local_dir = os.path.dirname(local_file)
        builder.download_from_gs(gs_dir, local_dir, directory=True, skip_existing=True)


def _download_chunk_files(
    builder: Task,
    chunks: List[Chunk],
    dataset_cache_dir: str,
) -> None:
    """Download individual chunk FILES rather than their whole parent directory.

    Use this when only a few specific shards are needed (e.g. a token-capped
    run) so we don't pull every shard sitting next to them in the same GCS
    directory.  Local paths match :func:`_resolve_chunk_path` so the dataset
    config resolves them identically to the directory-download path.
    """
    for chunk in chunks:
        if chunk.uri.startswith("/"):
            # Local absolute path: symlink into cache dir (create its parent
            # first — _resolve_chunk_path nests under the last two path
            # components, which gsutil would otherwise create on download).
            local = _resolve_chunk_path(chunk, dataset_cache_dir)
            builder.ensure_directory(os.path.dirname(local))
            builder.run_command(f'[ -e "{local}" ] || ln -s "{chunk.uri}" "{local}"')
            continue

        gs_file = chunk.uri if chunk.uri.startswith("gs://") else remote_path(chunk.uri)
        local_file = _resolve_chunk_path(chunk, dataset_cache_dir)
        builder.ensure_directory(os.path.dirname(local_file))
        builder.download_from_gs(gs_file, local_file, directory=False, skip_existing=True)


# ---------------------------------------------------------------------------
# JolmoModel — unified pre-training / finetuning artifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JolmoModel(Artifact):
    """A model trained (or continued-trained) with JOLMo.

    If ``base_model`` is None, this is a from-scratch pre-training run.
    If ``base_model`` is set, model weights are loaded from that model's
    checkpoint before training begins (continued training / finetuning).
    """

    # --- Identity ---
    model_name: str = ""
    model_type: str = ""

    # --- Base model (None = train from scratch) ---
    base_model: Optional["JolmoModel"] = None
    base_model_checkpoint: str = "final"
    reload_optimizer: bool = False
    reload_trainer_state: bool = False

    # --- Data ---
    train_chunks: Tuple[Chunk, ...] = ()
    validation_chunks: Tuple[Tuple[str, Chunk], ...] = ()
    validation_eval_interval: int = 1000
    # If set, cap the *unique* training set to this many tokens via a
    # SourceMixtureDatasetConfig (instead of consuming all of train_chunks).
    # Combine with n_tokens = epochs * dataset_requested_tokens to train for a
    # fixed number of genuine epochs over the capped data (data-repetition regime).
    dataset_requested_tokens: Optional[int] = None

    # If set, weight each training example's loss by w_i ∝ (i+1)^-loss_weight_alpha
    # (i = global dataset index), normalized to mean weight 1 — the power-law-loss
    # branch (see launch_jolmo/weighted_train_module.py). Requires that N (the
    # unique-instance count) be fixed, via either dataset_requested_tokens (FSL
    # token-cap path) or dataset_num_examples (padded one-doc-per-instance path).
    loss_weight_alpha: Optional[float] = None

    # If True, use a NumpyPaddedFSLDataset: one document per instance, each padded
    # to sequence_length with the padding masked out of the loss. This makes
    # batch["index"] == document index, so the 150-group loss weighting and the
    # per-group QA eval bin by person cleanly and shuffle-invariantly. The unique
    # set is capped by the data file itself (e.g. a first-N-docs shard), NOT by
    # dataset_requested_tokens (which the padded dataset does not support).
    padded_dataset: bool = False
    # For the padded path: the number of documents (== unique training instances
    # == people), N. Fixes N for power-law-loss normalization and group binning.
    dataset_num_examples: Optional[int] = None
    # If set (with loss_weight_alpha), bin the N examples into this many equal
    # index groups and weight every example in group g by (g+1)^-alpha instead of
    # (i+1)^-alpha — the per-group power-law-loss branch (see weighted_train_module).
    loss_weight_num_groups: Optional[int] = None

    # --- Per-group QA evaluation (Heavy-Tail Knowledge Task) ---
    # If set, attach QAGroupEvalCallback: at the end of EACH EPOCH it scores this
    # held-out QA eval .npz (built by scripts/make_qa_group_eval.py) and logs
    # CE/FTA/accuracy/ECE to W&B, overall and split by frequency group, plus
    # per-entity arrays to qa_eval_per_class_dir. Path must be readable on the
    # compute node (e.g. /data/user_data/<user>/bios/qa_group_eval.npz).
    # qa_eval_interval is optional extra in-loop evals every N steps (None = off).
    qa_eval_npz: Optional[str] = None
    qa_eval_interval: Optional[int] = None
    qa_eval_micro_batch_size: int = 64
    qa_eval_n_bins: int = 15
    qa_eval_on_finish: bool = True
    qa_eval_per_class_dir: Optional[str] = None

    # --- Tokenizer ---
    tokenizer: str = "allenai/OLMo-2-0425-1B-Instruct"

    # --- Training hyperparameters ---
    sequence_length: int = 2048
    global_batch_size: int = 524_288
    rank_microbatch_size: int = 524_288 // 8
    eval_rank_microbatch_size: Optional[int] = None

    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 2000
    optimizer: Literal["adamw", "adam", "muon"] = "adamw"
    muon_lr: float = 0.02
    muon_weight_decay: float = 0.1
    scheduler: Literal["cosine", "constant", "inv_sqrt", "wsd"] = "cosine"
    compile_model: bool = True
    max_grad_norm: float = 1.0
    parallelism: Literal["ddp", "fsdp"] = "fsdp"

    n_tokens: Optional[int] = None

    # --- Infrastructure ---
    num_processes: int = 8
    partition: str = "preempt"
    dp_param_dtype: str = "bfloat16"
    dp_reduce_dtype: str = "float32"

    save_overwrite: bool = True
    metrics_collect_interval: int = 10
    cancel_check_interval: int = 5
    save_interval: Optional[int] = 10_000
    ephemeral_save_interval: Optional[int] = 1_000

    experiment_name: Optional[str] = None
    init_seed: int = 12536
    weight_distance_interval: Optional[int] = 100

    # --- Post-processing flags ---
    unshard_checkpoint: bool = True
    convert_to_hf: bool = True
    upload: bool = True

    # Artifacts that must complete before training (for dependency tracking)
    data_dependencies: Tuple = ()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self):
        if self.model_type not in MODEL_ARCHS:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. "
                f"Supported: {sorted(MODEL_ARCHS)}"
            )
        if not self.train_chunks:
            raise ValueError("train_chunks must be a non-empty tuple of Chunk")
        if self.reload_optimizer and self.base_model is None:
            raise ValueError("reload_optimizer requires base_model to be set")
        if self.reload_trainer_state and self.base_model is None:
            raise ValueError("reload_trainer_state requires base_model to be set")
        if (self.loss_weight_alpha is not None
                and self.dataset_requested_tokens is None
                and self.dataset_num_examples is None):
            raise ValueError(
                "loss_weight_alpha requires N (the unique instance count) to be fixed via "
                "either dataset_requested_tokens (FSL token-cap) or dataset_num_examples "
                "(padded one-doc-per-instance) so each example's rank i is well defined."
            )
        if self.padded_dataset and self.dataset_requested_tokens is not None:
            raise ValueError(
                "padded_dataset is incompatible with dataset_requested_tokens "
                "(SourceMixtureDatasetConfig); cap the unique set via the data file "
                "(a first-N-docs shard) and set dataset_num_examples instead."
            )

    # ------------------------------------------------------------------
    # Naming & paths
    # ------------------------------------------------------------------

    @property
    def run_name(self) -> str:
        return self.model_name

    @property
    def relpath(self) -> str:
        return f"JolmoModel/{self.run_name}"

    @property
    def olmo_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-unsharded")

    @property
    def hf_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-hf")

    @property
    def exists(self) -> bool:
        return False
        return check_exists_remote(remote_path(self.hf_model_relpath, "config.json"))

    # ------------------------------------------------------------------
    # Slurm requirements
    # ------------------------------------------------------------------

    def get_requirements(self):
        return {
            "gres": f"gpu:{self.num_processes}",
            "gpus": self.num_processes,
            "cpus": self.num_processes + 2,
            "mem": "256G",
            "partition": self.partition,
        }

    # ------------------------------------------------------------------
    # YAML config generation
    # ------------------------------------------------------------------

    def _vocab_size(self) -> int:
        return _tokenizer_config(self.tokenizer).padded_vocab_size()

    def _build_dataset_spec(self, train_paths: List[str], work_dir: str) -> Dict[str, Any]:
        """Build the dataset config dict the trainer consumes.

        Either references the raw paths directly, or — when
        dataset_requested_tokens is set — caps the unique training set to a
        fixed token budget via a single-source SourceMixtureDatasetConfig.
        Capping the unique data is what makes "N epochs" well defined: the data
        loader wraps the capped dataset, so n_tokens = N * cap yields exactly N
        genuine passes.

        This is the single source of truth for "what data did this model train
        on": TrainExampleLossEvaluation rebuilds the dataset from this exact
        spec (same requested_tokens, global_batch_size, seed defaults, and
        source token counts) so it iterates the identical training instances.
        """
        # Padded one-doc-per-instance dataset: each EOS-delimited document becomes
        # a single instance padded to sequence_length, with the padding masked out
        # of the loss (label_mask). batch["index"] == document index, so per-group
        # loss weighting / eval bin by person. Incompatible with SourceMixture, so
        # the unique set is capped by the data file (a first-N-docs shard).
        if self.padded_dataset:
            return {
                "_CLASS_": "olmo_core.data.NumpyPaddedFSLDatasetConfig",
                "tokenizer": _build_tokenizer_yaml(self.tokenizer),
                "sequence_length": self.sequence_length,
                "work_dir": work_dir,
                "paths": train_paths,
            }

        dataset_spec: Dict[str, Any] = {
            "_CLASS_": "olmo_core.data.NumpyFSLDatasetConfig",
            "tokenizer": _build_tokenizer_yaml(self.tokenizer),
            "sequence_length": self.sequence_length,
            "work_dir": work_dir,
        }
        if self.dataset_requested_tokens is not None:
            dataset_spec["source_mixture_config"] = {
                "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureDatasetConfig",
                "requested_tokens": self.dataset_requested_tokens,
                "global_batch_size": self.global_batch_size,
                "processes": max(1, self.num_processes),
                "render_tables": False,
                "source_list": {
                    "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureList",
                    "sources": [
                        {
                            "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureConfig",
                            "source_name": "train",
                            "target_ratio": 1.0,
                            "paths": train_paths,
                        }
                    ],
                },
            }
        else:
            dataset_spec["paths"] = train_paths
        return dataset_spec

    def _build_yaml_config(
        self,
        save_folder: str,
        train_paths: List[str],
        val_datasets: Dict[str, List[str]],
        work_dir: str,
    ) -> Dict[str, Any]:
        vocab_size = self._vocab_size()

        dataset_spec = self._build_dataset_spec(train_paths, work_dir)

        cfg: Dict[str, Any] = {
            "model": _build_model_spec(self.model_type, vocab_size),
            "dataset": dataset_spec,
            "data_loader": {
                "_CLASS_": "olmo_core.data.NumpyDataLoaderConfig",
                "global_batch_size": self.global_batch_size,
                "seed": 0,
                "num_workers": 4,
            },
            "train_module_type": "normal",
            "train_module": {
                "_CLASS_": "olmo_core.train.train_module.transformer.config.TransformerTrainModuleConfig",
                "rank_microbatch_size": self.rank_microbatch_size,
                **({"eval_rank_microbatch_size": self.eval_rank_microbatch_size} if self.eval_rank_microbatch_size is not None else {}),
                "max_sequence_length": self.sequence_length,
                "optim": _build_optimizer_spec(
                    self.optimizer, self.learning_rate, self.betas, self.weight_decay,
                    muon_lr=self.muon_lr, muon_weight_decay=self.muon_weight_decay,
                ),
                "scheduler": _build_scheduler_spec(self.scheduler, self.warmup_steps),
                "compile_model": self.compile_model,
                "dp_config": {
                    "_CLASS_": "olmo_core.train.train_module.transformer.config.TransformerDataParallelConfig",
                    "name": self.parallelism,
                    "param_dtype": self.dp_param_dtype,
                    "reduce_dtype": self.dp_reduce_dtype,
                },
                "autocast_precision": self.dp_param_dtype,
                "max_grad_norm": self.max_grad_norm,
                # Power-law per-example loss weighting: swap in the weighted train
                # module (still train_module_type="normal" — launch_from_yaml calls
                # cfg.train_module.build(model), which resolves this _CLASS_).
                **({
                    "_CLASS_": "launch_jolmo.weighted_train_module.PowerLawWeightedTransformerTrainModuleConfig",
                    "loss_weight_alpha": self.loss_weight_alpha,
                    # N = unique-instance count: document count for the padded path,
                    # else capped-tokens // seq_len for the FSL token-cap path.
                    "loss_weight_num_examples": (
                        self.dataset_num_examples if self.padded_dataset
                        else self.dataset_requested_tokens // self.sequence_length
                    ),
                    **({"loss_weight_num_groups": self.loss_weight_num_groups}
                       if self.loss_weight_num_groups is not None else {}),
                } if self.loss_weight_alpha is not None else {}),
            },
            "trainer": {
                "_CLASS_": "olmo_core.train.config.TrainerConfig",
                "save_folder": save_folder,
                "save_overwrite": self.save_overwrite,
                "metrics_collect_interval": self.metrics_collect_interval,
                "cancel_check_interval": self.cancel_check_interval,
                "callbacks": {
                    "config_saver": {
                        "_CLASS_": "olmo_core.train.callbacks.config_saver.ConfigSaverCallback",
                    },
                },
            },
            "init_seed": self.init_seed,
        }

        # Weight distance tracking
        if self.weight_distance_interval is not None:
            cfg["trainer"]["callbacks"]["weight_distance"] = {
                "_CLASS_": "olmo_core.train.callbacks.weight_distance.WeightDistanceCallback",
                "log_interval": self.weight_distance_interval,
            }

        # Per-group QA evaluation (Heavy-Tail Knowledge Task). Logs CE/FTA/acc/ECE
        # to W&B overall and per frequency group; dumps per-entity arrays to disk.
        if self.qa_eval_npz is not None:
            per_class_dir = self.qa_eval_per_class_dir
            if per_class_dir is not None:
                per_class_dir = os.path.join(per_class_dir, self.run_name)
            cfg["trainer"]["callbacks"]["qa_group_eval"] = {
                "_CLASS_": "launch_jolmo.qa_eval_callback.QAGroupEvalCallbackConfig",
                "eval_npz": self.qa_eval_npz,
                "eval_interval": self.qa_eval_interval,
                "micro_batch_size": self.qa_eval_micro_batch_size,
                "n_bins": self.qa_eval_n_bins,
                "eval_on_finish": self.qa_eval_on_finish,
                "per_class_dir": per_class_dir,
            }

        # Checkpointer (omit entirely when save_interval is None → no checkpoints)
        if self.save_interval is not None:
            checkpointer = {
                "_CLASS_": "olmo_core.train.callbacks.checkpointer.CheckpointerCallback",
                "save_interval": self.save_interval,
                "save_async": True,
            }
            if self.ephemeral_save_interval is not None:
                checkpointer["ephemeral_save_interval"] = self.ephemeral_save_interval
            cfg["trainer"]["callbacks"]["checkpointer"] = checkpointer

        # W&B
        wandb_project = Project.config.name
        if self.experiment_name:
            wandb_project = f"{wandb_project}.{self.experiment_name}"
        cfg["wandb"] = {
            "enabled": True,
            "project": wandb_project,
            "entity": "catheri4-carnegie-mellon-university",
            "name": self.run_name,
            "resume": "allow",
            "id": f"{self.run_name}-1",
            "config": {
                "model_type": self.model_type,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": list(self.betas),
                "warmup_steps": self.warmup_steps,
                "max_grad_norm": self.max_grad_norm,
                "sequence_length": self.sequence_length,
                "global_batch_size": self.global_batch_size,
                "n_tokens": self.n_tokens,
                "dataset_requested_tokens": self.dataset_requested_tokens,
                "parallelism": self.parallelism,
                "dp_param_dtype": self.dp_param_dtype,
                "dp_reduce_dtype": self.dp_reduce_dtype,
                "compile_model": self.compile_model,
                "num_processes": self.num_processes,
                "tokenizer": self.tokenizer,
                "init_seed": self.init_seed,
                "muon_lr": self.muon_lr,
                "muon_weight_decay": self.muon_weight_decay,
                "base_model": self.base_model.model_name if self.base_model else None,
            },
        }

        # Token budget
        if self.n_tokens is not None:
            cfg["n_tokens"] = self.n_tokens

        # Validation datasets
        if val_datasets:
            cfg["validation_datasets"] = val_datasets
            cfg["validation_eval_interval"] = self.validation_eval_interval

        # Base model loading. The trainer resolves the actual checkpoint via
        # Checkpointer.latest_checkpoint(load_path), which expects the run/save
        # DIRECTORY (it appends "final/" itself, or falls back to the latest
        # step<N>/). So load_path must be the base model's run dir — NOT
        # "<run>/final" (passing the leaf checkpoint dir makes latest_checkpoint
        # scan inside it for step<N>/ subdirs and fail with "No checkpoints
        # found"). For a non-default base_model_checkpoint (a specific step),
        # join it so latest_checkpoint resolves that step dir's own final/.
        if self.base_model is not None:
            base_dir = remote_path(self.base_model.relpath)
            cfg["load_path"] = (
                base_dir if self.base_model_checkpoint == "final"
                else os.path.join(base_dir, self.base_model_checkpoint)
            )
            if self.reload_optimizer:
                cfg["load_optim_state"] = True
            if self.reload_trainer_state:
                cfg["load_trainer_state"] = True
                cfg["load_data_loader_state"] = False
        else:
            cfg["load_path"] = None

        return cfg

    # ------------------------------------------------------------------
    # construct()
    # ------------------------------------------------------------------

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)
        # Prevent OOM from memory fragmentation during torch.compile() dry-run.
        # Reserved-but-unallocated memory can exceed free memory; expandable
        # segments let the allocator grow into new non-contiguous regions instead.
        builder.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        data_dir = local_path()
        cache_dir = local_cache_path()

        save_folder = remote_path(self.relpath)
        output_dir = os.path.join(data_dir, self.relpath)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        work_dir = os.path.join(cache_dir, "training", self.run_name)

        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        # Download data. When the unique training set is token-capped
        # (dataset_requested_tokens), fetch only the specific train shards
        # file-by-file instead of their whole GCS directory — a capped run reads
        # far less than one shard, so pulling every sibling shard wastes
        # bandwidth and disk. Validation chunks always download by directory.
        val_chunks_only = [c for _, c in self.validation_chunks]
        if self.dataset_requested_tokens is not None or self.padded_dataset:
            # Token-capped or padded run: pull only the specific train shard(s)
            # file-by-file (the padded path trains on a single first-N-docs shard).
            _download_chunk_files(builder, list(self.train_chunks), dataset_cache_dir)
            _download_chunk_dirs(builder, val_chunks_only, dataset_cache_dir)
        else:
            _download_chunk_dirs(
                builder, list(self.train_chunks) + val_chunks_only, dataset_cache_dir
            )

        # Resolve training data paths
        train_paths: List[str] = [
            _resolve_chunk_path(c, dataset_cache_dir) for c in self.train_chunks
        ]

        # Resolve validation data paths
        val_datasets: Dict[str, List[str]] = {
            label: [_resolve_chunk_path(c, dataset_cache_dir)]
            for label, c in self.validation_chunks
        }

        # Generate and write YAML config
        yaml_config = self._build_yaml_config(save_folder, train_paths, val_datasets, work_dir)
        config_path = os.path.join(output_dir, "config.yaml")
        builder.create_yaml_file(config_path, yaml_config)

        # Launch training (rdzv-endpoint=localhost:0 picks a free port, avoids shell quoting issues)
        launch_script = os.path.join(
            Project.config.jolmo_path, "src", "scripts", "launch_from_yaml.py",
        )
        # When the QA-group eval callback is enabled, its _CLASS_ lives in
        # launch_jolmo.qa_eval_callback — make the project root importable on the
        # compute node so launch_from_yaml can resolve it.
        env_prefix = ""
        if self.qa_eval_npz is not None:
            env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        builder.run_command(
            f"{env_prefix}torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d "
            f"--nproc_per_node={self.num_processes} {launch_script} {config_path}"
        )

        # Post-training: unshard, convert to HF, upload
        self._postprocess(builder, save_folder, output_dir)

    def _postprocess(self, builder: Task, save_folder: str, output_dir: str,
                     olmo_upload_relpath: Optional[str] = None):
        """Unshard the final checkpoint, convert to HF format, and upload.

        olmo_upload_relpath: GCS relpath for the unsharded upload.  Defaults to
        self.olmo_model_relpath.  Pass CPTModel.olmo_model_relpath when calling
        from CPTModel.construct so the upload goes to CPTModel/ not JolmoModel/.
        """
        jolmo_path = Project.config.jolmo_path
        upload_relpath = olmo_upload_relpath if olmo_upload_relpath is not None else self.olmo_model_relpath

        final_checkpoint_dir = os.path.join(save_folder, "final")
        unsharded_dir = os.path.join(output_dir, "final-unsharded")
        hf_dir = os.path.join(output_dir, "final-hf")

        # Unshard. Resolve final/ — or the latest step*/ when final/ was never
        # written (the last step coincided with a scheduled save, so the
        # checkpointer skipped writing final/). unshard_resolve.py picks the
        # right checkpoint and copies its config.json into unsharded_dir.
        if self.unshard_checkpoint:
            unshard_script = os.path.join(jolmo_path, "src", "scripts", "unshard.py")
            resolve_script = os.path.join(
                Project.config.code_path, "scripts", "unshard_resolve.py",
            )
            builder.run_command(
                f'python3 {resolve_script} --save-folder "{save_folder}" '
                f'--output-dir "{unsharded_dir}" --work-dir "{output_dir}/cache" '
                f'--unshard-script "{unshard_script}"'
            )

        # Convert to HuggingFace format
        if self.convert_to_hf:
            convert_script = os.path.join(
                jolmo_path, "src", "examples", "huggingface", "convert_checkpoint_to_hf.py",
            )
            builder.run_command(
                f'python3 {convert_script} -i "{final_checkpoint_dir}" -o "{hf_dir}" '
                f"--skip-validation --device cpu"
            )

        if self.upload:
            # Upload unsharded OLMo checkpoint
            if self.unshard_checkpoint:
                _upload_to_gs_with_retry(
                    builder,
                    unsharded_dir,
                    remote_path(upload_relpath),
                    directory=True,
                    contents=True,
                )
                # Upload config.json alongside the unsharded weights. It was
                # placed into unsharded_dir by unshard_resolve.py (sourced from
                # whichever checkpoint was unsharded), so read it locally.
                _upload_to_gs_with_retry(
                    builder,
                    os.path.join(unsharded_dir, "config.json"),
                    remote_path(upload_relpath, "config.json"),
                    directory=False,
                )
            # Upload HuggingFace checkpoint
            if self.convert_to_hf:
                _upload_to_gs_with_retry(
                    builder,
                    hf_dir,
                    remote_path(self.hf_model_relpath),
                    directory=True,
                    contents=True,
                )


# ---------------------------------------------------------------------------
# CPT dataset registry
# ---------------------------------------------------------------------------

_CPT_GS = "gs://cmu-gpucloud-jspringe/shared/datasets/OLMo"

# Mirrors LIST_OF_CPT_FILES["dclm"] exactly.
# train_tokens (where present) are raw token counts = artifacts.py "M" values × 1_000_000.
# label_mask paths are omitted — CPTModel uses NumpyFSLDatasetConfig (next-token prediction).
CPT_DATASETS: Dict[str, Dict[str, Any]] = {
    "tulu": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/allenai_tulu-3-sft-mixture/train/input_ids.npy"),
        ),
        "val": {
            "tulu-validation": Chunk(uri=f"{_CPT_GS}/allenai_tulu-3-sft-mixture/val/input_ids-tulu.npy"),
        },
    },
    "starcoder": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/bigcode_starcoderdata/train/input_ids.npy"),
        ),
        "val": {
            "starcoder-validation": Chunk(uri=f"{_CPT_GS}/bigcode_starcoderdata/val/input_ids-starcoder.npy"),
        },
    },
    "musicpile": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/musicpile/train/input_ids.npy"),
        ),
        "val": {
            "musicpile-validation": Chunk(uri=f"{_CPT_GS}/musicpile/val/input_ids-musicpile.npy"),
        },
    },
    "alpaca": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/tatsu-lab_alpaca/train/input_ids.npy"),
        ),
        "val": {
            "alpaca-validation": Chunk(uri=f"{_CPT_GS}/tatsu-lab_alpaca/val/input_ids-alpaca.npy"),
        },
        "train_tokens": 3_600_000,   # dataset is small; cap from artifacts.py
    },
    "gsm8k": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/openai_gsm8k/train/input_ids.npy"),
        ),
        "val": {
            "gsm8k-validation": Chunk(uri=f"{_CPT_GS}/openai_gsm8k/val/input_ids-gsm8k.npy"),
        },
        "train_tokens": 1_200_000,   # dataset is small; cap from artifacts.py
    },
    "siqa": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/allenai_social_i_qa/train/input_ids.npy"),
        ),
        "val": {
            "siqa-validation": Chunk(uri=f"{_CPT_GS}/allenai_social_i_qa/val/input_ids-siqa.npy"),
        },
        "train_tokens": 1_180_000,   # dataset is small; cap from artifacts.py
    },
    "open-platypus": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/garage-bAInd_Open-Platypus/train/input_ids.npy"),
        ),
        "val": {
            "open-platypus-validation": Chunk(uri=f"{_CPT_GS}/garage-bAInd_Open-Platypus/val/input_ids-open-platypus.npy"),
        },
        "train_tokens": 7_000_000,
    },
    "stackmathqa": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/math-ai_StackMathQA/train/input_ids.npy"),
        ),
        "val": {
            "stackmathqa-validation": Chunk(uri=f"{_CPT_GS}/math-ai_StackMathQA/val/input_ids-stackmathqa.npy"),
        },
    },
    "helpsteer": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/nvidia_HelpSteer/train/input_ids.npy"),
        ),
        "val": {
            "helpsteer-validation": Chunk(uri=f"{_CPT_GS}/nvidia_HelpSteer/val/input_ids-helpsteer.npy"),
        },
    },
}

# Pretrain val chunks appended to every CPT run's validation set,
# matching how LIST_OF_CPT_FILES merges pretrain val in catastrophic-forgetting.
_DIVERSITY_GS = "gs://cmu-gpucloud-jspringe/outputs/diversity.pretraining.v2"
_PRETRAIN_VAL_CHUNKS = tuple(
    (n, Chunk(uri=f"{_DIVERSITY_GS}/ValidationDataset/Tokenized/{n}.bin"))
    for n in ("C4_val", "Reddit_val", "Wiki_val", "Books_val")
)


# ---------------------------------------------------------------------------
# CPTModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CPTModel(Artifact):
    """Continual pre-training on a CPT dataset, starting from a JolmoModel checkpoint."""

    pretrained_model: JolmoModel = None  # type: ignore[assignment]
    cpt_dataset: str = "tulu"
    train_tokens: int = 100_000_000        # CPT budget in tokens

    # Override optimizer/LR independently from the pretrained model
    optimizer: Literal["adamw", "adam", "muon"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 20
    muon_lr: float = 0.02
    muon_weight_decay: float = 0.1
    scheduler: Literal["cosine", "constant", "inv_sqrt", "wsd"] = "cosine"
    max_grad_norm: float = 1.0

    # Infrastructure
    num_processes: int = 1
    save_interval: Optional[int] = None
    ephemeral_save_interval: Optional[int] = None
    unshard_checkpoint: bool = True
    convert_to_hf: bool = False
    upload: bool = True
    experiment_name: Optional[str] = "cpt"
    weight_distance_interval: Optional[int] = 5

    @property
    def run_name(self) -> str:
        lr_str = f"{self.learning_rate:.1e}".replace("e-0", "e-")
        tk_str = f"{self.train_tokens // 1_000_000}M"
        base = f"{self.pretrained_model.run_name}-CPT-{self.cpt_dataset}-{tk_str}-{self.optimizer}-lr{lr_str}-exp"
        if self.optimizer == "muon":
            muon_lr_str = f"{self.muon_lr:.1e}".replace("e-0", "e-")
            base = f"{base}-muonlr{muon_lr_str}"
        return base

    @property
    def relpath(self) -> str:
        return f"CPTModel/{self.cpt_dataset}/{self.run_name}"

    @property
    def olmo_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-unsharded")

    @property
    def hf_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-hf")

    @property
    def validation_chunks(self) -> Tuple[Tuple[str, "Chunk"], ...]:
        """CPT-specific val + pretrain val, mirroring LIST_OF_CPT_FILES val merge."""
        cpt_val: Dict[str, "Chunk"] = CPT_DATASETS[self.cpt_dataset]["val"]
        return tuple(cpt_val.items()) + _PRETRAIN_VAL_CHUNKS

    @property
    def exists(self) -> bool:
        # Skip re-training (tier-0) once the finetune checkpoint is on GCS, so
        # downstream eval stages don't re-run the CPT/finetune as a dependency.
        return check_exists_remote(remote_path(self.olmo_model_relpath, "model.pt"))

    def get_requirements(self):
        return {
            "gres": f"gpu:{self.num_processes}",
            "gpus": self.num_processes,
            "cpus": self.num_processes + 2,
            "mem": "128G",
            "partition": "preempt",
            "exclude": "babel-n5-16",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)
        builder.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        cpt_info = CPT_DATASETS[self.cpt_dataset]
        train_chunks: Tuple[Chunk, ...] = cpt_info["train"]
        cpt_val: Dict[str, Chunk] = cpt_info["val"]

        # Honour per-dataset token cap (small datasets like alpaca/gsm8k/siqa)
        # matching the "train_tokens" field in LIST_OF_CPT_FILES["dclm"].
        effective_tokens = cpt_info.get("train_tokens", self.train_tokens)

        # Validation = CPT-specific val + pretrain val (mirrors catastrophic-forgetting
        # merge of LIST_OF_PRETRAIN_FILES val into LIST_OF_CPT_FILES val)
        validation_chunks: Tuple[Tuple[str, Chunk], ...] = (
            tuple(cpt_val.items()) + _PRETRAIN_VAL_CHUNKS
        )

        data_dir = local_path()
        cache_dir = local_cache_path()
        save_folder = remote_path(self.relpath)
        output_dir = os.path.join(data_dir, self.relpath)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        work_dir = os.path.join(cache_dir, "training", self.run_name)

        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        all_chunks = list(train_chunks) + [c for _, c in validation_chunks]
        _download_chunk_dirs(builder, all_chunks, dataset_cache_dir)

        train_paths = [_resolve_chunk_path(c, dataset_cache_dir) for c in train_chunks]
        val_datasets = {
            label: [_resolve_chunk_path(c, dataset_cache_dir)]
            for label, c in validation_chunks
        }

        # Reuse JolmoModel's YAML builder; temporarily substitute fields
        pseudo = JolmoModel(
            model_name=self.run_name,
            model_type=self.pretrained_model.model_type,
            tokenizer=self.pretrained_model.tokenizer,
            sequence_length=self.pretrained_model.sequence_length,
            global_batch_size=self.pretrained_model.global_batch_size,
            rank_microbatch_size=self.pretrained_model.rank_microbatch_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            warmup_steps=self.warmup_steps,
            muon_lr=self.muon_lr,
            muon_weight_decay=self.muon_weight_decay,
            scheduler=self.scheduler,
            max_grad_norm=self.max_grad_norm,
            compile_model=self.pretrained_model.compile_model,
            parallelism=self.pretrained_model.parallelism,
            dp_param_dtype=self.pretrained_model.dp_param_dtype,
            dp_reduce_dtype=self.pretrained_model.dp_reduce_dtype,
            n_tokens=effective_tokens,
            save_interval=self.save_interval,
            ephemeral_save_interval=self.ephemeral_save_interval,
            save_overwrite=True,
            num_processes=self.num_processes,
            experiment_name=self.experiment_name,
            unshard_checkpoint=self.unshard_checkpoint,
            convert_to_hf=self.convert_to_hf,
            upload=self.upload,
            weight_distance_interval=self.weight_distance_interval,
            base_model=self.pretrained_model,
            train_chunks=(train_chunks[0],),  # placeholder, overridden below
        )
        yaml_config = pseudo._build_yaml_config(save_folder, train_paths, val_datasets, work_dir)
        config_path = os.path.join(output_dir, "config.yaml")
        builder.create_yaml_file(config_path, yaml_config)

        launch_script = os.path.join(
            Project.config.jolmo_path, "src", "scripts", "launch_from_yaml.py",
        )
        builder.run_command(
            f"torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d "
            f"--nproc_per_node={self.num_processes} {launch_script} {config_path}"
        )
        pseudo._postprocess(builder, save_folder, output_dir,
                            olmo_upload_relpath=self.olmo_model_relpath)


# ---------------------------------------------------------------------------
# PerturbedModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerturbedModel(Artifact):
    """Perturb a JolmoModel's weights by adding scaled Gaussian noise.

    Noise scale per-parameter: std = gamma * ||W||_F  (matches perturb_weights.py),
    so the relative perturbation is consistent across tensors regardless of norm.
    The naming convention mirrors what perturb_weights.py generates so that
    ``gcs_dir/run_name/final-unsharded/model.pt`` is exactly where the script writes.
    """

    source_model: "JolmoModel" = None  # type: ignore[assignment]
    gamma: float = 0.01
    seed: int = 64

    @property
    def run_name(self) -> str:
        # Must match perturb_weights.get_perturbed_model_name() exactly
        sigma_str = f"{self.gamma:.2e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
        return f"{self.source_model.run_name}_perturbed_{sigma_str}"

    @property
    def relpath(self) -> str:
        return f"PerturbedModel/{self.run_name}"

    @property
    def olmo_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-unsharded")

    @property
    def exists(self) -> bool:
        return False
        return check_exists_remote(remote_path(self.olmo_model_relpath, "model.pt"))

    def get_requirements(self):
        return {"cpus": 4, "gres": "gpu:1", "mem": "64G",  "partition": "preempt"}

    def construct(self, builder: Task):
        perturb_script = os.path.join(
            Project.config.code_path, "new_utils", "perturb_weights.py",
        )
        # perturb_weights.py expects:
        #   input:  {gcs_dir}/{model_name}/final-unsharded/model.pt
        #   output: {output_gcs_dir}/{model_name}_perturbed_{sigma_str}/final-unsharded/model.pt
        src_parent = remote_path(self.source_model.relpath).rsplit("/", 1)[0]  # strip run_name
        builder.run_command(
            f"python3 {perturb_script} "
            f"--gcs_dir {src_parent} "
            f"--model_name {self.source_model.run_name} "
            f"--sigma {self.gamma} "
            f"--seed {self.seed} "
            f"--output_gcs_dir {remote_path('PerturbedModel')}"
        )


# ---------------------------------------------------------------------------
# Helpers for resolving validation chunks across all model types
# ---------------------------------------------------------------------------

def _get_val_chunks_for(model: Any) -> Tuple[Tuple[str, Chunk], ...]:
    """Return (label, Chunk) pairs for whichever artifact type is passed."""
    if isinstance(model, (JolmoModel, CPTModel)):
        return model.validation_chunks
    elif isinstance(model, PerturbedModel):
        return _get_val_chunks_for(model.source_model)
    return ()


# ---------------------------------------------------------------------------
# Helper: resolve the root JolmoModel from any artifact type
# ---------------------------------------------------------------------------

def _get_base_jolmo(model: Any) -> JolmoModel:
    """Drill through PerturbedModel / CPTModel to the root JolmoModel so we can
    read training hyper-params (sequence_length, etc.)."""
    if isinstance(model, JolmoModel):
        return model
    elif isinstance(model, CPTModel):
        return _get_base_jolmo(model.pretrained_model)
    elif isinstance(model, PerturbedModel):
        return _get_base_jolmo(model.source_model)
    raise TypeError(f"Cannot resolve base JolmoModel from {type(model)}")


# ---------------------------------------------------------------------------
# ModelEvaluation — mirrors catastrophic-forgetting ModelEvaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEvaluation(Artifact):
    """Standalone perplexity evaluation using JOLMo/src/scripts/validate.py.

    Uses the same script and the same evaluation parameters (sequence_length,
    eval_rank_microbatch_size) as the training run so results are directly
    comparable to the validation metrics logged during training.

    Mirrors the structure of catastrophic-forgetting/launch/artifacts.py ModelEvaluation.
    """

    model: Any = None   # JolmoModel | CPTModel | PerturbedModel | InterpolatedModel
    device: str = "cuda"
    model_format: str = "olmo"   # "olmo" or "hf"
    # Optional override of the validation datasets. When non-empty, these
    # (label, Chunk) pairs are evaluated instead of the model's own
    # validation_chunks — used to score a model on alternative eval sets
    # (e.g. the typo-corrupted C4 variants). `eval_tag` namespaces the output
    # JSON so these results don't collide with the default eval.
    val_chunks_override: Tuple[Tuple[str, Chunk], ...] = ()
    eval_tag: str = ""
    # Artifacts that must complete before this eval (dependency ordering only).
    # Used to make the typo evals wait on the TypoEvalData build artifact.
    data_dependencies: Tuple = ()

    @property
    def run_name(self) -> str:
        return self.model.run_name

    @property
    def eval_basename(self) -> str:
        """run_name plus the optional eval tag (used for output file names)."""
        return f"{self.run_name}-{self.eval_tag}" if self.eval_tag else self.run_name

    @property
    def relpath(self) -> str:
        return f"ModelEvaluation/{self.eval_basename}-eval.json"

    @property
    def exists(self) -> bool:
        # return False
        # cache_depth=1 lists only gs://.../ModelEvaluation/* in one shot,
        # rather than the default depth-3 which lists the entire bucket.
        return check_exists_remote(remote_path(self.relpath), cache_depth=1)

    def get_requirements(self):
        return {
            "gpus": 1,
            "cpus": 4,
            "mem": "64G",
            "partition": "preempt",
            "exclude": "babel-n5-16",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        # Inherit sequence_length and eval batch size from the source training run
        # so evaluation uses exactly the same parameters as training validation.
        base = _get_base_jolmo(self.model)
        chunk_size = base.sequence_length
        if base.eval_rank_microbatch_size is not None:
            batch_size = max(1, base.eval_rank_microbatch_size // chunk_size)
        else:
            batch_size = max(1, base.rank_microbatch_size // chunk_size)

        data_dir = local_path()
        cache_dir = local_cache_path()

        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        # Download model checkpoint (unsharded OLMo or HF directory)
        checkpoint_gs = remote_path(self.model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, self.model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        # Explicitly download config.json even if the directory already existed
        # (prior runs may have downloaded the directory before config.json was uploaded).
        builder.download_from_gs(
            remote_path(self.model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )

        # Resolve validation data (override takes precedence over the model's own)
        val_chunks = self.val_chunks_override or _get_val_chunks_for(self.model)
        _download_chunk_dirs(builder, [c for _, c in val_chunks], dataset_cache_dir)

        validation_datasets = [
            {"name": label, "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)]}
            for label, chunk in val_chunks
        ]

        # Build validate.py YAML config — same format used by the training framework.
        # chunk_size = sequence_length and batch_size = eval_rank_microbatch_size / seq_len
        # ensure results match the validation metrics logged to W&B during training.
        eval_cfg = {
            "model": {"type": self.model_format, "path": checkpoint_local},
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.eval_basename}-eval.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        # validate.py uses the same memmap evaluation loop as the training framework
        validate_script = os.path.join(
            Project.config.jolmo_path, "src", "scripts", "validate.py",
        )
        builder.run_command(
            f"python3 {validate_script} {config_path} --output {output_json}"
        )

        # Upload result JSON
        _upload_to_gs_with_retry(builder, output_json, remote_path(self.relpath), directory=False)

