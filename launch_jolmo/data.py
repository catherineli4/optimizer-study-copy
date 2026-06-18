import os
from dataclasses import dataclass
from typing import Optional, Tuple

from experiments import Artifact, Task

from launch_jolmo.utils import check_exists_remote, local_path, remote_path


@dataclass(frozen=True)
class Chunk:
    """A reference to a tokenized pre-training data chunk.

    The URI determines how the chunk is resolved:
    - "gs://..." or "/..." → global path (absolute GCS or filesystem path)
    - anything else → relative path (resolved against the project's remote_path)
    """

    uri: str

    @property
    def is_global(self) -> bool:
        return self.uri.startswith("gs://") or self.uri.startswith("/")

    @property
    def relpath(self) -> Optional[str]:
        """Return the relative path for relative URIs, or None for global paths."""
        if self.is_global:
            return None
        return self.uri

    @property
    def exists(self) -> bool:
        """Check whether the chunk data exists at its resolved location."""
        if self.uri.startswith("gs://"):
            return check_exists_remote(self.uri)
        elif self.uri.startswith("/"):
            return os.path.exists(self.uri)
        else:
            return check_exists_remote(remote_path(self.uri))


@dataclass(frozen=True)
class TokenizedMessagesDataset(Artifact):
    """Downloads and tokenizes a HuggingFace messages-format dataset.

    Applies the tokenizer's chat template to each conversation and produces
    raw binary .bin files compatible with NumpyFSLDatasetConfig.
    """

    hf_dataset: str
    messages_column: str = "messages"
    tokenizer: str = "allenai/OLMo-2-0425-1B-Instruct"
    hf_split: str = "train"
    num_train_chunks: int = 1
    val_fraction: float = 0.0
    seed: int = 42
    name: str = "dataset"
    dtype: str = "uint32"

    @property
    def relpath(self) -> str:
        return f"TokenizedMessagesDataset/{self.name}"

    @property
    def train_chunks(self) -> Tuple[Chunk, ...]:
        return tuple(
            Chunk(uri=f"{self.relpath}/train-{i}.bin")
            for i in range(self.num_train_chunks)
        )

    @property
    def train_chunk(self) -> Chunk:
        if self.num_train_chunks != 1:
            raise ValueError("train_chunk is only available when num_train_chunks == 1; use train_chunks")
        return self.train_chunks[0]

    @property
    def val_chunk(self) -> Chunk:
        return Chunk(uri=f"{self.relpath}/val.bin")

    @property
    def exists(self) -> bool:
        if self.num_train_chunks < 1:
            return False
        return all(
            check_exists_remote(remote_path(self.relpath, f"train-{i}.bin"))
            for i in range(self.num_train_chunks)
        )

    def get_requirements(self):
        return {"cpus": 8, "mem": "64G"}

    def construct(self, task: Task):
        if self.num_train_chunks < 1:
            raise ValueError("num_train_chunks must be >= 1")

        output_dir = os.path.join(local_path(), self.relpath)
        task.ensure_directory(output_dir)

        task.run_command(
            f"python -m mixture_pretraining.tokenize_messages",
            kwargs={
                "hf-dataset": self.hf_dataset,
                "messages-column": self.messages_column,
                "tokenizer": self.tokenizer,
                "hf-split": self.hf_split,
                "output-dir": output_dir,
                "num-train-chunks": self.num_train_chunks,
                "val-fraction": self.val_fraction,
                "seed": self.seed,
                "dtype": self.dtype,
            },
        )

        task.upload_to_gs(
            output_dir, remote_path(self.relpath), directory=True, contents=True,
        )
