# Using JOLMo in optimizer-study

This guide explains how to use JOLMo (OLMo-core) for training experiments in the optimizer-study project.

## Quick Start

### 1. Setup

The JOLMo codebase is already available in `optimizer-study/JOLMo/`. Make sure you have the dependencies:

```bash
cd JOLMo
pip install -e .[all]
```

### 2. Prepare Your Data

JOLMo uses `.npy` format for training data. If you have text data, convert it:

```python
import numpy as np
from olmo_core.data import TokenizerConfig

# Load tokenizer
tokenizer_config = TokenizerConfig.gpt2()
tokenizer = tokenizer_config.build()

# Tokenize your documents
documents = ["Your text here...", "More text..."]
token_sequences = []
for doc in documents:
    tokens = tokenizer.encode(doc)
    token_sequences.append(np.array(tokens, dtype=np.uint16))

# Save to .npy file
np.save("train_data.npy", np.concatenate(token_sequences))
```

### 3. Configure Your Experiment

Edit one of the example configs in `jolmo_configs/`:

**For standard AdamW training:**
```bash
cp jolmo_configs/example_adamw.yaml jolmo_configs/my_experiment.yaml
# Edit my_experiment.yaml with your settings
```

**For SAM training:**
```bash
cp jolmo_configs/example_sam.yaml jolmo_configs/my_sam_experiment.yaml
# Edit my_sam_experiment.yaml with your settings
```

Key settings to update:
- `dataset.paths`: Path to your `.npy` training data
- `trainer.save_folder`: Where to save checkpoints
- `wandb.*`: Your WandB settings
- `validation_datasets.*`: Optional validation data paths

### 4. Launch Training

**Single GPU:**
```bash
python launch_jolmo.py jolmo_configs/my_experiment.yaml
```

**Multi-GPU (e.g., 4 GPUs):**
```bash
torchrun --nproc-per-node=4 launch_jolmo.py jolmo_configs/my_experiment.yaml
```

**With command-line overrides:**
```bash
torchrun --nproc-per-node=4 launch_jolmo.py jolmo_configs/my_experiment.yaml \
    train_module.optim.lr=0.001 \
    n_tokens=200000000 \
    trainer.save_folder=/new/checkpoint/path
```

**Dry run (check config without training):**
```bash
python launch_jolmo.py jolmo_configs/my_experiment.yaml --dry-run
```

## Configuration Guide

### Model Architecture

Use factory methods for standard architectures:
```yaml
model_factory: llama2_271M  # or llama_1B, llama_7B, olmo_1B, etc.
model_factory_args:
  vocab_size: 50304
```

Or define custom architecture:
```yaml
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
  vocab_size: 50304
  # ... more settings
```

### Optimizer Settings

**AdamW:**
```yaml
train_module:
  optim:
    lr: 0.003
    betas: [0.9, 0.95]
    weight_decay: 0.1
```

**SAM (Sharpness-Aware Minimization):**
```yaml
train_module_type: sam
sam:
  rho: 0.05
  adaptive: false
```

### Training Duration

Specify by tokens:
```yaml
n_tokens: 100000000  # 100M tokens
```

Or by steps (in trainer config):
```yaml
trainer:
  max_duration: 10000  # 10K steps
```

### Distributed Training

FSDP (recommended):
```yaml
train_module:
  dp_config:
    name: fsdp
    param_dtype: bfloat16
    reduce_dtype: float32
```

### Checkpointing

```yaml
trainer:
  save_folder: /path/to/checkpoints
  save_interval: 1000  # Save every 1000 steps
  ephemeral_save_interval: 100  # Quick saves every 100 steps
```

### Loading from Checkpoint

For fine-tuning or continual pre-training:
```yaml
load_path: /path/to/checkpoint
load_optim_state: false      # Don't load optimizer state
load_trainer_state: false    # Don't load trainer state
load_data_loader_state: true # Do load data loader state
```

## Available Model Factories

From `TransformerConfig`:
- `llama2_271M` - 271M parameter Llama-style model
- `llama_1B` - 1B parameter model
- `llama_7B` - 7B parameter model
- `olmo_1B` - OLMo 1B architecture
- `olmo3_7B` - OLMo3 7B architecture

## Experiment Workflow

### 1. Pre-training
```bash
# Train base model
torchrun --nproc-per-node=4 launch_jolmo.py jolmo_configs/pretrain.yaml
```

### 2. Continual Pre-training (CPT)
```bash
# Fine-tune on domain-specific data
torchrun --nproc-per-node=4 launch_jolmo.py jolmo_configs/cpt_code.yaml \
    load_path=/checkpoints/pretrain/latest \
    load_optim_state=false \
    load_trainer_state=false
```

### 3. Evaluation
Use JOLMo's evaluation tools or convert to HuggingFace format:
```bash
# Convert to HF format
python JOLMo/src/examples/huggingface/convert_checkpoint_to_hf.py \
    --checkpoint-dir /checkpoints/my_run/step10000 \
    --output-dir /hf_models/my_run
```

## Comparison: Old OLMo vs JOLMo

| Feature | Old OLMo (catastrophic-forgetting) | JOLMo (OLMo-core) |
|---------|-----------------------------------|-------------------|
| Config Format | YAML with custom structure | YAML → dataclasses |
| Launch Script | `scripts/train.py` | `launch_jolmo.py` |
| SAM Support | ✅ Built-in | ✅ Built-in via `train_module_type: sam` |
| Attention | Flash Attention 2 | Flash 2/3, TransformerEngine |
| Data Format | Various | `.npy` files |
| Distributed | FSDP, DDP | FSDP, TP, PP, context parallelism |
| Checkpointing | Custom | Distributed checkpointing |

## Troubleshooting

### Import Errors
Make sure JOLMo is installed:
```bash
cd JOLMo
pip install -e .[all]
```

### Data Format Issues
JOLMo requires `.npy` format with `dtype=uint16` or `uint32`. Convert your data as shown in the data preparation section.

### Path Issues
All paths in YAML configs should be absolute or relative to where you run the script from.

### GPU Memory Issues
Reduce `train_module.rank_microbatch_size` (tokens per GPU) or enable gradient checkpointing.

## Advanced Features

### Multiple Validation Sets
```yaml
validation_datasets:
  c4: [/data/c4_val.npy]
  pile: [/data/pile_val.npy]
  code: [/data/code_val.npy]
validation_eval_interval: 250
```

### Custom Learning Rate Schedule
```yaml
train_module:
  scheduler:
    warmup_steps: 2000
    alpha_f: 0.1  # Final LR = 0.1 * initial_lr
```

### Float8 Training
Requires appropriate hardware and dependencies:
```yaml
train_module:
  dp_config:
    enable_fp8: true
```

## Support

For JOLMo-specific issues, refer to:
- JOLMo Documentation: `JOLMo/README.md`
- Example configs: `JOLMo/src/examples/`
- Official scripts: `JOLMo/src/scripts/official/`

For optimizer-study specific questions, see the main README.
