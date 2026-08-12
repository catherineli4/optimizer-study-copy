#!/usr/bin/env python3
"""A/B the offline-eval numerics that made Muon look worse than it is.

For each checkpoint, computes C4_val CE three ways:

  bf16_weights_bf16_ce   what validate.py did with the ``model.to(bfloat16)`` cast
  fp32_weights_bf16_ce   isolates the weight rounding from the loss dtype
  autocast_fp32_ce       the fixed path: fp32 master weights, bf16 matmuls, fp32 loss

The gap between the first and last column is the eval artifact; compare it against
the in-loop wandb C4_val CE to see which column the training loop agrees with.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from olmo_core.nn.transformer import Transformer, TransformerConfig


def load_model(ckpt_dir: str, device: torch.device) -> Transformer:
    with open(os.path.join(ckpt_dir, "config.json"), encoding="utf-8") as f:
        exp_cfg = json.load(f)
    model: Transformer = TransformerConfig.from_dict(exp_cfg["model"]).build(init_device="cpu")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu"),
                          strict=True)
    return model.to(device=device).eval()


def iter_batches(path: str, chunk_size: int, batch_size: int, max_instances: int):
    arr = np.memmap(path, mode="r", dtype=np.uint32)
    n_full = min(arr.shape[0] // chunk_size, max_instances)
    batch = []
    for i in range(n_full):
        batch.append(np.asarray(arr[i * chunk_size:(i + 1) * chunk_size]))
        if len(batch) == batch_size:
            yield np.stack(batch, axis=0)
            batch = []
    if batch:
        yield np.stack(batch, axis=0)


@torch.no_grad()
def ce(model, path, device, chunk_size, batch_size, max_instances, weight_dtype, autocast, loss_fp32):
    if weight_dtype is not None:
        model = model.to(dtype=weight_dtype)
    lsum, ntok = 0.0, 0
    for np_batch in iter_batches(path, chunk_size, batch_size, max_instances):
        ids = torch.from_numpy(np_batch.astype(np.int64)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
            logits = model(input_ids=ids)
        shift_logits = logits[:, :-1, :]
        if loss_fp32:
            shift_logits = shift_logits.float()
        loss = F.cross_entropy(
            shift_logits.contiguous().view(-1, shift_logits.shape[-1]),
            ids[:, 1:].contiguous().reshape(-1),
            reduction="sum",
        )
        lsum += float(loss.item())
        ntok += ids.shape[0] * (ids.shape[1] - 1)
    return lsum / ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="label=/path/to/final-unsharded (repeatable)")
    ap.add_argument("--val-path", required=True)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-instances", type=int, default=256)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for spec in args.checkpoint:
        label, ckpt = spec.split("=", 1)
        kw = dict(path=args.val_path, device=device, chunk_size=args.chunk_size,
                  batch_size=args.batch_size, max_instances=args.max_instances)
        cuda = device.type == "cuda"
        row = {}
        # Reload per variant: casting down to bf16 and back would silently keep the
        # rounded weights and contaminate the fp32 arms.
        for name, weight_dtype, autocast, loss_fp32 in (
            ("autocast_fp32_ce", torch.float32, cuda, True),
            ("fp32_weights_bf16_ce", torch.float32, cuda, False),
            ("bf16_weights_bf16_ce", torch.bfloat16, False, False),
        ):
            model = load_model(ckpt, device)
            row[name] = ce(model, weight_dtype=weight_dtype, autocast=autocast,
                           loss_fp32=loss_fp32, **kw)
            del model
            torch.cuda.empty_cache()
        row["bias_from_bf16_eval"] = row["bf16_weights_bf16_ce"] - row["autocast_fp32_ce"]
        results[label] = row
        print(f"{label}: " + json.dumps(row, indent=2), flush=True)

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
