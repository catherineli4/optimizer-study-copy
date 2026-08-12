#!/usr/bin/env python3
"""Score one or more unsharded OLMo checkpoints on a memmap validation shard.

Numerics match training: fp32 master weights, bf16 autocast matmuls, fp32 loss.
Used to place a saved checkpoint on the run's in-loop wandb eval curve.
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


@torch.no_grad()
def score(model, path, device, chunk_size, batch_size, max_instances):
    arr = np.memmap(path, mode="r", dtype=np.uint32)
    n = arr.shape[0] // chunk_size
    if max_instances:
        n = min(n, max_instances)
    lsum, ntok = 0.0, 0
    batch = []
    for i in range(n):
        batch.append(np.asarray(arr[i * chunk_size:(i + 1) * chunk_size]))
        if len(batch) == batch_size or i == n - 1:
            ids = torch.from_numpy(np.stack(batch).astype(np.int64)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(input_ids=ids)
            shift = logits[:, :-1, :].float()
            lsum += float(F.cross_entropy(
                shift.contiguous().view(-1, shift.shape[-1]),
                ids[:, 1:].contiguous().reshape(-1),
                reduction="sum",
            ).item())
            ntok += ids.shape[0] * (ids.shape[1] - 1)
            batch = []
    return lsum / ntok, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="label=/path/to/unsharded_dir (repeatable)")
    ap.add_argument("--val-path", required=True)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for spec in args.checkpoint:
        label, ckpt = spec.split("=", 1)
        model = load_model(ckpt, device)
        loss, ntok = score(model, args.val_path, device, args.chunk_size,
                           args.batch_size, args.max_instances)
        results[label] = {"loss": loss, "num_tokens": ntok}
        print(f"{label:12s} CE {loss:.4f}  ({ntok} tokens)", flush=True)
        del model
        torch.cuda.empty_cache()

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
