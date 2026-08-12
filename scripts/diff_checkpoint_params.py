#!/usr/bin/env python3
"""Report which parameters actually changed between two unsharded checkpoints.

Muon runs save checkpoints whose loss stops improving during the LR decay phase
even though the trainer's in-loop eval keeps improving. If only a subset of
tensors is still moving between saves, this prints which ones.
"""
import argparse
import os

import torch


def load(path: str):
    if os.path.isdir(path):
        path = os.path.join(path, "model.pt")
    return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="earlier checkpoint dir or model.pt")
    ap.add_argument("--b", required=True, help="later checkpoint dir or model.pt")
    ap.add_argument("--top", type=int, default=0, help="only print N largest/smallest movers")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    shared = [k for k in a if k in b]
    missing = sorted(set(a) ^ set(b))
    if missing:
        print(f"keys present in only one checkpoint: {missing}")

    rows = []
    for k in shared:
        ta, tb = a[k].float(), b[k].float()
        delta = (tb - ta).norm().item()
        base = ta.norm().item()
        rows.append((k, tuple(ta.shape), delta, base, delta / base if base else float("nan")))

    unchanged = [r for r in rows if r[2] == 0.0]
    print(f"\n{len(unchanged)}/{len(rows)} tensors are bit-identical between the two checkpoints")
    if unchanged:
        print("identical tensors:")
        for k, shape, *_ in unchanged:
            print(f"   {k:55s} {str(shape)}")

    rows.sort(key=lambda r: r[4])
    print(f"\n{'parameter':55s} {'shape':>18s} {'||Δ||':>12s} {'||W||':>12s} {'rel':>10s}")
    shown = rows if not args.top else rows[: args.top] + rows[-args.top:]
    for k, shape, delta, base, rel in shown:
        print(f"{k:55s} {str(shape):>18s} {delta:12.4e} {base:12.4e} {rel:10.3e}")

    ndim2 = [r for r in rows if len(r[1]) == 2 and "embeddings" not in r[0] and "w_out" not in r[0]]
    other = [r for r in rows if r not in ndim2]
    def avg(rs):
        return sum(r[4] for r in rs) / len(rs) if rs else float("nan")
    print(f"\nmean relative change, 2D hidden matrices (Muon group): {avg(ndim2):.3e}  "
          f"[{len(ndim2)} tensors]")
    print(f"mean relative change, everything else (AdamW group):   {avg(other):.3e}  "
          f"[{len(other)} tensors]")


if __name__ == "__main__":
    main()
