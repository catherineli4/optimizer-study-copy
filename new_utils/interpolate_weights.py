#!/usr/bin/env python3
"""
Interpolate between a pretrained model and a finetuned model and save to GCS.

Computes a convex combination of the two checkpoints' weights:

    W_interp = alpha * W_pretrained + (1 - alpha) * W_finetuned

per matching parameter tensor, and writes the result as a new unsharded OLMo
checkpoint (``final-unsharded/model.pt``) under a fresh run directory on GCS.

The finetuned model's checkpoint directory is copied first (so ``config.json``
and the rest of the unsharded layout come along), then ``model.pt`` is replaced
with the interpolated weights. Mirrors the I/O conventions of
``perturb_weights.py``.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from typing import Optional

import torch


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    subprocess.check_call(["gsutil", "-m", "cp", gcs_path, local_path])


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    subprocess.check_call(["gsutil", "-m", "cp", local_path, gcs_path])


def copy_gcs_directory(gcs_source: str, gcs_dest: str) -> None:
    """Copy a directory from one GCS location to another (recursive rsync)."""
    subprocess.check_call(
        ["gsutil", "-m", "rsync", "-r", gcs_source.rstrip("/") + "/", gcs_dest.rstrip("/") + "/"]
    )


def load_state_dict_from_gcs(gcs_path: str, device: str = "cpu") -> dict:
    """Load a (possibly wrapped) PyTorch state_dict from GCS, unwrapped."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_path = tmp_file.name
    try:
        print(f"📥 Downloading {gcs_path} ...")
        download_from_gcs(gcs_path, tmp_path)
        print("📂 Loading state dict ...")
        state_dict = torch.load(tmp_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        return state_dict
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_state_dict_to_gcs(state_dict: dict, gcs_path: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_path = tmp_file.name
    try:
        print("💾 Saving interpolated model locally ...")
        torch.save(state_dict, tmp_path)
        print(f"☁️  Uploading interpolated model to {gcs_path} ...")
        upload_to_gcs(tmp_path, gcs_path)
        print(f"✅ Uploaded interpolated model to {gcs_path}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def interpolate_state_dicts(pt_state: dict, ft_state: dict, alpha: float) -> dict:
    """W_interp = alpha * pt + (1 - alpha) * ft, per matching float tensor.

    Non-float tensors, or tensors missing/shape-mismatched between the two
    state dicts, fall back to the finetuned model's value (the interpolated
    checkpoint inherits the finetuned model's architecture/config)."""
    out = {}
    n_interp = 0
    n_copied = 0
    for name, W_ft in ft_state.items():
        W_pt = pt_state.get(name)
        is_float = W_ft.dtype in (torch.float32, torch.float16, torch.bfloat16)
        if (W_pt is not None and is_float
                and tuple(W_pt.shape) == tuple(W_ft.shape)):
            orig_dtype = W_ft.dtype
            mixed = (alpha * W_pt.to(torch.float32)
                     + (1.0 - alpha) * W_ft.to(torch.float32))
            out[name] = mixed.to(orig_dtype)
            n_interp += 1
        else:
            out[name] = W_ft
            n_copied += 1
    missing_in_ft = [k for k in pt_state if k not in ft_state]
    if missing_in_ft:
        print(f"⚠️  {len(missing_in_ft)} pretrained key(s) not in finetuned model "
              f"(ignored), e.g. {missing_in_ft[:3]}")
    print(f"🔧 interpolated {n_interp} tensor(s), copied {n_copied} from finetuned "
          f"(alpha={alpha})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interpolate alpha*pretrained + (1-alpha)*finetuned weights -> GCS"
    )
    parser.add_argument("--pt_gcs_dir", required=True,
                        help="GCS parent dir of the pretrained model run dir.")
    parser.add_argument("--pt_model_name", required=True,
                        help="Pretrained run dir name (holds final-unsharded/model.pt).")
    parser.add_argument("--ft_gcs_dir", required=True,
                        help="GCS parent dir of the finetuned model run dir.")
    parser.add_argument("--ft_model_name", required=True,
                        help="Finetuned run dir name (holds final-unsharded/model.pt).")
    parser.add_argument("--alpha", type=float, required=True,
                        help="Weight on the PRETRAINED model: alpha*pt + (1-alpha)*ft.")
    parser.add_argument("--output_gcs_dir", required=True,
                        help="GCS parent dir for the interpolated model run dir.")
    parser.add_argument("--output_model_name", required=True,
                        help="Run dir name for the interpolated model.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        pt_dir = f"{args.pt_gcs_dir.rstrip('/')}/{args.pt_model_name}"
        ft_dir = f"{args.ft_gcs_dir.rstrip('/')}/{args.ft_model_name}"
        pt_model_path = f"{pt_dir}/final-unsharded/model.pt"
        ft_model_path = f"{ft_dir}/final-unsharded/model.pt"

        out_dir = f"{args.output_gcs_dir.rstrip('/')}/{args.output_model_name}"
        out_model_path = f"{out_dir}/final-unsharded/model.pt"

        print(f"📋 pretrained : {pt_model_path}")
        print(f"📋 finetuned  : {ft_model_path}")
        print(f"📋 output     : {out_model_path}")

        # Step 1: copy the finetuned checkpoint dir (config.json etc.) to output.
        print(f"📦 Copying {ft_dir} -> {out_dir} ...")
        copy_gcs_directory(ft_dir, out_dir)

        # Step 2: load both state dicts and interpolate.
        pt_state = load_state_dict_from_gcs(pt_model_path, device=args.device)
        ft_state = load_state_dict_from_gcs(ft_model_path, device=args.device)
        interp_state = interpolate_state_dicts(pt_state, ft_state, args.alpha)

        # Step 3: replace model.pt in the copied (output) checkpoint dir.
        save_state_dict_to_gcs(interp_state, out_model_path)

        print("🎉 Done! Interpolated checkpoint created successfully.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
