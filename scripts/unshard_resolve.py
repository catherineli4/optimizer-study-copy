#!/usr/bin/env python3
"""Unshard the `final/` checkpoint — or the latest `step<N>/` if `final/` is absent.

JOLMo's checkpointer only writes a `final/` checkpoint when the last training
step does NOT coincide with a scheduled save. When the final step IS a save
point (e.g. total_steps is a multiple of save_interval / ephemeral_save_interval),
no `final/` is written, so a postprocess that hard-codes `final/` finds nothing
to unshard and never produces `final-unsharded/`.

This wrapper resolves the right checkpoint under a save folder — preferring
`final/`, falling back to the highest-numbered `step<N>/` — runs JOLMo's
`unshard.py` on it, and copies that checkpoint's `config.json` alongside the
unsharded weights (unshard.py does not copy it).

Used both by JolmoModel._postprocess (forward fix) and the
FinalizeUnshardedCheckpoint salvage artifact (recovery for already-trained runs).
"""

import argparse
import os
import re
import subprocess
import sys


def _ls(gs_dir: str):
    """List immediate entries under a gs:// (or local) dir; [] on failure."""
    r = subprocess.run(
        ["gsutil", "ls", gs_dir.rstrip("/") + "/"],
        capture_output=True, text=True,
    )
    return [line.rstrip("/") for line in r.stdout.splitlines()] if r.returncode == 0 else []


def resolve_checkpoint(save_folder: str) -> str:
    save_folder = save_folder.rstrip("/")
    entries = _ls(save_folder)
    if f"{save_folder}/final" in entries:
        return f"{save_folder}/final"
    steps = []
    for e in entries:
        m = re.search(r"/step(\d+)$", e)
        if m:
            steps.append((int(m.group(1)), e))
    if not steps:
        sys.exit(f"[unshard_resolve] no 'final/' or 'step*/' checkpoint under {save_folder}/")
    latest = max(steps, key=lambda t: t[0])[1]
    print(f"[unshard_resolve] no final/ — using latest checkpoint {latest}", file=sys.stderr)
    return latest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-folder", required=True, help="Dir containing final/ and/or step*/")
    ap.add_argument("--output-dir", required=True, help="Local dir for the unsharded model")
    ap.add_argument("--work-dir", required=True, help="Scratch dir for unshard.py")
    ap.add_argument("--unshard-script", required=True, help="Path to JOLMo src/scripts/unshard.py")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.save_folder)
    os.makedirs(args.output_dir, exist_ok=True)

    subprocess.check_call([
        "python3", args.unshard_script, ckpt, args.output_dir,
        "--overwrite", "--pre-download", "--work-dir", args.work_dir,
    ])

    # unshard.py writes model weights but not config.json — copy it next to them.
    if not os.path.isfile(os.path.join(args.output_dir, "config.json")):
        subprocess.run(
            ["gsutil", "cp", f"{ckpt}/config.json",
             os.path.join(args.output_dir, "config.json")],
            check=False,
        )


if __name__ == "__main__":
    main()
