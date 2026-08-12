#!/usr/bin/env python3
"""Stage 1 of the 60M chinchilla-4 PT-sweep plotting pipeline.

Downloads the PTSweep60M CPT eval JSONs from GCS and aggregates them into one
local results file, mirroring preprocess_cpt_evals.py -> cpt_results.json.

The difference from preprocess_cpt_evals.py is the name parsing: PTSweep60M runs
encode the PRETRAIN hyperparameters in the model name
(``-adamw-lr7.0e-3-wd0.1-bs1M-``), and the whole point of this sweep is to
compare those, so we record pretrain_lr / weight_decay / batch_size per record.
The MuonExpt3 regex drops them.

Usage:
  python -m new_utils.preprocess_pt_sweep_evals              # download + aggregate
  python -m new_utils.preprocess_pt_sweep_evals --no-download  # re-aggregate cache
"""
import argparse
import json
import os
import re
import subprocess

BUCKET = "cmu-gpucloud-catheri4"
EVAL_PREFIX = "Optim-60M-tuning/ModelEvaluation/"
RUN_PREFIX = "PTSweep60M"
_DIVERSITY = ("C4_val", "Reddit_val", "Wiki_val", "Books_val")

# Pretrain half of the name. The LR field differs by optimizer:
#   adamw -> lr7.0e-3
#   muon  -> muonlr1.0e-2-adamwlr7.0e-3
_PRE_RE = re.compile(
    r"^PTSweep60M-(?P<size>[0-9.]+B)-chinchilla-(?P<chin>[0-9]+)-"
    r"(?P<popt>adamw|muon)-(?P<plr>.+?)-wd(?P<wd>[0-9.]+)-bs(?P<bs>[0-9]+[kM]?)-wsd"
)
_PRE_ADAMW_LR = re.compile(r"^lr([0-9.eE+\-]+)$")
_PRE_MUON_LR = re.compile(r"^muonlr([0-9.eE+\-]+)-adamwlr([0-9.eE+\-]+)$")

# CPT half. Note this is DELIBERATELY looser than preprocess_cpt_evals._CPT_RE,
# which requires "-exp-muonlr<val>" to appear together and so silently drops every
# adamw-CPT file named "-lr<val>-exp-eval.json" (no muonlr). Here "-exp" and
# "-muonlr<val>" are independently optional, which matches all three forms:
#   -lr8.0e-4-eval.json                  (older MuonExpt3 naming)
#   -lr1.0e-2-exp-eval.json              (adamw CPT)
#   -lr1.5e-4-exp-muonlr6.0e-4-eval.json (muon CPT)
_CPT_RE = re.compile(
    r"-CPT-([a-z0-9]+)-\d+M-(adamw|muon)-lr([0-9._eE+\-]+?)"
    r"(?:-exp)?(?:-muonlr([0-9._eE+\-]+))?-eval\.json$"
)


def _num(s):
    return float(s.replace("_", "."))


def parse_name(filename):
    """-> dict of pretrain + CPT hyperparameters, or None if the name doesn't match."""
    pre = _PRE_RE.search(filename)
    cpt = _CPT_RE.search(filename)
    if not pre or not cpt:
        return None

    plr_raw = pre.group("plr")
    popt = pre.group("popt")
    if popt == "adamw":
        m = _PRE_ADAMW_LR.match(plr_raw)
        if not m:
            return None
        pretrain_lr, pretrain_muon_lr = _num(m.group(1)), None
    else:
        m = _PRE_MUON_LR.match(plr_raw)
        if not m:
            return None
        # For muon the sweep varies muon_lr, so that is the axis to group on.
        pretrain_muon_lr, pretrain_lr = _num(m.group(1)), _num(m.group(2))

    ds, copt, adamw_lr, muon_lr = cpt.groups()
    cpt_lr = _num(muon_lr) if (copt == "muon" and muon_lr) else _num(adamw_lr)

    return dict(
        dataset=ds,
        chinchilla=int(pre.group("chin")),
        pretrain_optimizer=popt,
        pretrain_lr=pretrain_lr,               # adamw LR (or muon's adamw component)
        pretrain_muon_lr=pretrain_muon_lr,     # None for adamw-pretrained
        # The value the sweep actually varied, for grouping the frontiers.
        sweep_lr=pretrain_muon_lr if popt == "muon" else pretrain_lr,
        weight_decay=float(pre.group("wd")),
        batch_size=pre.group("bs"),
        cpt_optimizer=copt,
        cpt_lr=cpt_lr,
    )


def _gsutil_ls(d):
    out = subprocess.run(["gsutil", "ls", d], capture_output=True, text=True)
    if out.returncode != 0:
        # Most often ReauthUnattendedError: gsutil's user creds expire in
        # non-interactive shells. `gcloud auth login` from a terminal fixes it.
        print(f"[warn] gsutil ls exited {out.returncode}:\n{out.stderr.strip()[:400]}")
    return [l.strip() for l in out.stdout.splitlines() if l.strip().endswith(".json")]


def _wanted(basename, kind):
    """kind: 'cpt' (finetune evals), 'pretrain' (base-model evals), or 'both'."""
    if not basename.startswith(RUN_PREFIX) or not basename.endswith("-eval.json"):
        return False
    is_cpt = "-CPT-" in basename
    return {"cpt": is_cpt, "pretrain": not is_cpt, "both": True}[kind]


def download(local_dir, kind="cpt"):
    """Download the PTSweep60M eval JSONs of `kind` from GCS into local_dir (gsutil)."""
    os.makedirs(local_dir, exist_ok=True)
    eval_dir = f"gs://{BUCKET}/{EVAL_PREFIX}"
    all_files = _gsutil_ls(eval_dir)
    files = [f for f in all_files if _wanted(os.path.basename(f), kind)]
    print(f"{len(files)} PTSweep60M {kind} eval JSON(s) to download -> {local_dir}")
    if not files:
        return files
    # -n skips files already present locally, so re-runs are cheap.
    proc = subprocess.run(
        ["gsutil", "-m", "cp", "-n", "-I", local_dir],
        input="\n".join(files), text=True,
    )
    if proc.returncode != 0:
        print(f"[warn] gsutil cp exited {proc.returncode} (some files may be missing)")
    return files


def parse_pretrain_name(filename):
    """Pretrain (non-CPT) eval: the model name alone, no -CPT- half."""
    pre = _PRE_RE.search(filename)
    if not pre or "-CPT-" in filename:
        return None
    plr_raw, popt = pre.group("plr"), pre.group("popt")
    if popt == "adamw":
        m = _PRE_ADAMW_LR.match(plr_raw)
        if not m:
            return None
        pretrain_lr, pretrain_muon_lr = _num(m.group(1)), None
    else:
        m = _PRE_MUON_LR.match(plr_raw)
        if not m:
            return None
        pretrain_muon_lr, pretrain_lr = _num(m.group(1)), _num(m.group(2))
    return dict(
        chinchilla=int(pre.group("chin")),
        pretrain_optimizer=popt,
        pretrain_lr=pretrain_lr,
        pretrain_muon_lr=pretrain_muon_lr,
        sweep_lr=pretrain_muon_lr if popt == "muon" else pretrain_lr,
        weight_decay=float(pre.group("wd")),
        batch_size=pre.group("bs"),
    )


def aggregate_pretrain(local_dir):
    """Pretrain evals have no fine-tuning loss — just the val losses per split."""
    records, skipped = [], 0
    for name in sorted(os.listdir(local_dir)):
        if not _wanted(name, "pretrain"):
            continue
        info = parse_pretrain_name(name)
        if info is None:
            skipped += 1
            continue
        try:
            with open(os.path.join(local_dir, name)) as f:
                d = json.load(f)
        except Exception as exc:
            print(f"[warn] read failed {name}: {exc}")
            continue
        by_label = d.get("by_label") or {}
        losses = {"overall": (d.get("overall") or {}).get("loss")}
        for k in ("DCLM_heldout",) + _DIVERSITY:
            if k in by_label:
                losses[k] = by_label[k].get("loss")
        if all(v is None for v in losses.values()):
            skipped += 1
            continue
        records.append({**info, "losses": losses, "file": name})
    print(f"aggregated {len(records)} pretrain record(s) ({skipped} skipped)")
    return records


def aggregate(local_dir):
    records, skipped = [], 0
    for name in sorted(os.listdir(local_dir)):
        if not _wanted(name, "cpt"):
            continue
        info = parse_name(name)
        if info is None:
            skipped += 1
            continue
        try:
            with open(os.path.join(local_dir, name)) as f:
                d = json.load(f)
        except Exception as exc:
            print(f"[warn] read failed {name}: {exc}")
            continue
        by_label = d.get("by_label") or {}
        ft = by_label.get(f"{info['dataset']}-validation", {}).get("loss")
        if ft is None:
            skipped += 1
            continue
        losses = {"overall": (d.get("overall") or {}).get("loss")}
        for k in ("DCLM_heldout",) + _DIVERSITY:
            if k in by_label:
                losses[k] = by_label[k].get("loss")
        records.append({**info, "finetune_loss": ft, "losses": losses, "file": name})
    print(f"aggregated {len(records)} record(s) ({skipped} skipped)")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default="results/pt_sweep_60m/eval_json")
    ap.add_argument("--out", default="results/pt_sweep_60m/pt_sweep_results.json")
    ap.add_argument("--pretrain-out",
                    default="results/pt_sweep_60m/pt_sweep_pretrain_results.json")
    ap.add_argument("--kind", default="both", choices=["cpt", "pretrain", "both"],
                    help="Which evals to fetch/aggregate (default both).")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    if not args.no_download:
        download(args.local_dir, args.kind)
    elif not os.path.isdir(args.local_dir):
        print(f"{args.local_dir} not found — run without --no-download first.")
        return

    def _write(recs, path, what):
        if not recs:
            print(f"no {what} records; not writing {path}")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(recs, f, indent=2)
        print(f"wrote {path}")
        lrs = sorted({r["sweep_lr"] for r in recs})
        print(f"  optimizers : {sorted({r['pretrain_optimizer'] for r in recs})}")
        print(f"  sweep LRs  : {['%.1e' % v for v in lrs]}")
        print(f"  weight decay: {sorted({r['weight_decay'] for r in recs})}")
        print(f"  batch sizes : {sorted({r['batch_size'] for r in recs})}")

    if args.kind in ("cpt", "both"):
        recs = aggregate(args.local_dir)
        _write(recs, args.out, "CPT")
        if recs:
            print(f"  datasets   : {sorted({r['dataset'] for r in recs})}")
    if args.kind in ("pretrain", "both"):
        _write(aggregate_pretrain(args.local_dir), args.pretrain_out, "pretrain")


if __name__ == "__main__":
    main()
