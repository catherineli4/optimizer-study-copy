#!/usr/bin/env python3
"""
Script to load evaluation JSON files from GCS, extract C4 perplexity values,
and plot them against the number of tokens, accounting for multiple groups.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """Download a file from GCS to local path."""
    subprocess.check_call(["gsutil", "cp", gcs_path, local_path])


def load_json_from_gcs(gcs_path: str) -> dict:
    """Load a JSON file from GCS."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        download_from_gcs(gcs_path, tmp_path)
        with open(tmp_path, 'r') as f:
            data = json.load(f)
        return data
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def list_gcs_files(gcs_dir: str) -> List[str]:
    """List all files in a GCS directory."""
    result = subprocess.run(
        ["gsutil", "ls", gcs_dir],
        capture_output=True,
        text=True,
        check=True
    )
    return [line.strip() for line in result.stdout.split('\n') if line.strip() and line.strip().endswith('.json')]


def extract_tokens_from_filename(filename: str) -> Optional[int]:
    """Extract token count from filename like 'OLMo-tk16B-...'."""
    match = re.search(r'tk(\d+)B', filename)
    if match:
        return int(match.group(1))
    return None

def extract_tokens_from_filename_adamw(filename: str) -> Optional[int]:
    """Extract token count from filename like 'OLMo-tk16B-...'."""
    match = re.search(r'(\d+)B', filename)
    if match:
        return int(match.group(1))
    return None


def parse_sci_number(s: str) -> float:
    """
    Parse a string representing a scientific number, handling underscores.
    Examples:
        '5e-3'    -> 0.005
        '5.0e-3'  -> 0.005
        '5_00e-3' -> 0.005
    """
    s_clean = s.replace("_", "")
    match = re.match(r'^([0-9.]+)e([+-]?[0-9]+)$', s_clean, re.IGNORECASE)
    if match:
        mantissa, exponent = match.groups()
        return float(mantissa) * (10 ** int(exponent))
    else:
        return float(s_clean)


def extract_group_info(filename: str) -> Dict[str, Optional[str]]:
    """Extract group information from filename."""
    info = {}

    # Extract tokens (e.g., tk8B → 8)
    info['tokens'] = extract_tokens_from_filename_adamw(filename)

    # Regex for scientific notation: digits, optional decimal, 'e', sign, digits
    sci_pattern = r'(\d+(?:[._]\d+)?[eE][+-]?\d+)'

    # Extract muon_lr (optional) - look for second occurrence
    muon_lrs = re.findall(r'-muon_lr' + sci_pattern, filename)
    info['muon_lr'] = muon_lrs[1] if len(muon_lrs) > 1 else None

    if info['muon_lr'] is None:
        # Extract lr - look for second occurrence (first is pretrain lr, second is CPT lr)
        lrs = re.findall(r'-lr' + sci_pattern, filename)
        lr_str = lrs[1] if len(lrs) > 1 else None
        info['lr'] = lr_str
        if lr_str:
            try:
                # Handle underscores in number (e.g., 1_00e-3 -> 1.00e-3)
                lr_clean = lr_str.replace('_', '.')
                lr_val = float(lr_clean)
                info['group'] = f"lr={lr_val:.2e}"
            except ValueError:
                info['group'] = f"lr={lr_str}"
        else:
            info['group'] = "lr=unknown"
    else:
        try:
            muon_lr_clean = info['muon_lr'].replace('_', '.')
            muon_lr_val = float(muon_lr_clean)
            info['group'] = f"lr={muon_lr_val:.2e}"
        except ValueError:
            info['group'] = f"lr={info['muon_lr']}"

    return info


def collect_results(gcs_base_dir: str, c4_key: str, baseline_map: Dict[int, float]) -> Dict[str, Dict[int, float]]:
    """Collect perturbation data from a single GCS directory."""
    try:
        print(f"\nListing files in {gcs_base_dir}...")
        all_files = list_gcs_files(gcs_base_dir)
        print(f"Found {len(all_files)} JSON files")
    except Exception as exc:
        print(f"Could not list directory {gcs_base_dir}: {exc}")
        return {}

    results: Dict[str, Dict[int, float]] = {}

    for gcs_file in all_files:
        try:
            # Extract filename from path
            filename = os.path.basename(gcs_file)

            if (not (("CPT" in filename) and ("starcoder" in filename))) :
                print(f"Skipping non starcoder CPT file: {filename}")
                continue
            
            # Extract group info
            info = extract_group_info(filename)
            group = info['group']
            tokens = info['tokens']
            
            if tokens is None:
                print(f"Warning: Could not extract tokens from {filename}, skipping")
                continue
            
            # Load JSON and extract C4 perplexity
            data = load_json_from_gcs(gcs_file)
            c4_pplx = data.get(c4_key)
            
            if c4_pplx is None:
                print(f"Warning: C4 perplexity not found in {filename}, skipping")
                continue
            
            # Store result
            if tokens not in baseline_map:
                print(f"Warning: No baseline value for {tokens}B tokens, skipping")
                continue

            if group not in results:
                results[group] = {}
            results[group][tokens] = float(c4_pplx)
            
            print(f"✓ {filename}: group={group}, tokens={tokens}B, c4_pplx={c4_pplx:.4f}")
            
        except Exception as e:
            print(f"Error processing {gcs_file}: {e}")
            continue
    
    print(f"\nCollected data for {len(results)} group(s) in {gcs_base_dir}")
    for group, data in results.items():
        print(f"  {group}: {len(data)} data points")

    return results


def plot_single(
    ax,
    results: Dict[str, Dict[int, float]],
    baseline_map: Dict[int, float],
    title: str,
    optimizer_label: str,
    linestyle: str,
    marker: str,
    perturb_color_map: Dict[str, tuple],
) -> None:
    """Plot ΔC4 perplexity curves for a single optimizer."""
    if not results:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return

    xticks = set()


    baseline_tokens = sorted(baseline_map.keys())
    baseline_values = [baseline_map[t] for t in baseline_tokens]
    # ax.plot(
    #     baseline_tokens,
    #     baseline_values,
    #     linestyle='-.',
    #     color='black',
    #     linewidth=2,
    #     label='Baseline',
    #  )

    def sort_key_for_group(group: str) -> float:
        """Extract numeric value for sorting groups."""
        match = re.search(r'(?:muon_)?lr\s*=\s*([0-9eE\+\-\._]+)', group)
        if match:
            try:
                return float(match.group(1).replace("_", ""))
            except ValueError:
                return float('inf')
        return float('inf')

    for group, data in sorted(results.items(), key=lambda x: sort_key_for_group(x[0])):
        tokens_list = sorted(data.keys())
        if not tokens_list:
            continue
        xticks.update(tokens_list)
        c4_values = [data[t] for t in tokens_list]
        color = perturb_color_map.get(group, '#444444')
        # Use just the lr value for legend (color already indicates lr)
        legend_label = group

        ax.plot(
            tokens_list,
            c4_values,
            marker=marker,
            label=legend_label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            markersize=7,
        )

        last_token = tokens_list[-1]
        last_value = c4_values[-1]
        # Extract the value part from group label (e.g., "lr=1.00e-04" -> "1.00e-04")
        label_value = group.replace("muon_lr=", "").replace("lr=", "")
        ax.annotate(
            label_value,
            (last_token, last_value),
            textcoords="offset points",
            xytext=(6, 0),
            ha='left',
            va='center',
            fontsize=8,
            color=color,
        )

    ax.set_xscale('log')
    ax.set_xticks(sorted({4, 8, 16, 32, 64}))
    ax.set_xticklabels(['4', '8', '16', '32', '64'])
    ax.set_xlabel('Pretrain Tokens (B)', fontsize=12)
    ax.set_ylabel(r'Alpaca Perplexity', fontsize=12)
    ax.set_title(title, fontsize=14)

    handles, labels = ax.get_legend_handles_labels()

    def key_for_label(label):
        """Extract numeric lr value for sorting. Returns (priority, value)."""
        # Try muon_lr first
        match = re.search(r'muon_lr\s*=\s*([0-9eE\+\-\._]+)', label)
        if match:
            try:
                return (0, float(match.group(1).replace("_", "")))
            except ValueError:
                return (2, 0)  # non-numeric goes last
        # Try lr
        match = re.search(r'lr\s*=\s*([0-9eE\+\-\._]+)', label)
        if match:
            try:
                return (0, float(match.group(1).replace("_", "")))
            except ValueError:
                return (2, 0)
        # Baseline or other labels go at the top
        if 'Baseline' in label:
            return (-1, 0)
        return (2, 0)

    if handles:
        # Sort by numeric value ascending (smallest lr first, baseline at top)
        pairs = sorted(zip(labels, handles), key=lambda lh: key_for_label(lh[0]))
        sorted_labels, sorted_handles = zip(*pairs)
        # Place legend outside the plot on the right
        ax.legend(
            sorted_handles, sorted_labels,
            fontsize=8,
            title='Runs',
            title_fontsize=9,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
        )

    ax.grid(True, alpha=0.3)
    if xticks:
        ax.set_xticks(sorted(xticks))


def main():
    parser = argparse.ArgumentParser(description="Plot perturbation results from two GCS directories.")
    parser.add_argument(
        "--gcs-base-dir-a",
        default="gs://cmu-gpucloud-catheri4/outputs/muon/ModelEvaluation",
        help="First GCS directory containing evaluation JSON files.",
    )
    parser.add_argument(
        "--gcs-base-dir-b",
        default="gs://cmu-gpucloud-catheri4/outputs/adamw/ModelEvaluation",
        help="Second GCS directory containing evaluation JSON files.",
    )
    parser.add_argument(
        "--title-a",
        default="Muon",
        help="Title for the first subplot (defaults to directory path).",
    )
    parser.add_argument(
        "--title-b",
        default="AdamW",
        help="Title for the second subplot (defaults to directory path).",
    )
    parser.add_argument(
        "--c4-key",
        default="preprocessed_starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00001", #eval-data_perplexity_v3_small_gptneox20b_c4_en_val_part-0-00000",
        help="Key to retrieve C4 perplexity from JSON files.",
    )
    parser.add_argument(
        "--output",
        default="starcoder_comparison_11-26.png",
        help="Path to save the output plot.",
    )
    args = parser.parse_args()

    tokens_list = [4, 8, 16, 32, 64]
    baseline_c4_adamw = [4.000, 3.903, 3.83577, 3.78500, 3.74899]
    baseline_map_adamw = {t: v for t, v in zip(tokens_list, baseline_c4_adamw)}
    baseline_c4_muon = [3.88679, 3.85614, 3.83773, 3.82851, 3.79068]
    baseline_map_muon = {t: v for t, v in zip(tokens_list, baseline_c4_muon)}

    results_muon = collect_results(args.gcs_base_dir_a, args.c4_key, baseline_map_muon)
    results_adamw = collect_results(args.gcs_base_dir_b, args.c4_key, baseline_map_adamw)

    if not (results_muon or results_adamw):
        print("No data collected from either directory. Exiting.")
        return

    all_groups = list(set(results_muon.keys()) | set(results_adamw.keys()))
    if not all_groups:
        print("No perturbation-matched runs found to plot.")
        return

    def parse_group_value(group: str) -> float:
        """Extract numeric value from group string like 'lr=1.00e-04'."""
        match = re.search(r'lr=([0-9eE.+-]+)', group)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return float('inf')
        return float('inf')

    # Sort groups by their numeric lr value
    all_groups_sorted = sorted(all_groups, key=parse_group_value)
    
    # Create color map - same lr string gets same color
    cmap = plt.get_cmap("tab10")
    perturb_color_map = {
        group: cmap(i % cmap.N)
        for i, group in enumerate(all_groups_sorted)
    }
    
    print(f"\nColor mapping:")
    for i, group in enumerate(all_groups_sorted):
        print(f"  {group} -> color {i}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    ax_muon, ax_adamw = axes

    plot_single(
        ax_muon,
        results_muon,
        baseline_map_muon,
        fr"Starcoder Perplexity vs Tokens ({args.title_a}/Post Starcoder-CPT)",
        args.title_a,
        '-',
        'o',
        perturb_color_map,
    )

    plot_single(
        ax_adamw,
        results_adamw,
        baseline_map_adamw,
        fr"Starcoder Perplexity vs Tokens ({args.title_b}/Post Starcoder-CPT)",
        args.title_b,
        '--',
        's',
        perturb_color_map,
    )

    # Adjust layout to make room for legends on the right
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print(f"Saved comparison plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()