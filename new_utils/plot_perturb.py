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
    """Extract group information from filename, using only perturbation for the legend."""
    info = {}

    # Extract tokens (e.g., tk8B → 8)
    info['tokens'] = extract_tokens_from_filename_adamw(filename)

    # Extract muon_lr (optional)
    muon_lr_match = re.search(r'-muon_lr([0-9eE\+\-\.]+)', filename)
    info['muon_lr'] = muon_lr_match.group(1) if muon_lr_match else None

    # Extract perturbation, handling underscores in scientific notation like 1_00e-2
    # Stop matching before '-eval' or end of string
    perturb_match = re.search(r'_perturbed_([0-9_\.eE+\-]+)(?=-eval|$)', filename)
    if perturb_match:
        raw = perturb_match.group(1)  # e.g., '7_00e-2'
        # Replace '_' with '.' to get proper float
        raw = raw.replace("_", ".")
        try:
            info['perturb'] = float(raw)
        except ValueError:
            print(f"Warning: could not parse perturbation '{raw}' in filename {filename}")
            info['perturb'] = 0.0
    else:
        info['perturb'] = 0.0

    # Only use perturbation for group label (scientific notation)
    info['group'] = f"perturb={info['perturb']:.1e}"

    return info


def collect_results(gcs_base_dir: str, c4_key: str, baseline_map: Dict[int, float],
                    min_perturb: Optional[float] = None, max_perturb: Optional[float] = None) -> Dict[str, Dict[int, float]]:
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

            if not (("perturbed" in filename) or ("perturbed_4" in filename) or ("perturbed_3" in filename) or ("perturbed_2" in filename) or ("perturbed_1" in filename)):
                print(f"Skipping not perturbed_norm file: {filename}")
                continue
            
            if not ("20m" in filename):
                print(f"skip non-20m")
                continue 
            
            # Extract group info
            info = extract_group_info(filename)
            group = info['group']
            tokens = info['tokens']
            perturb = info['perturb']
            
            # Apply perturbation filters
            if min_perturb is not None and perturb < min_perturb:
                print(f"Skipping {filename}: perturb={perturb:.2e} < min_perturb={min_perturb:.2e}")
                continue
            if max_perturb is not None and perturb > max_perturb:
                print(f"Skipping {filename}: perturb={perturb:.2e} > max_perturb={max_perturb:.2e}")
                continue
            
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
    add_legend: bool = True,
) -> tuple:
    """Plot ΔC4 perplexity curves for a single optimizer.
    
    Returns:
        tuple: (handles, labels) for legend creation
    """
    if not results:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return (None, None)

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
    # )


    for group, data in sorted(results.items()):
        tokens_list = sorted(data.keys())
        if not tokens_list:
            continue
        xticks.update(tokens_list)
        c4_values = [data[t] for t in tokens_list]
        color = perturb_color_map.get(group, '#444444')
        # Use just the perturbation value for legend (without optimizer name)
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
        perturb_value = group.replace("perturb=", "")
        ax.annotate(
            perturb_value,
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
    ax.set_ylabel(r'C4 Perplexity', fontsize=12)
    ax.set_title(title, fontsize=14)

    handles, labels = ax.get_legend_handles_labels()

    def key_for_label(label):
        match = re.search(r'perturb\s*=\s*([0-9eE\+\-\.]+)', label)
        if match:
            try:
                return (0, float(match.group(1).replace("_", "")))
            except ValueError:
                return (1, label)
        return (1, label)

    sorted_handles, sorted_labels = None, None
    if handles:
        pairs = sorted(zip(labels, handles), key=lambda lh: key_for_label(lh[0]), reverse=True)
        sorted_labels, sorted_handles = zip(*pairs)
        if add_legend:
            ax.legend(sorted_handles, sorted_labels, fontsize=8, title='Perturbation', title_fontsize=9)

    ax.grid(True, alpha=0.3)
    if xticks:
        ax.set_xticks(sorted(xticks))
    
    return (sorted_handles, sorted_labels)


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
        default="eval-data_perplexity_v3_small_gptneox20b_c4_en_val_part-0-00000",
        help="Key to retrieve C4 perplexity from JSON files.",
    )
    parser.add_argument(
        "--output",
        default="c4_perturbation_comparison_01-17-all.png",
        help="Path to save the output plot.",
    )
    parser.add_argument(
        "--min-perturb",
        type=float,
        default=None,
        help="Minimum perturbation value to include (optional)",
    )
    parser.add_argument(
        "--max-perturb",
        type=float,
        default=None,
        help="Maximum perturbation value to include (optional)",
    )
    args = parser.parse_args()


    #personal experiment params
    tokens_list = [4, 8, 16, 32, 64]
    baseline_c4_adamw = [4.000, 3.903, 3.83577, 3.78500, 3.74899]
    baseline_c4_muon = [3.88679, 3.85614, 3.83773, 3.82851, 3.79068]

    #kaiyue experiment params
    #tokens_list = [1, 2, 4, 8, 16]
    baseline_kaiyue_c4_adamw = [3.6448381544717527, 3.5188259041025405, 3.4351424379539797, 3.3841493971520054, 3.3210996010891427]
    baseline_kaiyue_c4_muon = [3.577510022075361, 3.484690251639783, 3.400681450039192, 3.329430983608479, 3.270687291336516]


    #combined, mod depending on what used
    baseline_map_muon = {t: v for t, v in zip(tokens_list, baseline_c4_muon)}
    baseline_map_adamw = {t: v for t, v in zip(tokens_list, baseline_c4_adamw)}

    results_muon = collect_results(args.gcs_base_dir_a, args.c4_key, baseline_map_muon,
                                   min_perturb=args.min_perturb, max_perturb=args.max_perturb)
    results_adamw = collect_results(args.gcs_base_dir_b, args.c4_key, baseline_map_adamw,
                                    min_perturb=args.min_perturb, max_perturb=args.max_perturb)

    if not (results_muon or results_adamw):
        print("No data collected from either directory. Exiting.")
        return

    perturbations = sorted({*results_muon.keys(), *results_adamw.keys()})
    if not perturbations:
        print("No perturbation-matched runs found to plot.")
        return

    cmap = plt.get_cmap("tab20")
    color_palette = [cmap(i % cmap.N) for i in range(len(perturbations))]
    perturb_color_map = {
        perturb: color_palette[idx]
        for idx, perturb in enumerate(perturbations)
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    ax_muon, ax_adamw = axes

    # Plot muon without individual legend and get handles/labels
    handles_muon, labels_muon = plot_single(
        ax_muon,
        results_muon,
        baseline_map_muon,
        fr"C4 Perplexity vs Tokens ({args.title_a} Pretrain)",
        args.title_a,
        '-',
        'o',
        perturb_color_map,
        add_legend=False,
    )

    # Plot adamw without individual legend
    plot_single(
        ax_adamw,
        results_adamw,
        baseline_map_adamw,
        fr"C4 Perplexity vs Tokens ({args.title_b} Pretrain)",
        args.title_b,
        '--',
        's',
        perturb_color_map,
        add_legend=False,
    )

    # Add shared legend to the right of both subplots (using muon handles/labels)
    if handles_muon and labels_muon:
        fig.legend(handles_muon, labels_muon,
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=9,
                  title='Perturbation',
                  title_fontsize=10,
                  frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(right=0.95)  # Make room for the legend (less whitespace)
    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    print(f"Saved comparison plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()