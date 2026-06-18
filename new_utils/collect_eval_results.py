#!/usr/bin/env python3
"""
Script to collect evaluation results from GCS and export to CSV.
Extracts token budget, C4 perplexity, and test values from JSON files.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Any


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """Download a file from GCS to local path."""
    subprocess.check_call(["gsutil", "cp", gcs_path, local_path], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    return [line.strip() for line in result.stdout.split('\n') 
            if line.strip() and line.strip().endswith('.json')]


def extract_tokens_from_filename(filename: str) -> Optional[int]:
    """Extract token count from filename like 'OLMo-tk16B-...'."""
    match = re.search(r'tk(\d+)B', filename)
    if match:
        return int(match.group(1))
    # Fallback: look for any NB pattern
    match = re.search(r'(\d+)B', filename)
    if match:
        return int(match.group(1))
    return None

def cpt_adamw(filename: str) -> Optional[str]:
    """Extract learning rate from filename."""
    # Pattern for scientific notation
    sci_pattern = r'(\d+(?:[._]\d+)?[eE][+-]?\d+)'
    
    # Try to find CPT lr (second occurrence)
    lrs = re.findall('-20m' , filename)

    return len(lrs) < 1

def extract_muon_lr_from_filename(filename: str) -> Optional[str]:
    """Extract learning rate from filename."""
    # Pattern for scientific notation
    sci_pattern = r'(\d+(?:[._]\d+)?[eE][+-]?\d+)'
    
    # Try to find CPT lr (second occurrence)
    lrs = re.findall(r'-muon_lr' + sci_pattern, filename)
    if len(lrs) > 1:
        lr_str = lrs[1].replace('_', '.')
        try:
            return f"{float(lr_str):.2e}"
        except ValueError:
            return lrs[1]
    elif len(lrs) == 1:
        lr_str = lrs[0].replace('_', '.')
        try:
            return f"{float(lr_str):.2e}"
        except ValueError:
            return lrs[0]
    return None

def extract_first_muon_lr_from_filename(filename: str) -> Optional[str]:
    """Extract learning rate from filename."""
    # Pattern for scientific notation
    sci_pattern = r'(\d+(?:[._]\d+)?[eE][+-]?\d+)'
    
    # Try to find CPT lr (second occurrence)
    lrs = re.findall(r'-muon_lr' + sci_pattern, filename)
    if len(lrs) > 0:
        return float(lrs[0])
    return None
 


def extract_lr_from_filename(filename: str) -> Optional[str]:
    """Extract learning rate from filename."""
    # Pattern for scientific notation
    sci_pattern = r'(\d+(?:[._]\d+)?[eE][+-]?\d+)'
    
    # Try to find CPT lr (second occurrence)
    lrs = re.findall(r'-lr' + sci_pattern, filename)
    if len(lrs) > 1:
        lr_str = lrs[1].replace('_', '.')
        try:
            return f"{float(lr_str):.2e}"
        except ValueError:
            return lrs[1]
    elif len(lrs) == 1:
        lr_str = lrs[0].replace('_', '.')
        try:
            return f"{float(lr_str):.2e}"
        except ValueError:
            return lrs[0]
    return None


def extract_optimizer_from_filename(filename: str) -> str:
    """Extract optimizer from filename."""
    if 'muon' in filename.lower():
        return 'muon'
    elif 'adamw' in filename.lower():
        return 'adamw'
    elif 'sgd' in filename.lower():
        return 'sgd'
    elif 'sam' in filename.lower():
        return 'sam'
    elif 'lionw' in filename.lower():
        return 'lionw'
    return 'unknown'


def extract_model_type(filename: str) -> str:
    """Extract model type (pretrain or CPT) from filename."""
    if 'CPT' in filename:
        return 'CPT'
    return 'pretrain'


def extract_dataset(filename: str) -> str:
    """Extract dataset name from filename."""
    if 'alpaca' in filename.lower():
        return 'alpaca'
    elif 'starcoder' in filename.lower():
        return 'starcoder'
    elif 'siqa' in filename.lower():
        return 'siqa'
    elif 'gsm8k' in filename.lower():
        return 'gsm8k'
    elif 'tulu' in filename.lower():
        return 'tulu'
    elif 'rte' in filename.lower():
        return 'rte'
    elif 'tinygsm' in filename.lower():
        return 'tinygsm'
    elif 'hotpotqa' in filename.lower():
        return 'hotpotQA'
    elif 'stackmathqa' in filename.lower():
        return 'stackMathQA'
    elif 'musicpile' in filename.lower():
        return 'musicPile'
    return 'unknown'


def simplify_key(key: str) -> str:
    """Simplify JSON key names for CSV columns."""
    if 'c4' in key.lower():
        return 'c4'
    if 'starcoder' in key.lower():
        return 'starcoder'
    return key.split('_')[-1] if '_' in key else key


def collect_results(gcs_base_dir: str) -> List[Dict[str, Any]]:
    """Collect all evaluation results from GCS directory."""
    print(f"Listing files in {gcs_base_dir}...")
    try:
        all_files = list_gcs_files(gcs_base_dir)
        print(f"Found {len(all_files)} JSON files")
    except Exception as exc:
        print(f"Error listing directory: {exc}")
        return []

    results = []
    
    for i, gcs_file in enumerate(all_files):
        try:
            filename = os.path.basename(gcs_file)
            print(f"[{i+1}/{len(all_files)}] Processing {filename}...")
            
            # Extract metadata from filename
            tokens = extract_tokens_from_filename(filename)
            lr = extract_lr_from_filename(filename)
            muon_lr = extract_muon_lr_from_filename(filename)
            optimizer = extract_optimizer_from_filename(filename)
            model_type = extract_model_type(filename)
            dataset = extract_dataset(filename)

            #filter
            if (model_type != 'CPT'):
                continue
            if cpt_adamw(filename):
                continue
            # if tokens == 4 and extract_first_muon_lr_from_filename(filename) == 4e-3:
            #     continue 

            if(dataset != "starcoder"):
                continue


            # Load JSON data
            data = load_json_from_gcs(gcs_file)

            
            # Build result row
            row = {
                'filename': filename,
                'Token Budget': tokens,
                'LR': lr,
                'optimizer': optimizer,
                'model_type': model_type,
                'dataset': dataset,
            }
            
            # Add all evaluation metrics from JSON
            for key, value in data.items():
                col_name = simplify_key(key)
                row[col_name] = value
            
            results.append(row)
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect evaluation results from GCS and export to CSV."
    )
    parser.add_argument(
        "--gcs-dir",
        type=str,
        default="gs://cmu-gpucloud-catheri4/outputs/adamw/ModelEvaluation",
        help="GCS directory containing evaluation JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="adamw_20m_01-17_star.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--filter-cpt",
        action="store_true",
        help="Only include CPT models.",
    )
    parser.add_argument(
        "--filter-pretrain",
        action="store_true",
        help="Only include pretrain models.",
    )
    parser.add_argument(
        "--filter-dataset",
        type=str,
        default=None,
        help="Filter by dataset (e.g., 'alpaca', 'starcoder').",
    )
    
    args = parser.parse_args()
    
    # Collect results
    results = collect_results(args.gcs_dir)
    
    if not results:
        print("No results collected.")
        return
    
    # Apply filters
    if args.filter_cpt:
        results = [r for r in results if r.get('model_type') == 'CPT']
    if args.filter_pretrain:
        results = [r for r in results if r.get('model_type') == 'pretrain']
    if args.filter_dataset:
        results = [r for r in results if r.get('dataset') == args.filter_dataset]
    
    if not results:
        print("No results after filtering.")
        return



    
    # Get all unique columns
    all_columns = set()
    for row in results:
        all_columns.update(row.keys())
    
    # Order columns: metadata first, then metrics
    metadata_cols = ['filename', 'Token Budget','LR', 'optimizer', 'model_type', 'dataset']
    metric_cols = sorted([c for c in all_columns if c not in metadata_cols])
    columns = metadata_cols + metric_cols
    
    # Write CSV
    print(f"\nWriting {len(results)} rows to {args.output}...")
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Done! Saved to {args.output}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total rows: {len(results)}")
    print(f"  Columns: {', '.join(columns)}")
    
    # Print sample
    if results:
        print(f"\nSample row:")
        for k, v in list(results[0].items())[:10]:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

