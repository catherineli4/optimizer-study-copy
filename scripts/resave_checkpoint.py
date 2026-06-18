#!/usr/bin/env python3
"""
Script to re-save a checkpoint to be compatible with Python 3.10.

This script loads a checkpoint that was saved with Python 3.13+ and re-saves it
using Python 3.10, fixing the pathlib._local compatibility issue.

Usage:
    # If running with Python 3.13 (to load), it will extract and save to temp, then you need to run with 3.10
    # Or use the wrapper script: resave_checkpoint_wrapper.sh
    python3.13 resave_checkpoint.py <checkpoint_dir> --extract-only
    python3.10 resave_checkpoint.py <checkpoint_dir> --from-temp
"""

import argparse
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import torch
from datetime import datetime


def load_with_python313(filepath: Path, python313_path: str = "/home/catheri4/miniconda3/bin/python3.13"):
    """
    Load a checkpoint file using Python 3.13 (which can read pathlib._local).
    Returns the loaded data.
    """
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        script_path = script_file.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_output_file:
        temp_output_path = temp_output_file.name
    
    # Create a script to load with Python 3.13 and save to temp file
    script_content = f"""
import torch
import sys

filepath = r"{filepath}"
temp_output = r"{temp_output_path}"

try:
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    torch.save(data, temp_output)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    
    try:
        # Write the script
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Run the script with Python 3.13
        result = subprocess.run(
            [python313_path, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "SUCCESS" not in result.stdout:
            raise RuntimeError(f"Python 3.13 script did not complete successfully: {result.stderr}")
        
        # Load from the temporary file
        data = torch.load(temp_output_path, map_location='cpu', weights_only=False)
        
        return data
        
    except subprocess.CalledProcessError as e:
        print(f"Error loading with Python 3.13: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: Python 3.13 not found at {python313_path}")
        print("Please install Python 3.13 or specify the path with --python313-path")
        raise
    finally:
        # Clean up temporary files
        for p in [script_path, temp_output_path]:
            if Path(p).exists():
                try:
                    Path(p).unlink()
                except:
                    pass


def resave_checkpoint_file(checkpoint_dir: Path, filename: str, backup: bool = True, 
                          use_python313: bool = False, python313_path: str = None):
    """
    Re-save a checkpoint file to be compatible with Python 3.10.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        filename: Name of the checkpoint file to re-save (e.g., 'train.pt', 'optim.pt')
        backup: Whether to create a backup of the original file
        use_python313: If True, use Python 3.13 to load the file first
        python313_path: Path to Python 3.13 executable
    """
    filepath = checkpoint_dir / filename
    
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist, skipping...")
        return False
    
    print(f"Processing {filepath}...")
    
    # Create backup if requested
    if backup:
        backup_path = filepath.with_suffix(f".pt.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"Creating backup: {backup_path}")
        shutil.copy2(filepath, backup_path)
    
    try:
        # Load the checkpoint
        print(f"Loading {filepath}...")
        
        if use_python313:
            if python313_path is None:
                python313_path = "/home/catheri4/miniconda3/bin/python3.13"
            print(f"Using Python 3.13 to load (from {python313_path})...")
            data = load_with_python313(filepath, python313_path)
        else:
            data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        print(f"Successfully loaded {filepath}")
        
        # Re-save it (this will re-pickle with current Python version)
        print(f"Re-saving {filepath} with Python {sys.version_info.major}.{sys.version_info.minor}...")
        torch.save(data, filepath)
        print(f"Successfully re-saved {filepath}")
        return True
        
    except ModuleNotFoundError as e:
        if "pathlib._local" in str(e) or "pathlib" in str(e).lower():
            print(f"\nError: Cannot load checkpoint with Python {sys.version_info.major}.{sys.version_info.minor}")
            print("This checkpoint was saved with Python 3.13+ and needs to be loaded with Python 3.13 first.")
            print("\nTry one of these solutions:")
            print("1. Run this script with Python 3.13 first, then with Python 3.10")
            print("2. Use the --use-python313 flag (requires Python 3.13 to be available)")
            print("3. Use the wrapper script: resave_checkpoint_wrapper.sh")
            return False
        else:
            raise
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Re-save checkpoint files to be compatible with Python 3.10"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the checkpoint directory (e.g., final-unsharded)"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=["train.pt", "optim.pt", "model.pt"],
        help="Checkpoint files to re-save (default: train.pt optim.pt model.pt)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Re-save all .pt files in the directory"
    )
    parser.add_argument(
        "--use-python313",
        action="store_true",
        help="Use Python 3.13 to load files (for pathlib._local compatibility)"
    )
    parser.add_argument(
        "--python313-path",
        type=str,
        default="/home/catheri4/miniconda3/bin/python3.13",
        help="Path to Python 3.13 executable"
    )
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
        sys.exit(1)
    
    if not checkpoint_dir.is_dir():
        print(f"Error: Path is not a directory: {checkpoint_dir}")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Backup enabled: {not args.no_backup}")
    print()
    
    # Determine which files to process
    if args.all_files:
        files_to_process = list(checkpoint_dir.glob("*.pt"))
        files_to_process = [f.name for f in files_to_process]
        print(f"Found {len(files_to_process)} .pt files to process")
    else:
        files_to_process = args.files
    
    # Process each file
    success_count = 0
    for filename in files_to_process:
        if resave_checkpoint_file(
            checkpoint_dir, 
            filename, 
            backup=not args.no_backup,
            use_python313=args.use_python313,
            python313_path=args.python313_path
        ):
            success_count += 1
        print()
    
    print(f"Successfully processed {success_count}/{len(files_to_process)} files")
    
    if success_count < len(files_to_process):
        sys.exit(1)


if __name__ == "__main__":
    main()

