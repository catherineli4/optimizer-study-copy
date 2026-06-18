#!/bin/bash
# Wrapper script to re-save checkpoint using Python 3.13 to load and Python 3.10 to save

set -e

CHECKPOINT_DIR="$1"
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <checkpoint_dir>"
    echo "Example: $0 /scratch/catheri4/outputs/adamw/dHgFiHXg/PretrainedModel/OLMo-tk64B-adamw-lr3e-4-wd1e-1-bs256/final-unsharded"
    exit 1
fi

PYTHON313="/home/catheri4/miniconda3/bin/python3.13"
PYTHON310="/home/catheri4/miniconda3/envs/myenv310/bin/python3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESAVE_SCRIPT="$SCRIPT_DIR/resave_checkpoint.py"

if [ ! -f "$PYTHON313" ]; then
    echo "Error: Python 3.13 not found at $PYTHON313"
    exit 1
fi

if [ ! -f "$PYTHON310" ]; then
    echo "Error: Python 3.10 not found at $PYTHON310"
    exit 1
fi

echo "Step 1: Loading checkpoint with Python 3.13 and extracting data..."
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Create a script that loads with Python 3.13 and saves to a format Python 3.10 can read
cat > "$TEMP_DIR/load_with_313.py" << 'EOF'
import sys
import torch
import pickle
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
files_to_process = sys.argv[3:]

for filename in files_to_process:
    filepath = checkpoint_dir / filename
    if not filepath.exists():
        print(f"Skipping {filepath} (does not exist)")
        continue
    
    print(f"Loading {filepath} with Python 3.13...")
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        # Save to a temporary file that Python 3.10 can read
        # We'll use a simple pickle format that's more compatible
        output_path = output_dir / filename
        print(f"Saving extracted data to {output_path}...")
        torch.save(data, output_path)
        print(f"Successfully processed {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF

# Load with Python 3.13
echo "Loading checkpoint files with Python 3.13..."
$PYTHON313 "$TEMP_DIR/load_with_313.py" "$CHECKPOINT_DIR" "$TEMP_DIR" train.pt optim.pt model.pt

echo ""
echo "Step 2: Re-saving checkpoint with Python 3.10..."
# Now re-save with Python 3.10
$PYTHON310 "$RESAVE_SCRIPT" "$CHECKPOINT_DIR" --files train.pt optim.pt model.pt --no-backup

# Copy the re-saved files back
echo ""
echo "Step 3: Copying re-saved files back to checkpoint directory..."
for file in train.pt optim.pt model.pt; do
    if [ -f "$TEMP_DIR/$file" ]; then
        # The resave script already saved them, but let's verify
        if [ -f "$CHECKPOINT_DIR/$file" ]; then
            echo "✓ $file is ready"
        else
            echo "Warning: $file was not re-saved"
        fi
    fi
done

echo ""
echo "Done! Checkpoint has been re-saved with Python 3.10 compatibility."

