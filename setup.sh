#! /bin/bash

echo "Preparing Python..."

# Get script directory
SCRIPT_DIR=$(dirname "$0")

# Move to project level
cd "$SCRIPT_DIR" || exit

echo "Building venv in project dir: $(pwd)"

# Create venv and install echolabel
uv venv --clear
uv pip install https://github.com/gaspardringuenet/echolabel.git ipykernel nbformat nbclient
source .venv/bin/activate

# Download demo data
echo "Fetching demo data..."

# Run Python
python -c "
from pathlib import Path
from echolabel.utils.demo import download_demo_data
output_dir = Path('./data')
download_demo_data(output_dir)
"
