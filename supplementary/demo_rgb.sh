#! /bin/bash

echo "Preparing directory & Python..."

# Get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Move to project level
cd "$SCRIPT_DIR/.." || exit

# Create data dir in project
mkdir -p data

# Create venv and install echolabel
uv venv --clear
uv pip install https://github.com/gaspardringuenet/echolabel.git
source .venv/bin/activate

# Run Python
python -c "
from pathlib import Path
from echolabel.utils.demo import download_demo_data
output_dir = Path('./data')
download_demo_data(output_dir)
"

# Use echolabel CLI to run in RGB mode
echolabel process \
-s data/demo.zarr \
-o data/demo_regions.csv \
-f 38000 -f 70000 -f 120000 \
--vmin -90 --vmax -50 \
--cmap RGB
