#!/bin/bash
set -e

echo "Downloading models from Hugging Face..."
. /opt/venv/bin/activate

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os

zip_path = hf_hub_download(
    repo_id='intellegentJonna/aird-models',
    filename='aird-models.zip',
    repo_type='model'
)
print(f"Downloaded to: {zip_path}")
print(f"File size: {os.path.getsize(zip_path) / (1024**3):.2f} GB")
EOF

echo "Extracting..."
cd /tmp
unzip -q "${zip_path}" || { echo "Unzip failed"; exit 1; }

echo "Copying to /app/models..."
mkdir -p /app/models
cp -r aird-models/* /app/models/

echo "Verifying..."
ls -la /app/models/
echo "Model files:"
find /app/models -name "model.safetensors" -o -name "config.json" | sort
