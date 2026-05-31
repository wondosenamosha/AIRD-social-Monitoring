#!/bin/bash

echo "=== Checking tools ==="
which python3 unzip || echo "WARNING: Missing tools"

echo "=== Downloading models from HuggingFace ==="
. /opt/venv/bin/activate

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import zipfile
import os
import shutil

try:
    zip_path = hf_hub_download(
        repo_id='intellegentJonna/aird-models',
        filename='aird-models.zip',
        repo_type='model'
    )
    print(f"✓ Downloaded: {zip_path}")

    # Extract using Python's zipfile module
    print("✓ Extracting ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('/tmp')

    # Copy to app directory
    print("✓ Copying to /app/models...")
    os.makedirs('/app/models', exist_ok=True)

    src = '/tmp/aird-models'
    if os.path.exists(src):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join('/app/models', item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
                print(f"  - {item}/")

    print("✓ DONE: Models ready")

except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

echo "=== Verifying ==="
ls -la /app/models/ 2>/dev/null || echo "ERROR: /app/models does not exist"
find /app/models -name "model.safetensors" 2>/dev/null | wc -l | xargs echo "Found safetensors files:"
