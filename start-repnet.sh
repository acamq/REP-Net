#!/usr/bin/env bash
set -euo pipefail

cd /workspace

# Clone or update the repo
if [ ! -d "REP-Net" ]; then
    echo "[start-repnet] Cloning REP-Net..."
    git clone https://github.com/acamq/REP-Net.git
else
    echo "[start-repnet] Updating REP-Net..."
    cd REP-Net
    git pull --rebase
    cd ..
fi

cd /workspace/REP-Net

# Optional: unzip data if needed
if [ -f "data_dir.zip" ] && [ ! -d "data_dir" ]; then
    echo "[start-repnet] Unzipping data_dir.zip..."
    unzip -q data_dir.zip
fi

# Activate prebuilt venv and run
echo "[start-repnet] Activating venv"
. /opt/repnet-venv/bin/activate
echo "[start-repnet] Running REP-Net"
python3 run.py
