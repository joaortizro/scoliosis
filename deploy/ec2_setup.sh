#!/bin/bash
# Run once on a fresh EC2 instance to prepare the environment.
set -e

APP_DIR=/home/ec2-user/scoliosis

# System deps
sudo yum update -y
sudo yum install -y python3.11 python3.11-pip git

# App dirs
mkdir -p $APP_DIR/dist
mkdir -p $APP_DIR/ai/models/checkpoints

# Python venv
python3.11 -m venv ~/.venv
source ~/.venv/bin/activate

# Server deps only (no torch/training deps needed here)
pip install fastapi uvicorn[standard] pydantic python-dotenv dvc dvc-s3

# DVC pull (fetches model checkpoint from remote storage)
cd $APP_DIR
dvc pull ai/models/checkpoints

# Install systemd service
sudo cp deploy/scoliosis-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable scoliosis-server
sudo systemctl start scoliosis-server

echo "EC2 setup complete. Server running on :8000"
