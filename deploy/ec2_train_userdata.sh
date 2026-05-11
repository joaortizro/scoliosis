#!/bin/bash
# EC2 user-data script for g4dn.xlarge training instance (Amazon Linux 2023).
# Installs NVIDIA Tesla driver + Python 3.11 venv + repo + DVC data.
# Idempotent. Logs to /var/log/user-data.log.
set -eux
exec > >(tee -a /var/log/user-data.log) 2>&1

REPO_URL=https://github.com/joaortizro/scoliosis.git
BRANCH=feature/epoch-resume
APP_DIR=/home/ec2-user/scoliosis
VENV=/home/ec2-user/.venv

# ── 1. System packages ────────────────────────────────────────────────────────
dnf update -y
dnf install -y git tmux htop unzip jq \
  python3.11 python3.11-pip python3.11-devel \
  gcc make dkms kernel-devel kernel-modules-extra

# ── 2. NVIDIA Tesla driver for T4 GPU ─────────────────────────────────────────
# AWS-hosted installer (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  dnf install -y awscli
  cd /tmp
  aws s3 cp --no-sign-request \
    s3://ec2-linux-nvidia-drivers/latest/ . --recursive --exclude "*" --include "NVIDIA-Linux-x86_64*.run" || \
    aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ . --exclude "*" --include "NVIDIA-Linux-x86_64*.run"
  chmod +x NVIDIA-Linux-x86_64*.run
  ./NVIDIA-Linux-x86_64*.run --silent --dkms
fi

# ── 3. Repo + Python env (as ec2-user) ────────────────────────────────────────
sudo -u ec2-user -H bash <<EOF
set -eux
cd /home/ec2-user
if [ ! -d "$APP_DIR/.git" ]; then
  git clone -b $BRANCH $REPO_URL $APP_DIR
fi
cd $APP_DIR
git fetch origin
git checkout $BRANCH
git pull --ff-only

python3.11 -m venv $VENV
source $VENV/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
pip install -e .
EOF

# ── 4. DVC pull (uses IAM instance profile creds via boto3) ───────────────────
sudo -u ec2-user -H bash <<EOF
set -eux
cd $APP_DIR
source $VENV/bin/activate
dvc pull data/raw
EOF

echo "EC2 setup complete. SSH in, activate $VENV, run training."
