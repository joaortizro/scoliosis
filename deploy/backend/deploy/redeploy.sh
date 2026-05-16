#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# redeploy.sh — Actualizar código en EC2 sin reinstalar dependencias
#
# Uso: ./redeploy.sh
# ─────────────────────────────────────────────────────────────────────────────

EC2_HOST="ec2-23-20-223-172.compute-1.amazonaws.com"   # ← igual que en setup_ec2.sh
EC2_USER="ubuntu"
KEY_FILE="./testServer.pem"
APP_DIR="/home/ubuntu/spine_api"

SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST"
SCP="scp -i $KEY_FILE -o StrictHostKeyChecking=no"

set -e
echo "🔄 Subiendo main.py actualizado..."
$SCP main.py "$EC2_USER@$EC2_HOST:$APP_DIR/"

echo "♻️  Reiniciando servicio..."
$SSH "sudo systemctl restart spine-api"

echo "✅ Actualización completada"
$SSH "sudo systemctl status spine-api --no-pager"
