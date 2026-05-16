#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_ec2.sh — Configuración inicial de instancia EC2 (Ubuntu 22.04)
# Instala dependencias, sube los pesos del modelo y registra el servicio.
#
# Uso (desde tu máquina local):
#   chmod +x setup_ec2.sh
#   ./setup_ec2.sh
#
# Variables que DEBES editar antes de correr:
# ─────────────────────────────────────────────────────────────────────────────

EC2_HOST="ec2-23-20-223-172.compute-1.amazonaws.com"   # ← IP pública o DNS de tu instancia
EC2_USER="ubuntu"
KEY_FILE="./testServer.pem"                        # ← ruta a tu archivo .pem
APP_DIR="/home/ubuntu/spine_api"

# Rutas locales a los pesos (ajustar si están en otro lugar)
WEIGHTS_FULL="./spine_yolo/yolov8s_seg_768/weights/best.pt"
WEIGHTS_T1T8="./spine_yolo/yolov8s_seg_t1t8/weights/best.pt"

# ─────────────────────────────────────────────────────────────────────────────
set -e
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🦴 Spine API — Despliegue en EC2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST"
SCP="scp -i $KEY_FILE -o StrictHostKeyChecking=no"

# ── 1. Subir código ───────────────────────────────────────────────────────────
echo ""
echo "📤 [1/5] Subiendo código fuente..."
$SSH "mkdir -p $APP_DIR/spine_yolo/yolov8s_seg_768/weights \
                $APP_DIR/spine_yolo/yolov8s_seg_t1t8/weights"

$SCP main.py requirements.txt "$EC2_USER@$EC2_HOST:$APP_DIR/"

# ── 2. Subir pesos ────────────────────────────────────────────────────────────
echo ""
echo "📤 [2/5] Subiendo pesos de los modelos (puede tardar varios minutos)..."
$SCP "$WEIGHTS_FULL" "$EC2_USER@$EC2_HOST:$APP_DIR/spine_yolo/yolov8s_seg_768/weights/best.pt"
$SCP "$WEIGHTS_T1T8" "$EC2_USER@$EC2_HOST:$APP_DIR/spine_yolo/yolov8s_seg_t1t8/weights/best.pt"

# ── 3. Instalar dependencias en EC2 ──────────────────────────────────────────
echo ""
echo "🔧 [3/5] Instalando dependencias del sistema..."
$SSH << 'REMOTE'
set -e
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    nginx

# Crear entorno virtual
cd /home/ubuntu/spine_api
python3 -m venv venv
source venv/bin/activate

# Instalar PyTorch CPU primero (más liviano que el default con CUDA)
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
pip install -r requirements.txt --quiet

echo "✅ Dependencias instaladas"
REMOTE

# ── 4. Configurar systemd ─────────────────────────────────────────────────────
echo ""
echo "⚙️  [4/5] Configurando servicio systemd..."
$SSH "sudo tee /etc/systemd/system/spine-api.service > /dev/null" << 'SERVICE'
[Unit]
Description=Spine Segmentation API (FastAPI)
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/spine_api
ExecStart=/home/ubuntu/spine_api/venv/bin/uvicorn main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 2 \
    --timeout-keep-alive 60
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment="OMP_NUM_THREADS=4"
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
SERVICE

$SSH "sudo systemctl daemon-reload && sudo systemctl enable spine-api && sudo systemctl start spine-api"

# ── 5. Configurar nginx como reverse proxy ────────────────────────────────────
echo ""
echo "🌐 [5/5] Configurando Nginx como reverse proxy..."
$SSH "sudo tee /etc/nginx/sites-available/spine-api > /dev/null" << 'NGINX'
server {
    listen 80;
    server_name _;

    client_max_body_size 20M;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }
}
NGINX

$SSH "sudo ln -sf /etc/nginx/sites-available/spine-api /etc/nginx/sites-enabled/spine-api \
      && sudo rm -f /etc/nginx/sites-enabled/default \
      && sudo nginx -t \
      && sudo systemctl restart nginx"

# ── Resultado ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Despliegue completado"
echo ""
echo "  API disponible en:  http://$EC2_HOST"
echo "  Docs Swagger:       http://$EC2_HOST/docs"
echo "  Health check:       http://$EC2_HOST/health"
echo ""
echo "  Comandos útiles en la instancia:"
echo "    sudo systemctl status spine-api"
echo "    sudo journalctl -u spine-api -f"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
