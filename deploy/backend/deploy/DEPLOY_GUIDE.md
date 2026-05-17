# 🚀 Guía de despliegue — Spine API en AWS EC2

## Paso 1: Crear la instancia EC2

### Desde la consola AWS (https://console.aws.amazon.com/ec2)

**AMI:** Ubuntu Server 22.04 LTS (ami-0c7217cdde317cfec en us-east-1)
**Tipo de instancia:** `c6i.xlarge` (4 vCPU, 8 GB RAM) — recomendado
- Alternativa económica: `t3.large` (2 vCPU, 8 GB) para pruebas (~$60/mes)
- Alternativa potente: `c6i.2xlarge` (8 vCPU, 16 GB) para producción alta carga

**Almacenamiento:** 30 GB gp3 (suficiente para OS + pesos + logs)

**Security Group — reglas de entrada:**
| Tipo  | Puerto | Origen         | Descripción          |
|-------|--------|----------------|----------------------|
| SSH   | 22     | Tu IP /32      | Acceso administración|
| HTTP  | 80     | 0.0.0.0/0      | API pública          |

> ⚠️ No abras el puerto 8000 directamente — Nginx actúa de proxy en el 80.

**Key pair:** Crea o usa uno existente `.pem` y guárdalo en `~/.ssh/`

```bash
chmod 400 ~/.ssh/tu-clave.pem
```

---

## Paso 2: Editar las variables del script

Abre `setup_ec2.sh` y edita las 3 primeras variables:

```bash
EC2_HOST="ec2-54-123-45-67.compute-1.amazonaws.com"  # DNS público de tu instancia
EC2_USER="ubuntu"
KEY_FILE="~/.ssh/tu-clave.pem"
```

También edita las rutas a tus pesos locales si son diferentes:
```bash
WEIGHTS_FULL="./spine_yolo/yolov8s_seg_768/weights/best.pt"
WEIGHTS_T1T8="./spine_yolo/yolov8s_seg_t1t8/weights/best.pt"
```

---

## Paso 3: Correr el setup

Desde la carpeta donde tienes `main.py`:

```bash
chmod +x deploy/setup_ec2.sh
./deploy/setup_ec2.sh
```

El script hace todo automáticamente:
1. Sube `main.py` y `requirements.txt`
2. Sube los pesos de ambos modelos
3. Instala Python, venv, PyTorch CPU y dependencias
4. Registra y arranca el servicio con `systemd`
5. Configura Nginx como reverse proxy en el puerto 80

Tiempo estimado: **10-15 minutos** (principalmente por la subida de pesos y la instalación de PyTorch).

---

## Paso 4: Verificar el despliegue

```bash
# Health check
curl http://TU_IP_PUBLICA/health

# Probar segmentación
curl -X POST "http://TU_IP_PUBLICA/segment" \
     -F "file=@radiografia.jpg" \
     | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
print(f'Detectadas: {d[\"total_detected\"]} vertebras')
for v in d['vertebrae']:
    print(f'  {v[\"label\"]}: {v[\"confidence\"]:.3f}')
open('resultado.png','wb').write(base64.b64decode(d['image_base64']))
"

# Swagger UI
open http://TU_IP_PUBLICA/docs
```

---

## Comandos de administración

```bash
# Conectarse a la instancia
ssh -i ~/.ssh/tu-clave.pem ubuntu@TU_IP_PUBLICA

# Estado del servicio
sudo systemctl status spine-api

# Logs en tiempo real
sudo journalctl -u spine-api -f

# Reiniciar servicio
sudo systemctl restart spine-api

# Ver logs de Nginx
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## Actualizar código (sin reinstalar)

Cuando modifiques `main.py`:

```bash
# Editar redeploy.sh con los mismos datos de conexión
./deploy/redeploy.sh
```

---

## Costos estimados (us-east-1)

| Instancia     | vCPU | RAM   | $/hora  | $/mes (~730h) |
|---------------|------|-------|---------|---------------|
| t3.large      | 2    | 8 GB  | $0.083  | ~$61          |
| c6i.xlarge    | 4    | 8 GB  | $0.170  | ~$124         |
| c6i.2xlarge   | 8    | 16 GB | $0.340  | ~$248         |

> 💡 Para reducir costos: usa **Spot Instances** (hasta 70% descuento) si la API puede tolerar interrupciones ocasionales, o Reserved Instances (1 año) para descuentos fijos.

---

## Tiempos de inferencia esperados (CPU)

| Instancia   | Inferencia/imagen |
|-------------|-------------------|
| t3.large    | ~3-5 segundos     |
| c6i.xlarge  | ~1.5-2.5 segundos |
| c6i.2xlarge | ~0.8-1.5 segundos |

> YOLOv8s es relativamente liviano; con `OMP_NUM_THREADS=4` usa todos los cores disponibles.

---

## HTTPS (opcional pero recomendado para producción)

Si tienes un dominio apuntando a la instancia:

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d tu-dominio.com
```

Certbot configura el certificado SSL y renueva automáticamente cada 90 días.
