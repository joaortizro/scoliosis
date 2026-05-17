# 🦴 Spine Segmentation API

FastAPI con ensemble YOLOv8s-seg para segmentación de vértebras en radiografías de columna vertebral.

## Arquitectura del ensemble

```
Imagen (512×512)
       │
       ├─ Crop superior (42%) ──► Modelo T1-T8 ──► T1…T8
       │
       └─ Imagen completa ──────► Modelo completo ──► vértebras con centroide Y > 42%
                                                              (T8-T12, L1-L5)
                                   Resultado fusionado (orden craniocaudal)
```

## Estructura de archivos

```
spine_api/
├── main.py           ← API FastAPI
├── requirements.txt
├── Dockerfile
├── test_api.py       ← script de prueba
└── spine_yolo/       ← pesos de los modelos (crear/copiar aquí)
    ├── yolov8s_seg_768/
    │   └── weights/
    │       └── best.pt   ← modelo completo
    └── yolov8s_seg_t1t8/
        └── weights/
            └── best.pt   ← modelo T1-T8
```

## Instalación y arranque

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Con Docker:
```bash
docker build -t spine-api .
docker run -p 8000:8000 -v $(pwd)/spine_yolo:/app/spine_yolo spine-api
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Estado y modelos cargados |
| GET | `/health` | Health check detallado |
| POST | `/segment` | **Ensemble** (recomendado) |
| POST | `/segment/full-only` | Solo modelo completo T1-L5 |
| POST | `/segment/t1t8-only` | Solo modelo especializado T1-T8 |

### Parámetros comunes (query string)

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `conf` | float | 0.01 | Umbral de confianza YOLO |
| `return_image` | bool | true | Incluir PNG segmentado en base64 |

### Ejemplo con curl

```bash
# Ensemble (recomendado)
curl -X POST "http://localhost:8000/segment?conf=0.01" \
     -F "file=@radiografia.jpg" \
     | python -c "
import sys, json, base64
d = json.load(sys.stdin)
print(f'Detectadas: {d[\"total_detected\"]} vértebras')
for v in d['vertebrae']:
    print(f'  {v[\"label\"]}: conf={v[\"confidence\"]:.3f}')
open('resultado.png','wb').write(base64.b64decode(d['image_base64']))
"
```

### Ejemplo con Python (requests)

```python
import requests, base64

with open("radiografia.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/segment",
        files={"file": f},
        params={"conf": 0.01}
    )

data = r.json()
print(f"Detectadas: {data['total_detected']} vértebras")
for v in data["vertebrae"]:
    print(f"  {v['label']}: confianza={v['confidence']:.3f}, fuente={v['source']}")

# Guardar imagen segmentada
with open("segmentada.png", "wb") as f:
    f.write(base64.b64decode(data["image_base64"]))
```

## Esquema de respuesta

```json
{
  "vertebrae": [
    {
      "label": "T1",
      "confidence": 0.8231,
      "centroid_x": 256.3,
      "centroid_y": 87.1,
      "area_px": 312,
      "source": "t1t8_model"
    },
    {
      "label": "L3",
      "confidence": 0.9105,
      "centroid_x": 251.8,
      "centroid_y": 380.4,
      "area_px": 489,
      "source": "full_model"
    }
  ],
  "total_detected": 15,
  "image_base64": "<PNG en base64>",
  "model_used": "ensemble",
  "device": "cuda"
}
```

## Docs interactivos

Con la API corriendo, visitar:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
