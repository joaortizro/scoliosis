# Detección de Escoliosis con IA

> **Título académico (artículo / tesis):** *Segmentación multiclase de columna y vértebras T1–L5 en radiografías de escoliosis con robustez a campo de visión parcial.*

Sistema de detección de escoliosis basado en *deep learning* sobre radiografías de columna. Incluye:

- Pipeline reproducible de preprocesamiento, entrenamiento y evaluación (DVC).
- Backend FastAPI que expone el modelo entrenado como API REST.
- Frontend Next.js para que un usuario final cargue una radiografía y obtenga: máscara segmentada por vértebra (T1–L5), ángulo Cobb estimado y clasificación de severidad (*normal / mild / moderate / severe*).
- Backend secundario DeepLab para comparación de arquitecturas.

---

## Integrantes

- Beto Javi
- Fedys Moreno
- Jonathan Ortiz
- Jorge Oñate
- Milton Rentería

**Curso:** Maestría en Inteligencia Artificial (MaIA).

---

## Enlaces

| Recurso | Enlace |
|---|---|
| Repositorio GitHub | https://github.com/joaortizro/scoliosis |
| **Aplicación desplegada** | **<https://miro.lat/>** |
| Artículo académico (LaTeX) | `docs/thesis/scoliosis.tex` |

**Credenciales demo:** usuario `admin` · contraseña `demo123`

---

## Mapeo a la rúbrica (carpetas obligatorias)

La rúbrica pide tres carpetas (`Notebooks/`, `Modelos/`, `Datos/`). En este repositorio mantenemos los nombres reales usados durante el desarrollo, con esta correspondencia explícita:

| Rúbrica | Ruta real en el repo | Contenido |
|---|---|---|
| `Notebooks/` | [`notebooks/`](./notebooks/) | Cuadernos de análisis, entrenamiento y evaluación (tres niveles: `sandbox/`, `experiments/`, `final/`). Ver [Guía de notebooks](#guía-de-notebooks). |
| `Modelos/` | [`ai/models/releases/`](./ai/models/releases/) | Pesos del modelo final (`.pt`) versionados con DVC. Pointer `.dvc` en Git, pesos en el remoto. Carpeta canónica: `phase1_chain_2026-05-08_ec2_t4_v2_corrected/` (7 checkpoints + README con métricas y cfg de reproducción). |
| `Datos/` | [`data/`](./data/) | Estructura del dataset MaIA Scoliosis v2 (250 radiografías) más índices CSV y subset auditado. Las imágenes son confidenciales — solo pointers `.dvc` viven en Git; los píxeles se obtienen vía `dvc pull`. |

> **Nota ética:** las imágenes radiográficas son material clínico anonimizado. Por acuerdo institucional con el Grupo de Ingeniería Biomédica de la Universidad de los Andes, **no se publican en Git**. Se distribuyen vía DVC a un remoto S3 con acceso controlado. La estructura, los índices y las máscaras de etiquetado están documentados en [`data/raw/Scoliosis_Dataset_v2/`](./data/raw/) y en [`.claude/CLAUDE.md`](./.claude/CLAUDE.md).

---

## Recorrido rápido para evaluadores (5 minutos)

Si solo tienes 5 minutos para revisar el proyecto, esta es la ruta sugerida:

1. **App desplegada** → abrir <https://miro.lat/>, autenticarse con `admin` / `demo123` y subir cualquier radiografía AP de columna. La UI muestra segmentación + Cobb + severidad.
2. **Modelo final** → [`ai/models/releases/phase1_chain_2026-05-08_ec2_t4_v2_corrected/README.md`](./ai/models/releases/) — tabla con 7 checkpoints, métricas, cfg, entorno EC2.
3. **Notebook de entrenamiento canónico** → [`notebooks/sandbox/model_primer_v3_corrected.ipynb`](./notebooks/sandbox/model_primer_v3_corrected.ipynb) — pipeline reproducible del modelo headline (val Dice 0.643).
4. **Notebook de exploración del dataset** → [`notebooks/sandbox/data_exploration_v2_corrected.ipynb`](./notebooks/sandbox/data_exploration_v2_corrected.ipynb) — EDA + auditoría.
5. **Backend** → [`server/main.py`](./server/main.py) y [`server/api/predict.py`](./server/api/predict.py).
6. **Frontend** → [`frontend/src/app/prediction/`](./frontend/src/app/prediction/) — flujo de subida + visualización.
7. **Pipeline DVC** → [`dvc.yaml`](./dvc.yaml) + [`params.yaml`](./params.yaml).
8. **Artículo académico** → [`docs/thesis/scoliosis.tex`](./docs/thesis/) — formato IEEE.

---

## Stack

| Capa | Tecnología |
|---|---|
| ML | PyTorch · segmentation-models-pytorch |
| Tracking | DVC + JSON metrics (`experiments/results/*.json`) |
| Backend | FastAPI + Uvicorn |
| Backend secundario | FastAPI (DeepLab) |
| Frontend | Next.js 15 + React + Tailwind |
| Empaquetado | wheel `scoliosis-ai` |
| Entornos | tox |
| Contenedores | Docker + Docker Compose |
| Registro de imágenes | AWS ECR |
| Despliegue | AWS ECS Fargate / EC2 + systemd |
| CI | GitHub Actions |

---

## Arquitectura

```
┌─────────────────┐    HTTPS     ┌─────────────────┐
│  Frontend Web   │ ───────────▶ │ FastAPI Backend │
│  (Next.js)      │              │ (server/)       │
└─────────────────┘              └────────┬────────┘
                                          │
                                          ▼
                                 ┌─────────────────┐
                                 │  ai/ (PyTorch)  │
                                 │  EncoderUNet    │
                                 │  ResNet34 + CE  │
                                 │  + Dice + SDHL  │
                                 └─────────────────┘
```

El servidor sigue **Arquitectura Hexagonal** (puertos y adaptadores). El paquete `ai/` es una librería ML pura, importada solo desde adaptadores de infraestructura. Adicionalmente, un backend secundario DeepLabV3+ vive en `deploy/deeplab_api/` para experimentos de comparación de arquitecturas (ver sección [DeepLab API Backend](#deeplab-api-backend)).

---

## Requisitos del entorno

| Componente | Versión mínima |
|---|---|
| Python | 3.11 |
| Node.js | 20 LTS |
| Docker | 24.x |
| DVC | 3.x |
| GPU | Opcional (CPU funciona; CUDA/ROCm acelera entrenamiento) |
| RAM | 8 GB (16 GB recomendado) |
| Disco | 5 GB para imágenes Docker + dataset |

---

## Instalación y ejecución local

### 1. Clonar y preparar Python

```bash
git clone https://github.com/joaortizro/scoliosis.git
cd scoliosis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Variables de entorno

Copiar la plantilla y editar valores:

```bash
cp .env.example .env
```

Variables principales (`.env`):

```dotenv
# Backend
MODEL_CHECKPOINTS_DIR=ai/models/checkpoints
MODEL_RELEASE=phase1_chain_2026-05-08_ec2_t4_v2_corrected
PORT=8001

# Auth (frontend)
AUTH_USER=admin
AUTH_PASSWORD=demo123

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8001
```

### 3. Descargar dataset y modelos (DVC)

```bash
dvc pull        # baja data/raw/ y ai/models/releases/ desde el remoto S3
```

### 4. Levantar backend

```bash
tox -e run                       # uvicorn server.main:app --reload --port 8001
# o vía Docker:
docker compose up server
```

Documentación interactiva: <http://localhost:8001/docs>

### 5. Levantar frontend

```bash
cd frontend
npm install
npm run dev                      # http://localhost:3000
```

---

## Credenciales de ejemplo

Para acceder a la aplicación desplegada en <https://miro.lat/>:

| Campo | Valor demo |
|---|---|
| Usuario | `admin` |
| Contraseña | `demo123` |

> Estas mismas credenciales sirven para ejecutar la app localmente (ver `.env.example`).

---

## Ejemplos de uso

### Vía interfaz web

1. Abrir <https://miro.lat/> (o <http://localhost:3000> en local).
2. Autenticarse con las credenciales demo.
3. Subir una radiografía AP/PA de columna (JPG/PNG).
4. La interfaz muestra:
   - Imagen original.
   - *Overlay* con segmentación por vértebra (T1–L5).
   - Ángulo Cobb estimado en grados.
   - Clasificación de severidad: `normal` (<10°), `mild` (10–25°), `moderate` (25–40°), `severe` (>40°).
   - *Score* de confianza.

### Vía API REST (cURL)

```bash
# Health check
curl http://localhost:8001/health

# Predicción
curl -X POST http://localhost:8001/predict/ \
  -F "file=@data/raw/Scoliosis_Dataset_v2/Scoliosis/S_001.jpg" \
  | jq
```

Respuesta esperada (resumida):

```json
{
  "diagnosis_id": "a1b2c3d4",
  "cobb_angle_deg": 27.4,
  "severity": "moderate",
  "vertebrae_detected": 17,
  "mask_url": "/static/masks/a1b2c3d4.png",
  "inference_ms": 312
}
```

### Vía CLI de Python (inferencia directa)

```python
from ai import Predictor

p = Predictor.from_release("phase1_chain_2026-05-08_ec2_t4_v2_corrected")
result = p.predict("data/raw/Scoliosis_Dataset_v2/Scoliosis/S_001.jpg")
print(result.cobb_angle_deg, result.severity)
```

---

## Parametrización

Toda la configuración se centraliza en dos lugares:

### 1. `params.yaml` — Hiperparámetros del modelo

Fuente única de verdad para entrenamiento y evaluación. Editar aquí para reentrenar:

```yaml
train:
  encoder_name: resnet34
  batch_size: 4
  epochs: 100
  lr_enc: 1.0e-4
  lr_dec: 1.0e-3
  augment: v4
  ema:
    enabled: true
    decay: 0.999
  loss:
    boundary_lambda: 0.10
```

### 2. `.env` — Configuración de runtime

Variables que cambian entre entornos (local / staging / producción). Documentadas en `.env.example`.

### 3. `Makefile` — Variables de despliegue

```bash
make deploy-ecs AWS_ACCOUNT=123456789012 AWS_REGION=us-east-1
```

Variables soportadas: `EC2_HOST`, `PEM`, `AWS_ACCOUNT`, `AWS_REGION`, `ECR_REPO`, `ECS_CLUSTER`, `ECS_SERVICE`.

---

## Pipeline reproducible (DVC)

```bash
make pipeline      # preprocess → train → evaluate
make preprocess
make train
make evaluate
```

Comparar corridas:

```bash
dvc params diff
dvc metrics diff
```

Las métricas finales se guardan en `experiments/results/*.json` y son diffeables en Git.

---

## Pruebas

```bash
make test                  # tests de paquete
tox -e test_api            # tests de API
pytest tests/ -v           # todos
```

Incluye:
- Tests unitarios de dominio.
- Tests de adaptadores REST con `TestClient`.
- *Fitness function* arquitectónico que valida que el dominio no importe FastAPI / SQLAlchemy / PyTorch.
- Test de no-fuga (`tests/test_no_leakage.py`) que congela el slice de test (25 casos) con agrupación por paciente y estratificación por severidad.

---

## API REST

### Backend principal (`server/`)

EncoderUNet ResNet-34 + EMA, expone:

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict/` | Sube una imagen, devuelve segmentación + Cobb + severidad |

Docs OpenAPI interactivos en <http://localhost:8001/docs>.

### Backend secundario DeepLabV3+ (`deploy/deeplab_api/`)

Servicio independiente que aloja la arquitectura alterna DeepLabV3+ (binaria + multiclase). Se mantiene separado del backend principal para experimentos de comparación de arquitecturas y para alojamiento independiente (Hugging Face Spaces).

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Health check + estado del modelo cargado |
| POST | `/predict-binary` | Segmentación binaria espina vs fondo |
| POST | `/predict-multiclass` | Segmentación multiclase T1–L5 |
| POST | `/predict-full` | Respuesta combinada binaria + multiclase |

Los `.pth` deben estar presentes en `deploy/deeplab_api/models/` en runtime. No se comitean a Git por tamaño y por convenio institucional.

---

## Despliegue

### Opción A — EC2 + systemd (rápido, etapa temprana)

```bash
# Primera vez en EC2:
scp -i ~/.ssh/key.pem -r server/ deploy/ .env ec2-user@<ip>:~/scoliosis/
ssh -i ~/.ssh/key.pem ec2-user@<ip> "bash ~/scoliosis/deploy/ec2_setup.sh"

# Subsecuentes:
make deploy EC2_HOST=ec2-user@<ip> PEM=~/.ssh/key.pem
```

### Opción B — ECS Fargate + ECR (producción, zero-downtime)

```bash
# Despliegue completo: push checkpoint + build image + push ECR + redeploy service
make deploy-ecs AWS_ACCOUNT=<id> AWS_REGION=us-east-1

# O paso a paso:
make push-data            # dvc push → S3
make docker-push          # build image → push a ECR
make ecs-deploy           # aws ecs update-service (rolling update)
```

### Opción C — Frontend en Vercel

```bash
cd frontend
vercel --prod
```

Configurar `NEXT_PUBLIC_API_URL` apuntando al backend desplegado.

### Variables del Makefile

| Variable | Default | Descripción |
|---|---|---|
| `EC2_HOST` | `ec2-user@<ip>` | Target SSH del EC2 |
| `PEM` | `~/.ssh/your-key.pem` | Ruta de la llave SSH |
| `AWS_ACCOUNT` | `<id>` | Número de cuenta AWS |
| `AWS_REGION` | `us-east-1` | Región AWS |
| `ECR_REPO` | `scoliosis-api` | Repositorio ECR |
| `ECS_CLUSTER` | `scoliosis-cluster` | Cluster ECS |
| `ECS_SERVICE` | `scoliosis-service` | Servicio ECS |

---

## Guía de notebooks

El directorio [`notebooks/`](./notebooks/) sigue un flujo de tres niveles (sandbox → experiments → final) heredado de la convención de la maestría. Convenciones completas en [`notebooks/README.md`](./notebooks/README.md).

> **Honestidad:** este es un proyecto de investigación activo. La carpeta `sandbox/` contiene ~30 notebooks de exploración — es el laboratorio del equipo, no la versión limpia. Para revisión rápida usa los notebooks canónicos listados abajo.

### Estructura

| Carpeta | Propósito | Reglas | Estado |
|---|---|---|---|
| [`sandbox/`](./notebooks/sandbox/) | Scratchpad por integrante | Sin reglas — experimentación libre | ~30 notebooks (EDA, model primers, multitask, yolov8…) |
| [`experiments/`](./notebooks/experiments/) | Ideas curadas por tópico | Cell de Conclusiones obligatorio | `audit/` y `vertebra-segmentation/` con contenido; otros tópicos pendientes |
| [`final/`](./notebooks/final/) | Tesis-ready, numerados | Reproducibles top-to-bottom | Vacío — pipeline reproducible vive en `scripts/` |

### Notebooks canónicos (lo que un evaluador debe abrir)

| Notebook | Para qué sirve |
|---|---|
| [`sandbox/data_exploration_v2_corrected.ipynb`](./notebooks/sandbox/data_exploration_v2_corrected.ipynb) | EDA + auditoría del dataset v2 corregido (152 casos entrenables) |
| [`sandbox/model_primer_v3_corrected.ipynb`](./notebooks/sandbox/model_primer_v3_corrected.ipynb) | **Modelo headline single-task** — EncoderUNet(ResNet-34) + augment_v4, val Dice 0.643 |
| [`sandbox/multitask_v4_5fold_cv.ipynb`](./notebooks/sandbox/multitask_v4_5fold_cv.ipynb) | Validación cruzada 5-fold del modelo multitarea |
| [`experiments/audit/audit_findings.ipynb`](./notebooks/experiments/audit/audit_findings.ipynb) | Reporte formal de la auditoría del dataset |
| [`experiments/vertebra-segmentation/vertebrae_multiclass_deeplabv3plus_5fold.ipynb`](./notebooks/experiments/vertebra-segmentation/vertebrae_multiclass_deeplabv3plus_5fold.ipynb) | Arquitectura alterna DeepLabV3+ multiclase (segundo backend) |
| [`experiments/vertebra-segmentation/vertebra_yolov8_segmentation.ipynb`](./notebooks/experiments/vertebra-segmentation/vertebra_yolov8_segmentation.ipynb) | Experimento de baseline con YOLOv8-seg |

### Por qué `sandbox/` está "desordenado"

Es deliberado. El equipo trabaja con la convención **"sandbox = pensamiento en voz alta"**. Cada model primer (`model_primer.ipynb`, `model_primer_v2.ipynb`, `model_primer_v3.ipynb`, `model_primer_v3_corrected.ipynb`) representa una iteración cronológica documentada, no archivos olvidados. La cadena `multitask_v1` → `multitask_v7` captura las decisiones de diseño de la pérdida multitarea (segmentación + Cobb regression + clasificación).

La versión "limpia y reproducible" de cada experimento exitoso vive como **script** en [`scripts/`](./scripts/) (`phase0_ablations.py`, `phase1_2_d1_roi.py`, `cv5_train.py`, etc.) — esos son los entrypoints DVC que producen los pesos en `ai/models/releases/`.

### Flujo recomendado para reproducir un experimento

1. Abrir notebook canónico → entender la idea y los hiperparámetros.
2. Revisar el script equivalente en `scripts/` → versión productiva.
3. Editar `params.yaml` con los hiperparámetros del experimento.
4. Ejecutar `dvc repro train` o el script directo.
5. Verificar resultados contra `experiments/results/*.json`.

---

## Pipeline y scripts

[`scripts/`](./scripts/) contiene los entrypoints DVC y las versiones productivas de los experimentos. Agrupados:

### Construcción del dataset
| Script | Función |
|---|---|
| `preprocess.py` | Stage DVC `preprocess` — genera índices + auditoría |
| `build_corrected_dataset.py` | Genera `Scoliosis_Dataset_v2_corrected` con máscaras corregidas |
| `build_case_summaries.py` | Tarjetas resumen por caso (PNG con overlay + métricas) |
| `build_ablation_indices.py` | Índices para ablaciones de subset |

### Entrenamiento
| Script | Función |
|---|---|
| `train.py` | Stage DVC `train` — entrenamiento canónico |
| `cv5_train.py` | Validación cruzada 5-fold |
| `phase0_ablations.py` | Ablación de CLAHE × boundary λ (cadena Fase 0) |
| `phase1_2_d1_roi.py` | Recipe headline con ROI crop (val Dice 0.674) |
| `phase1_3a_cutmix.py` | Experimento CutMix |
| `train_partial_fov.py` | Entrenamiento con campo de visión parcial |

### Evaluación
| Script | Función |
|---|---|
| `evaluate.py` | Stage DVC `evaluate` — métricas finales |
| `eval_test.py` | Evaluación en el slice sellado de test |
| `eval_partial_fov.py` | Robustez a FOV parcial |
| `eval_cobb_on_5fold.py` | Cobb angle MAE en CV 5-fold |
| `eval_models_on_roboflow.py` | Transferencia a dataset externo (Roboflow) |

### Visualización / figuras
| Script | Función |
|---|---|
| `render_paper_fig2_gallery.py` | Galería para el paper |
| `render_partial_fov_demo.py` | Demo de FOV parcial |
| `render_roi_crop_demo.py` | Demo del ROI crop |
| `viz_s208_paper.py` | Caso de estudio S208 |

---

## Versionado

`VERSION` es la fuente única de verdad de la versión del paquete `ai/`. El wheel y la tag de Docker se derivan automáticamente de este archivo.

```bash
echo "0.2.0" > VERSION   # bump
make build               # genera dist/scoliosis_ai-0.2.0-py3-none-any.whl
```

---

## Conjunto de datos

**Dataset principal:** *MaIA Scoliosis Dataset v2 corrected* (`data/raw/Scoliosis_Dataset_v2_corrected/`) — 250 radiografías AP de columna provistas en formato anonimizado por el Grupo de Ingeniería Biomédica de la Universidad de los Andes.

| Composición | Conteo |
|---|---|
| Escoliosis | 179 |
| Normales | 71 |
| **Total** | **250** |

### Artefactos por caso
- Imagen original (`Scoliosis/S_{id}.jpg` o `Normal/N_{id}.jpg`)
- Máscara binaria (`LabelBinaryJPG/`)
- Máscara multiclase por vértebra (`LabelMultiClass_ID_PNG/`, valores 1..17 → T1..L5)
- Métricas Cobb pre-computadas — solo casos escoliosis (`RadiographMetrics/`)
- Índice maestro `indice_dataset.csv`

### Auditoría
`data/processed/audit_v2_corrected/` contiene la auditoría completa:
- `clean_index.csv` — estado por caso (`ok` / `warn` / `excluded`)
- `known_issues.csv` — issues por caso con severidad
- `test_holdout.csv` — slice sellado de test (25 casos), agrupado por paciente y estratificado por severidad

### Conjunto entrenable
Tras auditoría: **152 casos** (82 escoliosis + 70 normales) con status `ok` o `warn` y ≥14 vértebras T1–L5 anotadas.

### Conjunto externo (validación cruzada de dominio)
`data/raw/Scoliosis_Dataset_extra_roboflow/` — set público de Roboflow Universe (*scoliosis2*) usado para evaluar transferencia de dominio.

---

## Estado del modelo

| Métrica | Valor | Notas |
|---|---|---|
| Mejor val Dice (80/20 canónico) | **0.643** | `model_primer_v3_corrected` — EncoderUNet(ResNet-34) + augment_v4 + CE+Dice |
| Headline con ROI crop (D1) | **0.674** | `phase1_2_d1_roi_resnet34_dice_0.674_HEADLINE` — primer salto sobre el winner DirectML |
| Winner Fase 0 (boundary λ=0.10) | 0.642 | `phase0_D2_clahe_real_boundary_0.10_dice_0.642_WINNER` |
| MAE Cobb (pipeline GT) | ~10.4° | piso teórico del pipeline con máscaras ground truth |
| Vértebras objetivo | 17 (T1–L5) | sin cervicales |
| Casos entrenables | 152 | 82 escoliosis + 70 normales |
| Split | 80/20 + CV 5-fold | 25 casos sellados como test final |

Detalles completos: [`ai/models/releases/phase1_chain_2026-05-08_ec2_t4_v2_corrected/README.md`](./ai/models/releases/).

---

## Estructura del repositorio

```
scoliosis/
│
├── ai/                              # Librería ML (paquete scoliosis-ai) ─────────────────────
│   ├── __init__.py                  # API pública del paquete
│   ├── preprocessing/
│   │   ├── pipeline.py              # Pipeline de pre-procesamiento (resize, normalizar)
│   │   ├── transforms.py            # Transformaciones determinísticas
│   │   ├── segmentation.py          # Utilidades de máscara (remap T1–L5, binarize)
│   │   ├── roi_crop.py              # ROI crop desde máscara o YOLO
│   │   └── keypoints.py             # Conversión máscara multiclase → 68 keypoints
│   ├── training/
│   │   ├── trainer.py               # Orquestador de entrenamiento con EMA, checkpointing
│   │   ├── dataset.py               # Dataset PyTorch (lee índice CSV, devuelve tensores)
│   │   ├── augmentation.py          # augment_v1..v4 + partial-FOV variants
│   │   ├── losses.py                # CE + Dice + SDHL (boundary)
│   │   ├── splits.py                # Grupo por paciente + estratificación por severidad
│   │   ├── ema.py                   # Exponential Moving Average de pesos
│   │   └── checkpoint.py            # Save / load con cfg-hash
│   ├── evaluation/
│   │   ├── evaluator.py             # Pipeline end-to-end de evaluación
│   │   ├── seg_metrics.py           # Dice, IoU, per-class
│   │   └── cobb.py                  # Cálculo del ángulo Cobb desde la segmentación
│   ├── inference/
│   │   └── predictor.py             # ai.Predictor.from_release(...) — flip-TTA
│   ├── visualization/
│   │   └── case_summary.py          # Tarjetas resumen por caso (overlay + métricas)
│   ├── models/
│   │   ├── architectures/           # nn.Module subclasses (EncoderUNet, etc.)
│   │   ├── checkpoints/             # pesos en curso (gitignored, DVC) ─ ai/models/checkpoints/<run>/
│   │   └── releases/                # ←→ rúbrica "Modelos/"
│   │       └── phase1_chain_2026-05-08_ec2_t4_v2_corrected/
│   │           ├── README.md        # Tabla de 7 runs, cfg de reproducción, entorno EC2
│   │           ├── phase0_*.json    # Resúmenes de ablaciones Fase 0
│   │           ├── phase1_*.json    # Resultados Fase 1.1 y 1.2
│   │           └── <run_dir>/       # cfg.json + history.csv + metrics.json + model.pt
│   └── utils/
│       ├── device.py                # get_device() — CPU / CUDA / ROCm / DirectML
│       └── seeding.py               # Semillas determinísticas
│
├── server/                          # Backend FastAPI ─────────────────────────────────────────
│   ├── main.py                      # FastAPI app factory
│   ├── api/
│   │   ├── health.py                # GET /health
│   │   └── predict.py               # POST /predict/
│   ├── services/
│   │   └── prediction_service.py    # Lógica de inferencia (envuelve ai.Predictor)
│   └── schemas/
│       └── prediction.py            # Pydantic request/response
│
├── frontend/                        # Next.js 16 + React 19 + Tailwind 4 ─────────────────────
│   ├── src/
│   │   ├── app/                     # App Router
│   │   │   ├── login/               # Página de login
│   │   │   ├── prediction/          # Página principal: upload + visualización
│   │   │   └── api/                 # Rutas API server-side (proxy al backend)
│   │   │       ├── demo-login/      # Auth demo (set cookie)
│   │   │       ├── demo-logout/
│   │   │       ├── demo-session/    # Verifica sesión
│   │   │       ├── model-predict/   # Proxy al backend EncoderUNet (server/)
│   │   │       ├── segment-rbunet/  # Proxy al backend secundario DeepLab
│   │   │       └── legacy-segment-rbunet/
│   │   ├── components/
│   │   │   ├── auth/                # LoginForm
│   │   │   ├── prediction/          # ImageUploader, ResultViewer, CobbBadge
│   │   │   ├── site/                # Header, footer, layout
│   │   │   └── ui/                  # Primitivas (button, card, badge)
│   │   ├── lib/                     # Helpers compartidos (fetch, formatters)
│   │   ├── types/                   # Tipos TS compartidos
│   │   └── proxy.ts                 # Config de proxy al backend
│   ├── public/
│   ├── DESIGN_SYSTEM.md             # Tokens visuales del frontend
│   ├── FRONTEND_ARCHITECTURE.md     # Arquitectura del frontend
│   └── README.md                    # Cómo levantar el frontend
│
├── deploy/                          # Despliegue ──────────────────────────────────────────────
│   ├── backend/                     # Despliegue del backend principal (server/)
│   │   └── ec2_setup.sh             # Bootstrap EC2 + systemd
│   ├── deeplab_api/                 # Backend secundario DeepLabV3+
│   │   ├── main.py                  # FastAPI con /predict-binary, /predict-multiclass, /predict-full
│   │   ├── models/                  # .pth checkpoints (runtime-only, no Git)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── ec2_setup.sh                 # Setup inicial EC2 (Python + systemd)
│   ├── ec2_train_userdata.sh        # User-data para EC2 de entrenamiento
│   └── scoliosis-server.service     # systemd unit
│
├── notebooks/                       # ←→ rúbrica "Notebooks/" ─────────────────────────────────
│   ├── README.md                    # Convenciones del flujo sandbox → experiments → final
│   ├── sandbox/                     # ~30 notebooks de exploración (laboratorio del equipo)
│   ├── experiments/
│   │   ├── audit/                   # audit_findings.ipynb
│   │   ├── vertebra-segmentation/   # 5 notebooks (YOLOv8, DeepLab, MONAI U-Net…)
│   │   ├── preprocessing/           # placeholder
│   │   ├── augmentation/            # placeholder
│   │   ├── architectures/           # placeholder
│   │   └── evaluation/              # placeholder
│   └── final/                       # vacío — versión limpia vive en scripts/
│
├── data/                            # ←→ rúbrica "Datos/" ─────────────────────────────────────
│   ├── raw.dvc                      # Pointer DVC al dataset crudo
│   ├── raw/                         # Datasets (DVC, no Git)
│   │   ├── MaIA_Scoliosis_Dataset/                # v1 legacy
│   │   ├── Scoliosis_Dataset_v2/                  # v2 (Spanish, T1–L5)
│   │   ├── Scoliosis_Dataset_v2_corrected/        # v2 con máscaras corregidas
│   │   ├── Scoliosis_Dataset_v2_corrected_x2/     # x2 augmentation set
│   │   └── Scoliosis_Dataset_extra_roboflow/      # Set externo Roboflow
│   ├── processed/                   # Auditoría + splits (texto, en Git)
│   │   ├── audit/                   # v1
│   │   ├── audit_v2/                # v2
│   │   ├── audit_v2_corrected/      # v2c (canónico) — clean_index.csv + test_holdout.csv
│   │   ├── audit_v2_corrected_x2/
│   │   ├── audit_v2_corrected_x2_plus_roboflow/
│   │   ├── case_summaries/          # Tarjetas resumen por caso
│   │   └── v2_corrected_x2_triage/  # Triage manual
│   └── interim/                     # Intermedios temporales
│
├── scripts/                         # Entrypoints DVC y scripts productivos (41 archivos) ─────
│   ├── preprocess.py train.py evaluate.py             # Stages canónicos DVC
│   ├── phase0_ablations.py phase1_2_d1_roi.py …       # Cadenas de experimentos
│   ├── cv5_train.py                                   # CV 5-fold
│   ├── eval_test.py eval_partial_fov.py …             # Evaluadores
│   ├── render_*.py viz_*.py                           # Generación de figuras
│   ├── build_corrected_dataset.py …                   # Dataset utilities
│   └── audit_*.py verify_*.py                         # Auditoría
│
├── experiments/                     # Configs y métricas DVC ──────────────────────────────────
│   └── results/                     # *.json diffeables en Git (resúmenes de runs)
│
├── tests/                           # pytest ──────────────────────────────────────────────────
│   ├── test_no_leakage.py           # Patient-grouped + severity-stratified splits
│   └── ...                          # Unit + arch fitness functions
│
├── docs/                            # Documentación académica + diagramas ────────────────────
│   ├── thesis/scoliosis.tex         # Artículo IEEE
│   ├── diagrams/                    # Diagramas de arquitectura
│   ├── figures/                     # Figuras del paper (gitignored — patient data)
│   ├── papers/                      # Referencias bibliográficas
│   └── presentation/                # Slides de la entrega
│
├── tox.ini                          # Entornos: test_package, test_api, run, train
├── pyproject.toml                   # Build del paquete scoliosis-ai
├── params.yaml                      # Hiperparámetros — fuente única
├── dvc.yaml                         # DAG: preprocess → train → evaluate
├── docker-compose.yml               # Stack local (server + frontend + deeplab_api)
├── Dockerfile                       # Imagen del backend
├── run.sh                           # Container entrypoint
├── Makefile                         # Atajos: pipeline, build, test, deploy
├── VERSION                          # Semver — fuente única
├── .env.example                     # Plantilla de variables de entorno
└── README.md                        # Este archivo
```

### Notas sobre la arquitectura del backend

El paquete `server/` actualmente expone una API plana (`api/`, `services/`, `schemas/`). La especificación objetivo (descrita en [`.claude/CLAUDE.md`](./.claude/CLAUDE.md)) es **Arquitectura Hexagonal** completa (`domain/`, `application/`, `infrastructure/`). La migración es trabajo en curso — el contrato HTTP público no cambia y el código actual cumple los mismos endpoints, solo que con menos separación interna de capas.

---

## Convenciones del proyecto

- **Versionado**: editar solo `VERSION`; `pyproject.toml` y `Makefile` lo leen.
- **Hiperparámetros**: editar solo `params.yaml`.
- **Datos y checkpoints**: vía DVC (`dvc add`, `dvc repro`, `dvc push`).
- **Python ≥ 3.11**, type hints estrictos (`mypy --strict`), linting con `ruff`.
- **Tres niveles de notebook**: `sandbox/` (libre) → `experiments/` (requiere conclusión) → `final/` (tesis-ready).

---

## Licencia y consideraciones éticas

- Uso académico únicamente.
- Radiografías anonimizadas bajo acuerdo institucional con la Universidad de los Andes.
- No se publican imágenes pixel-a-pixel en Git; solo pointers DVC con acceso controlado.
- El sistema es una herramienta de apoyo diagnóstico — **no sustituye el criterio médico profesional**.

---

## Contacto

Para preguntas o acceso a credenciales / DVC remote, contactar a cualquiera de los integrantes.
