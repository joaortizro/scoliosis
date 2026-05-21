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
| Aplicación desplegada | _TBD — se completará antes de la entrega_ |
| Artículo académico (LaTeX) | `docs/thesis/scoliosis.tex` |

---

## Mapeo a la rúbrica (carpetas obligatorias)

La rúbrica pide tres carpetas (`Notebooks/`, `Modelos/`, `Datos/`). En este repositorio mantenemos los nombres reales usados durante el desarrollo, con esta correspondencia explícita:

| Rúbrica | Ruta real en el repo | Contenido |
|---|---|---|
| `Notebooks/` | [`notebooks/`](./notebooks/) | Cuadernos de análisis, entrenamiento y evaluación (tres niveles: `sandbox/`, `experiments/`, `final/`). |
| `Modelos/` | [`ai/models/releases/`](./ai/models/releases/) | Pesos del modelo final (`.pt`) versionados con DVC. Pointer `.dvc` en Git, pesos en el remoto. Ver `phase1_chain_2026-05-08_ec2_t4_v2_corrected/`. |
| `Datos/` | [`data/`](./data/) | Estructura del dataset MaIA Scoliosis v2 (250 radiografías) más índices CSV y subset auditado. Las imágenes son confidenciales — solo pointers `.dvc` viven en Git; los píxeles se obtienen vía `dvc pull`. |

> **Nota:** las imágenes radiográficas son material clínico anonimizado. Por acuerdo institucional con el Grupo de Ingeniería Biomédica de la Universidad de los Andes, **no se publican en Git**. Se distribuyen vía DVC a un remoto S3 con acceso controlado. La estructura, los índices y las máscaras de etiquetado están documentados en [`data/raw/Scoliosis_Dataset_v2/`](./data/raw/) y en [`CLAUDE.md`](./.claude/CLAUDE.md).

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
AUTH_USER=demo
AUTH_PASSWORD=demo123        # TBD — credenciales reales se entregan aparte

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

> _Las credenciales reales de la app desplegada se entregan junto con el documento PDF de la entrega. Las que aparecen abajo sirven solo para ejecutar la app localmente._

| Campo | Valor demo |
|---|---|
| Usuario | _TBD_ |
| Contraseña | _TBD_ |

(Sustituir antes de entrega una vez confirmadas las credenciales.)

---

## Ejemplos de uso

### Vía interfaz web

1. Abrir <http://localhost:3000> (o la URL desplegada).
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

## Despliegue

### Opción A — EC2 + systemd

```bash
# Primera vez en EC2:
scp -i ~/.ssh/key.pem -r server/ deploy/ .env ec2-user@<ip>:~/scoliosis/
ssh -i ~/.ssh/key.pem ec2-user@<ip> "bash ~/scoliosis/deploy/ec2_setup.sh"

# Subsecuentes:
make deploy EC2_HOST=ec2-user@<ip> PEM=~/.ssh/key.pem
```

| Method | Route | Description |
| --- | --- | --- |
| GET | `/health` | Health check |
| POST | `/predict/` | Upload image, get prediction |

Docs at [http://localhost:8001/docs](http://localhost:8001/docs)

## DeepLab API Backend

The newer DeepLabV3+ inference service lives in `deploy/deeplab_api/`. It is
kept separate from the existing backend under `deploy/backend/`.

Expected layout:

```text
deploy/
  backend/                 existing backend, do not touch for DeepLab changes
  deeplab_api/             new backend
    app/
      main.py
    models/
      .gitkeep
      best_model_binary.pth
      best_model_multi.pth
    Dockerfile
    requirements.txt
    README.md
    .gitignore
```

The `.pth` model files must be present at runtime in `deploy/deeplab_api/models/`.
They are large binary artifacts and should not be committed to Git. Add them
locally, through the hosting provider file storage/build context, or through the
deployment process before starting the API.

The current DeepLab API exposes:

| Method | Route | Description |
| --- | --- | --- |
| GET | `/health` | Health check and model-loaded status |
| POST | `/predict-binary` | Binary segmentation |
| POST | `/predict-multiclass` | T1-L5 multiclass segmentation |
| POST | `/predict-full` | Combined binary + multiclass response |

---

## Notebooks

Three-tier system — see [notebooks/README.md](notebooks/README.md) for full conventions.

| Folder | Purpose | Rules |
| --- | --- | --- |
| `sandbox/` | Personal scratchpads | No rules — experiment freely |
| `experiments/<topic>/` | Shared topic notebooks | Must have a Conclusions cell |
| `final/` | Thesis-ready, numbered | Reproducible top-to-bottom, reviewed |

**Promotion flow:** `sandbox/` → `experiments/` → `final/`

Topics under `experiments/`: `preprocessing/`, `augmentation/`, `architectures/`, `evaluation/`

---

## Versioning

Bump `VERSION` before any release — the wheel name and Docker image tag are derived from it:

```bash
make deploy-ecs AWS_ACCOUNT=<id> AWS_REGION=us-east-1
```

### Opción C — Frontend en Vercel

```bash
cd frontend
vercel --prod
```

Configurar `NEXT_PUBLIC_API_URL` apuntando al backend desplegado.

---

## Estado del modelo

| Métrica | Valor | Notas |
|---|---|---|
| Mejor val Dice (80/20 canónico) | **0.643** | `model_primer_v3_corrected` — EncoderUNet(resnet34) + augment_v4 |
| Headline con ROI crop (D1) | **0.674** | `phase1_2_d1_roi_resnet34_dice_0.674_HEADLINE` |
| MAE Cobb (pipeline GT) | ~10.4° | piso teórico del pipeline con máscaras GT |
| Vértebras objetivo | 17 (T1–L5) | sin cervicales |
| Casos entrenables | 152 | 82 escoliosis + 70 normales |

---

## Estructura del repositorio

```
scoliosis/
├── ai/                    # Librería ML (paquete scoliosis-ai)
│   ├── preprocessing/
│   ├── models/
│   │   ├── architectures/
│   │   ├── checkpoints/   # pesos en curso (gitignored, DVC)
│   │   └── releases/      # ←→ rúbrica: "Modelos/"
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── server/                # Backend FastAPI (hexagonal)
│   ├── domain/
│   ├── application/
│   ├── infrastructure/
│   └── main.py
├── frontend/              # Next.js 15
├── notebooks/             # ←→ rúbrica: "Notebooks/"
│   ├── sandbox/
│   ├── experiments/
│   └── final/
├── data/                  # ←→ rúbrica: "Datos/"
│   ├── raw/               # dataset original (DVC)
│   └── processed/         # auditoría + splits
├── scripts/               # entrypoints DVC
├── experiments/results/   # métricas JSON diffeables
├── deploy/                # systemd + scripts EC2/ECS
├── docs/
│   └── thesis/scoliosis.tex
├── tests/
├── params.yaml            # hiperparámetros (fuente única)
├── dvc.yaml               # DAG del pipeline
├── pyproject.toml
├── docker-compose.yml
├── Makefile
└── VERSION
```

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
