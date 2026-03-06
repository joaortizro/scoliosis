# Scoliosis Detection — Master's Degree Project

AI-powered scoliosis detection system using deep learning, with full experiment tracking and reproducible pipelines.

---

## Stack

| Layer | Technology |
|---|---|
| ML framework | PyTorch |
| Experiment tracking | MLflow |
| Pipeline / data versioning | DVC |
| API server | FastAPI + Uvicorn |
| Packaging | setuptools wheel (`.whl`) |
| Dev environments | tox |
| Containerization | Docker + Docker Compose |
| Container registry | AWS ECR |
| Deployment | AWS ECS Fargate / EC2 + systemd |
| CI | GitHub Actions |

---

## Project Structure

```
scoliosis/
├── .github/workflows/     # CI: run tests + build wheel on every push
├── data/                  # DVC-tracked datasets (not in Git)
│   ├── raw/
│   ├── processed/
│   └── interim/
├── ai/                    # Core ML logic — packaged as scoliosis-ai wheel
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── experiments/           # DVC configs and result metrics
├── mlflow/                # MLflow server setup
├── notebooks/             # Three-tier: sandbox → experiments → final
│   ├── sandbox/           # Personal scratchpads (no rules)
│   ├── experiments/       # Shared ideas by topic (needs Conclusions cell)
│   └── final/             # Thesis-ready, numbered, reproducible
├── scripts/               # DVC stage entrypoints (preprocess, train, evaluate)
├── server/                # FastAPI backend
├── front/                 # Frontend app
├── deploy/                # EC2 deployment helpers (systemd service)
├── docs/
│   ├── references/        # PDFs, papers (gitignored — share via Drive)
│   ├── diagrams/
│   └── thesis/
├── dist/                  # Built wheels (gitignored)
├── tests/
├── params.yaml            # Hyperparameters — single source of truth
├── dvc.yaml               # Pipeline DAG (preprocess → train → evaluate)
├── pyproject.toml         # Package definition for ai/ wheel
├── tox.ini                # Unified environments: test_package, train, run
├── VERSION                # Semantic version — bump here only
├── run.sh                 # Container entrypoint (reads $PORT from env)
└── Makefile               # Shortcuts: train, build, test, deploy-ecs, deploy-ec2
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd scoliosis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize DVC

```bash
dvc init
dvc remote add -d myremote <your-storage-url>   # s3://, gdrive://, ssh://
dvc pull                                         # fetch data and checkpoints
```

### 3. Configure environment

```bash
cp .env .env.local
# edit .env.local with your values
```

---

## Running the Pipeline

```bash
make pipeline              # preprocess → train → evaluate (via dvc repro)
make preprocess            # single stage
make train
make evaluate
```

Compare runs:

```bash
dvc params diff
dvc metrics diff
```

---

## Tests

```bash
make test                  # tox -e test_package
tox -e test_api            # API-specific tests
```

---

## Experiment Tracking (MLflow)

```bash
make mlflow-ui
# or:
docker compose up mlflow
```

Open [http://localhost:5000](http://localhost:5000)

---

## API Server

```bash
tox -e run                 # local dev with reload
# or:
docker compose up server
```

| Method | Route | Description |
| --- | --- | --- |
| GET | `/health` | Health check |
| POST | `/predict/` | Upload image, get prediction |

Docs at [http://localhost:8001/docs](http://localhost:8001/docs)

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
echo "0.2.0" > VERSION
```

---

## Deploy

Two paths depending on the stage of the project:

```
Local                    S3 / ECR                     EC2 / ECS
─────                    ────────                     ─────────
make pipeline ──dvc──▶  checkpoint              ──dvc pull──▶ model on server
make build    ──scp──▶  (EC2 path)              ──pip install *.whl
              ──ecr──▶  Docker image (ECS path) ──ecs update-service
```

### Option A — EC2 + systemd (simple, early-stage)

```bash
# First time on EC2 (run once):
scp -i ~/.ssh/your-key.pem -r server/ deploy/ .env ec2-user@<ip>:~/scoliosis/
ssh -i ~/.ssh/your-key.pem ec2-user@<ip> "bash ~/scoliosis/deploy/ec2_setup.sh"

# Subsequent deploys:
make deploy EC2_HOST=ec2-user@<ip> PEM=~/.ssh/your-key.pem
```

### Option B — ECS Fargate + ECR (production, zero-downtime)

```bash
# Full deploy: push checkpoint + build image + push to ECR + redeploy service
make deploy-ecs AWS_ACCOUNT=<id> AWS_REGION=us-east-1

# Or step by step:
make push-data            # dvc push → S3
make docker-push          # build image → push to ECR
make ecs-deploy           # aws ecs update-service (rolling update)
```

### Makefile variables

| Variable | Default | Description |
| --- | --- | --- |
| `EC2_HOST` | `ec2-user@<ip>` | EC2 SSH target |
| `PEM` | `~/.ssh/your-key.pem` | SSH key path |
| `AWS_ACCOUNT` | `<your-aws-account-id>` | AWS account number |
| `AWS_REGION` | `us-east-1` | AWS region |
| `ECR_REPO` | `scoliosis-api` | ECR repository name |
| `ECS_CLUSTER` | `scoliosis-cluster` | ECS cluster name |
| `ECS_SERVICE` | `scoliosis-service` | ECS service name |

---

## What lives where

| Artifact | Versioned by | Location |
| --- | --- | --- |
| Source code | Git | GitHub / GitLab |
| Datasets | DVC | S3 / GDrive / SSH |
| Model checkpoints | DVC | S3 / GDrive / SSH |
| `ai/` inference logic | Wheel (`.whl`) | `dist/` → EC2 or baked into Docker image |
| Docker image | ECR | `<account>.dkr.ecr.<region>.amazonaws.com/scoliosis-api` |
| Experiment metadata | MLflow | `mlflow/mlruns/` |
| Version | `VERSION` file | Read by `pyproject.toml` and `Makefile` |
