FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project's ai/ package + pyproject so we can `pip install -e .`.
# Build context must be the repo root for these COPYs to resolve; see
# README → "Build with RB-UNet" section.
COPY ../../ai /repo/ai
COPY ../../pyproject.toml /repo/pyproject.toml
COPY ../../VERSION /repo/VERSION
RUN pip install --no-cache-dir -e /repo

# Backend code
COPY main.py rbunet.py ./

# YOLO weights (mount as volume in dev, or COPY here for prod images)
# COPY spine_yolo/ spine_yolo/

# RB-UNet checkpoints + experiment sentinel (paths resolved by rbunet.py)
# COPY ../../ai/models/checkpoints/encoder_unet /repo/ai/models/checkpoints/encoder_unet
# COPY ../../experiments/results/phase1_2_d2_5fold.json /repo/experiments/results/phase1_2_d2_5fold.json
# In practice these are large; prefer `dvc pull` on the host before building,
# or mount them as a volume:
#   -v $(pwd)/../../ai/models/checkpoints:/repo/ai/models/checkpoints:ro
#   -v $(pwd)/../../experiments/results:/repo/experiments/results:ro
ENV SCOLIOSIS_REPO_ROOT=/repo

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
