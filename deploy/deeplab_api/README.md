---
title: Scoliosis DeepLab API
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Scoliosis DeepLab API

FastAPI backend for binary and multiclass scoliosis segmentation.

## Expected Files

This backend is separate from the existing deployment backend:

```text
deploy/
  backend/                 existing backend, do not touch for this API
  deeplab_api/             new DeepLabV3+ API
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

The model checkpoints must be available at runtime in:

```text
deploy/deeplab_api/models/
```

Expected checkpoint filenames:

- `best_model_binary.pth`
- `best_model_multi.pth`

Keep `.pth` files out of Git. The `models/.gitkeep` file exists only so the
folder can be present in the repository. Add the real `.pth` files locally or as
part of the deployment/build process before running the container.

## Endpoints

Endpoints:

- `GET /health`
- `POST /predict-binary`
- `POST /predict-multiclass`
- `POST /predict-full`
