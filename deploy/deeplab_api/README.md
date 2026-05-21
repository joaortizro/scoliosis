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

Endpoints:

- `GET /health`
- `POST /predict-binary`
- `POST /predict-multiclass`
- `POST /predict-full`