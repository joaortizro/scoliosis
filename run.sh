#!/bin/bash
# Container entrypoint — PORT is injected by ECS task definition or docker run -e
set -e

PORT=${PORT:-8001}
exec uvicorn server.main:app --host 0.0.0.0 --port "$PORT"
