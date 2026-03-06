from fastapi import FastAPI
from server.api import health, predict

app = FastAPI(title="Scoliosis API", version="0.1.0")

app.include_router(health.router)
app.include_router(predict.router, prefix="/predict")
