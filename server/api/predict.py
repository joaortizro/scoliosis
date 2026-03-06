from fastapi import APIRouter, UploadFile, File
from server.services.prediction_service import run_prediction

router = APIRouter()


@router.post("/")
async def predict(file: UploadFile = File(...)):
    result = await run_prediction(file)
    return result
