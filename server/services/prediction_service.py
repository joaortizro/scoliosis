from fastapi import UploadFile
from ai.inference.predictor import Predictor
import os

_predictor = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        checkpoint = os.getenv("MODEL_CHECKPOINTS_DIR", "ai/models/checkpoints")
        _predictor = Predictor(checkpoint)
    return _predictor


async def run_prediction(file: UploadFile) -> dict:
    # TODO: read image bytes, preprocess, predict
    predictor = get_predictor()
    return {"prediction": None, "confidence": None}
