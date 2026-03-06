from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: str | None
    confidence: float | None
