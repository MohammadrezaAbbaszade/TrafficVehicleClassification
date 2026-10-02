"""HTTP interface for the selected traffic-vehicle classifier."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from predict import VehiclePredictor, format_prediction_percentages


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
app = FastAPI(title="Traffic Vehicle Classification", version="1.0.0")


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: str = Field(description="Percentage, without scientific notation", examples=["93%"])
    probabilities: dict[str, str] = Field(
        description="Class probabilities as percentage strings, without scientific notation"
    )
    needs_review: bool


@lru_cache(maxsize=1)
def load_predictor() -> VehiclePredictor:
    """Reuse the same model across requests instead of reloading every image."""
    return VehiclePredictor()


def get_predictor() -> VehiclePredictor:
    try:
        return load_predictor()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Model checkpoint is unavailable") from exc


@app.get("/health")
def health(predictor: Annotated[VehiclePredictor, Depends(get_predictor)]):
    return {"status": "ready", "classes": predictor.class_names}


@app.post("/predict", response_model=PredictionResponse)
def predict_upload(
    file: Annotated[UploadFile, File(description="A cropped vehicle image")],
    predictor: Annotated[VehiclePredictor, Depends(get_predictor)],
):
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit")

    try:
        image = Image.open(file.file)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image") from exc

    try:
        return format_prediction_percentages(predictor.predict(image))
    finally:
        image.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
