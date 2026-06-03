from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import HealthResponse, ModelInfoResponse, ScreeningRequest, ScreeningResponse
from app.services.predictor import Predictor


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

app = FastAPI(
    title="Renal Evidence Studio",
    version="1.0.0",
    description="Educational CKD screening API with model comparison and explanation artifacts.",
)
predictor = Predictor()


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/model-info", response_model=ModelInfoResponse)
def model_info() -> dict[str, Any]:
    return predictor.model_info()


@app.get("/api/v1/metrics")
def metrics() -> dict[str, Any]:
    return predictor.metrics


@app.post("/api/v1/screen", response_model=ScreeningResponse)
def screen(payload: ScreeningRequest) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    else:
        data = payload.dict()
    return predictor.predict(data)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
