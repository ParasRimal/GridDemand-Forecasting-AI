"""GridPredict ML service entrypoint.

Run locally with:
    uvicorn ml_service.app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ml_service.app.inference.production_model import ProductionModel
from ml_service.app.routes.predict import router as predict_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.production_model = ProductionModel()
    logger.info("Production model loaded: %s", app.state.production_model.info())
    yield
    app.state.production_model = None


app = FastAPI(title="GridPredict ML Service", version="0.1.0", lifespan=lifespan)
app.include_router(predict_router)


@app.get("/health")
def health() -> dict:
    """Basic liveness + what's currently serving predictions."""
    production_model = getattr(app.state, "production_model", None)
    if production_model is None:
        return {"status": "starting", "production_model": None}
    return {"status": "ok", "production_model": production_model.info()}


@app.get("/model")
def model_info() -> dict:
    """Detail on the model currently serving predictions."""
    production_model = getattr(app.state, "production_model", None)
    if production_model is None:
        raise HTTPException(status_code=503, detail="Production model is not loaded yet.")
    info = production_model.info()
    info["required_features"] = production_model.required_features
    return info


@app.get("/metrics")
def metrics() -> dict:
    """Recorded performance metrics for the current production model, if any."""
    production_model = getattr(app.state, "production_model", None)
    if production_model is None:
        raise HTTPException(status_code=503, detail="Production model is not loaded yet.")
    production_entry = production_model.production
    return {
        "model_type": production_entry["type"],
        "model_name": production_entry.get("name"),
        "mae": production_entry.get("mae"),
        "metrics": production_entry.get("metrics"),
    }
