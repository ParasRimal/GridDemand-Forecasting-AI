"""GridPredict ML service entrypoint.

Run locally with:
    uvicorn ml_service.app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ml_service.app.inference.production_model import ProductionModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Loaded once at startup, reused across requests.
production_model: ProductionModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global production_model
    production_model = ProductionModel()
    logger.info("Production model loaded: %s", production_model.info())
    yield
    production_model = None


app = FastAPI(title="GridPredict ML Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Basic liveness + what's currently serving predictions."""
    if production_model is None:
        return {"status": "starting", "production_model": None}
    return {"status": "ok", "production_model": production_model.info()}
