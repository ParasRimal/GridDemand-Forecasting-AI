"""POST /predict: day-ahead load forecast for a single target hour.

Required fields depend on what's currently in production (baseline needs
only lag_24; a trained candidate needs its full recorded feature list).
This is checked here against ProductionModel.required_features, not
hardcoded into the request schema, so the API contract survives promotion.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ml_service.app.caching.prediction_cache import get_cached_prediction, set_cached_prediction
from ml_service.app.schemas.predict import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchResultItem,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    production_model = request.app.state.production_model
    if production_model is None:
        raise HTTPException(status_code=503, detail="Production model is not loaded yet.")

    cache_payload = payload.model_dump(by_alias=True)
    cached = get_cached_prediction(cache_payload)
    if cached is not None:
        return PredictResponse(**cached)

    row = payload.to_feature_row()
    missing = [c for c in production_model.required_features if c not in row.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Missing required feature(s) for the current production model "
                f"({production_model.model_type}): {missing}"
            ),
        )

    prediction = production_model.predict(row)
    info = production_model.info()
    result = PredictResponse(
        timestamp=payload.timestamp,
        predicted_load_mw=float(prediction.iloc[0]),
        model_type=info["type"],
        model_name=info["name"],
    )
    set_cached_prediction(cache_payload, result.model_dump())
    return result


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    production_model = request.app.state.production_model
    if production_model is None:
        raise HTTPException(status_code=503, detail="Production model is not loaded yet.")

    results: list[BatchResultItem] = []
    for i, item in enumerate(payload.items):
        try:
            row = item.to_feature_row()
            missing = [c for c in production_model.required_features if c not in row.columns]
            if missing:
                raise ValueError(
                    f"Missing required feature(s) for the current production model "
                    f"({production_model.model_type}): {missing}"
                )
            prediction = production_model.predict(row)
            info = production_model.info()
            results.append(BatchResultItem(
                index=i,
                success=True,
                result=PredictResponse(
                    timestamp=item.timestamp,
                    predicted_load_mw=float(prediction.iloc[0]),
                    model_type=info["type"],
                    model_name=info["name"],
                ),
            ))
        except Exception as exc:
            results.append(BatchResultItem(index=i, success=False, error=str(exc)))

    return BatchPredictResponse(results=results)
