"""Loads whatever the registry says is currently in production, and exposes
a single predict() interface regardless of whether that's the naive baseline
or a trained model artifact.

This is intentionally the ONLY place that inspects registry["production"]["type"];
everything downstream (routes, schemas) just calls predict().
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ml_service.app import config
from ml_service.app.preprocessing.pipeline import feature_columns
from ml_service.app.training.artifacts import load_artifact
from ml_service.app.training.registry import load_registry

logger = logging.getLogger(__name__)


class ProductionModel:
    """Wraps the current production entry from registry.json behind predict()."""

    def __init__(self, registry: Optional[dict] = None):
        self.registry = registry if registry is not None else load_registry()
        self.production = self.registry["production"]
        self.model_type = self.production["type"]  # "baseline" or "candidate"
        self._sk_model = None
        self._feature_columns = feature_columns()

        if self.model_type == "candidate":
            artifact_folder = self.production.get("artifact_folder")
            if not artifact_folder:
                raise ValueError(
                    "Production entry has type='candidate' but no 'artifact_folder' recorded."
                )
            self._sk_model, meta = load_artifact(artifact_folder)
            self._feature_columns = meta["feature_columns"]
            logger.info("Loaded production candidate model from %s", artifact_folder)
        elif self.model_type == "baseline":
            logger.info("Production is the naive baseline (%s).", self.production.get("name"))
        else:
            raise ValueError(f"Unknown production model type: '{self.model_type}'")

    @property
    def required_features(self) -> list[str]:
        """The feature columns the caller must supply to predict()."""
        if self.model_type == "baseline":
            return [f"lag_{config.BASELINE_LAG_HOURS}"]
        return list(self._feature_columns)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return a prediction per row of X. X must contain required_features."""
        missing = [c for c in self.required_features if c not in X.columns]
        if missing:
            raise KeyError(f"Missing required feature columns: {missing}")

        if self.model_type == "baseline":
            lag_col = f"lag_{config.BASELINE_LAG_HOURS}"
            return X[lag_col].rename("prediction")

        return pd.Series(self._sk_model.predict(X[self._feature_columns]), index=X.index, name="prediction")

    def info(self) -> dict:
        """Small dict describing what's currently serving predictions."""
        return {
            "type": self.model_type,
            "name": self.production.get("name"),
            "mae": self.production.get("mae"),
            "promoted_at": self.production.get("promoted_at"),
        }
