"""Train a model and score it. Works for any model in model_factory."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ml_service.app.evaluation.metrics import evaluate
from ml_service.app.training.model_factory import build_model

logger = logging.getLogger(__name__)


def train_model(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Build the named model and fit it on the training data."""
    if len(X_train) == 0:
        raise ValueError("Training data is empty.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different lengths.")
    if y_train.isna().any():
        raise ValueError("y_train contains missing values; drop those rows first.")

    model = build_model(name)
    model.fit(X_train, y_train)
    logger.info("Trained '%s' on %d rows.", name, len(X_train))
    return model


def predict(model, X: pd.DataFrame) -> pd.Series:
    """Predict the load, keeping the Timestamp index."""
    return pd.Series(model.predict(X), index=X.index, name="prediction")


def score_model(
    model, X: pd.DataFrame, y: pd.Series, rows: Optional[pd.Series] = None
) -> dict:
    """Score the model with all four metrics.

    `rows` is an optional True/False Series (same index as X) that limits scoring
    to selected rows, e.g. the rows where a baseline can also predict.
    """
    prediction = predict(model, X)
    if rows is not None:
        prediction, y = prediction[rows], y[rows]
    return evaluate(y, prediction)
