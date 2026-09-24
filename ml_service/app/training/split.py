"""Chronological train / validation / test split.

The data is cut by DATE, never shuffled, so the models are always tested on
a period that comes after everything they trained on.

    train:       index <= train_end
    validation:  train_end < index <= validation_end
    test:        index > validation_end
"""
from __future__ import annotations

import logging

import pandas as pd

from ml_service.app import config
from ml_service.app.preprocessing.pipeline import feature_columns

logger = logging.getLogger(__name__)


def split_chronological(
    df: pd.DataFrame,
    train_end: str = config.TRAIN_END,
    validation_end: str = config.VALIDATION_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, validation, test) by date."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted oldest to newest before splitting.")

    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    if train_end_ts >= validation_end_ts:
        raise ValueError("train_end must be earlier than validation_end.")

    train = df[df.index <= train_end_ts]
    validation = df[(df.index > train_end_ts) & (df.index <= validation_end_ts)]
    test = df[df.index > validation_end_ts]

    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if part.empty:
            raise ValueError(f"The {name} split is empty; check the split dates.")

    logger.info(
        "Split: train=%d (%s to %s) | validation=%d (%s to %s) | test=%d (%s to %s)",
        len(train), train.index.min(), train.index.max(),
        len(validation), validation.index.min(), validation.index.max(),
        len(test), test.index.min(), test.index.max(),
    )
    return train, validation, test


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the model inputs X (features only) from the target y."""
    X = df[feature_columns()]
    y = df[config.TARGET_COLUMN]
    return X, y
