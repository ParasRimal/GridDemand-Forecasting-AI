"""Build forecasting models by name, using the settings in config.py."""
from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from ml_service.app import config


def _build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(random_state=config.RANDOM_SEED, **config.RANDOM_FOREST_PARAMS)


def _build_xgboost() -> XGBRegressor:
    return XGBRegressor(random_state=config.RANDOM_SEED, **config.XGBOOST_PARAMS)


def _build_lightgbm() -> LGBMRegressor:
    return LGBMRegressor(random_state=config.RANDOM_SEED, **config.LIGHTGBM_PARAMS)


_BUILDERS = {
    "random_forest": _build_random_forest,
    "xgboost": _build_xgboost,
    "lightgbm": _build_lightgbm,
}


def available_models() -> list[str]:
    """Names accepted by build_model()."""
    return list(_BUILDERS)


def build_model(name: str):
    """Return a new, untrained model."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")
    return _BUILDERS[name]()
