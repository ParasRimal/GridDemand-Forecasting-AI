"""Build forecasting models by name, using the settings in config.py."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from ml_service.app import config


def _build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(random_state=config.RANDOM_SEED, **config.RANDOM_FOREST_PARAMS)


_BUILDERS = {
    "random_forest": _build_random_forest,
}


def available_models() -> list[str]:
    """Names accepted by build_model()."""
    return list(_BUILDERS)


def build_model(name: str):
    """Return a new, untrained model."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")
    return _BUILDERS[name]()
