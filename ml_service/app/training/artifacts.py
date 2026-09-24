"""Save and load trained models together with a metadata file.

Each save creates a NEW folder, so an older model is never overwritten:

    models/<model_name>_<UTC timestamp>/
        model.joblib     the trained model
        metadata.json    feature list, scores, training dates, settings, ...
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib

from ml_service.app import config

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def _json_default(obj: Any):
    """Let json handle numpy numbers; anything else becomes a string."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def save_artifact(
    model,
    model_name: str,
    metadata: Mapping[str, Any],
    base_dir: Optional[Path] = None,
) -> Path:
    """Save the model and its metadata into a new folder and return that folder."""
    base_dir = Path(base_dir or config.MODELS_DIR)
    saved_at = datetime.now(timezone.utc)
    folder = base_dir / f"{model_name}_{saved_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    folder.mkdir(parents=True, exist_ok=False)  # refuses to reuse an existing folder

    full_metadata = dict(metadata)
    full_metadata["model_name"] = model_name
    full_metadata["saved_at"] = saved_at.isoformat()

    joblib.dump(model, folder / MODEL_FILE)
    with open(folder / METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2, default=_json_default)

    logger.info("Saved model artifact to %s", folder)
    return folder


def load_artifact(folder: Path) -> tuple[Any, dict]:
    """Load (model, metadata) from a folder made by save_artifact()."""
    folder = Path(folder)
    model_path = folder / MODEL_FILE
    metadata_path = folder / METADATA_FILE
    for path in (model_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {path}")

    model = joblib.load(model_path)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata
