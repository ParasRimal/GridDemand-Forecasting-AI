from rest_framework import viewsets

from .models import ModelVersion
from .serializers import ModelVersionSerializer


class ModelVersionViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only: promotion decisions are recorded by the training pipeline, not via the API."""
    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer


import glob
import sys
from pathlib import Path

from django.http import JsonResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def feature_importance(request):
    """Real feature importances from the saved random_forest candidate model
    (the backtest-selected 'no season' variant). Random Forest exposes
    feature_importances_ natively; other model types may not."""
    try:
        from ml_service.app.training.artifacts import load_artifact
    except ImportError as exc:
        return JsonResponse({"error": f"Could not load ML pipeline: {exc}"}, status=500)

    folders = sorted(glob.glob(str(PROJECT_ROOT / "models" / "random_forest_*")))
    if not folders:
        return JsonResponse({"error": "No saved random_forest model found."}, status=404)

    model, meta = load_artifact(folders[-1])
    if not hasattr(model, "feature_importances_"):
        return JsonResponse({"error": "Saved model does not expose feature importances."}, status=404)

    pairs = list(zip(meta["feature_columns"], model.feature_importances_))
    pairs.sort(key=lambda p: p[1], reverse=True)

    return JsonResponse({
        "model_name": meta.get("model_name", "random_forest"),
        "trained_on": meta.get("trained_on"),
        "importances": [
            {"feature": name, "importance": round(float(value), 4)}
            for name, value in pairs
        ],
    })
