from rest_framework import viewsets

from .models import Forecast
from .serializers import ForecastSerializer


class ForecastViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only: forecasts are created by the ML pipeline/Celery, not via the API."""
    queryset = Forecast.objects.all()
    serializer_class = ForecastSerializer


import sys
from pathlib import Path

from django.http import JsonResponse

# ml_service lives outside backend/, at the project root -- same bridge
# pattern used in monitoring/tasks.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def historical_demand(request):
    """Real historical load, aggregated to daily min/avg/max, from the
    actual processed training dataset (ahmedabad_processed.csv)."""
    try:
        from ml_service.app import config
        from ml_service.app.preprocessing.pipeline import load_processed
    except ImportError as exc:
        return JsonResponse({"error": f"Could not load ML pipeline: {exc}"}, status=500)

    try:
        df = load_processed()
    except FileNotFoundError:
        return JsonResponse({"error": "Processed dataset not found. Run the training pipeline first."}, status=404)

    target = config.TARGET_COLUMN
    daily = df[target].resample("D").agg(["mean", "min", "max"]).dropna()

    data = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "avg_load_mw": round(row["mean"], 2),
            "min_load_mw": round(row["min"], 2),
            "max_load_mw": round(row["max"], 2),
        }
        for idx, row in daily.iterrows()
    ]
    return JsonResponse(data, safe=False)
