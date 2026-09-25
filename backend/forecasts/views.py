from rest_framework import viewsets

from .models import Forecast
from .serializers import ForecastSerializer


class ForecastViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only: forecasts are created by the ML pipeline/Celery, not via the API."""
    queryset = Forecast.objects.all()
    serializer_class = ForecastSerializer
