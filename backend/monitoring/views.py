from rest_framework import viewsets

from .models import DataMonitoring
from .serializers import DataMonitoringSerializer


class DataMonitoringViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only: monitoring checks are recorded by Celery/the monitoring pipeline, not via the API."""
    queryset = DataMonitoring.objects.all()
    serializer_class = DataMonitoringSerializer
