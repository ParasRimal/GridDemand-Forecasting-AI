from rest_framework import viewsets

from .models import ModelVersion
from .serializers import ModelVersionSerializer


class ModelVersionViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only: promotion decisions are recorded by the training pipeline, not via the API."""
    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer
