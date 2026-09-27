from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import ModelVersionViewSet, feature_importance

router = DefaultRouter()
router.register("model-versions", ModelVersionViewSet, basename="modelversion")

urlpatterns = router.urls + [
    path("feature-importance/", feature_importance),
]
