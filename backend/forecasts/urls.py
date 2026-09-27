from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import ForecastViewSet, historical_demand, modeling_findings

router = DefaultRouter()
router.register("forecasts", ForecastViewSet, basename="forecast")

urlpatterns = router.urls + [
    path("historical-demand/", historical_demand),
    path("modeling-findings/", modeling_findings),
]
