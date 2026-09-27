from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import ForecastViewSet, historical_demand

router = DefaultRouter()
router.register("forecasts", ForecastViewSet, basename="forecast")

urlpatterns = router.urls + [
    path("historical-demand/", historical_demand),
]
