from rest_framework.routers import DefaultRouter

from .views import ForecastViewSet

router = DefaultRouter()
router.register("forecasts", ForecastViewSet, basename="forecast")

urlpatterns = router.urls
