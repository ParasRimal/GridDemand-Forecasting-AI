from rest_framework.routers import DefaultRouter

from .views import DataMonitoringViewSet

router = DefaultRouter()
router.register("monitoring", DataMonitoringViewSet, basename="datamonitoring")

urlpatterns = router.urls
