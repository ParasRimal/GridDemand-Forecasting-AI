from rest_framework.routers import DefaultRouter

from .views import ModelVersionViewSet

router = DefaultRouter()
router.register("model-versions", ModelVersionViewSet, basename="modelversion")

urlpatterns = router.urls
