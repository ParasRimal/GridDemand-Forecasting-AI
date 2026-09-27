"""
URL configuration for config project.
"""
from django.contrib import admin
from django.http import JsonResponse
import redis
import os
from django.urls import path, include


def root_status(request):
    return JsonResponse({"service": "GridPredict Django backend", "status": "ok"})


def redis_status(request):
    try:
        r = redis.Redis(
            host=os.environ.get("REDIS_HOST", "127.0.0.1"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            socket_connect_timeout=2,
        )
        r.ping()
        return JsonResponse({"status": "ok"})
    except redis.RedisError as exc:
        return JsonResponse({"status": "error", "detail": str(exc)}, status=503)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('forecasts.urls')),
    path('api/', include('model_registry.urls')),
    path('api/', include('monitoring.urls')),
    path('health/redis/', redis_status),
    path('', root_status),
]
