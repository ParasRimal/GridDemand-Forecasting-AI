"""
URL configuration for config project.
"""
from django.contrib import admin
from django.http import JsonResponse
from django.urls import path


def root_status(request):
    return JsonResponse({"service": "GridPredict Django backend", "status": "ok"})


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', root_status),
]
