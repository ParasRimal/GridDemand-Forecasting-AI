from django.contrib import admin

from .models import ModelVersion


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ["model_name", "model_type", "status", "is_current_production", "mae", "evaluated_at"]
    list_filter = ["status", "model_type", "is_current_production"]
    ordering = ["-evaluated_at"]
