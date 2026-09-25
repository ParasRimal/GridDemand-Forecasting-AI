from django.contrib import admin

from .models import Forecast


@admin.register(Forecast)
class ForecastAdmin(admin.ModelAdmin):
    list_display = ["timestamp", "predicted_load_mw", "actual_load_mw", "model_type", "model_name"]
    list_filter = ["model_type", "model_name"]
    ordering = ["-timestamp"]
