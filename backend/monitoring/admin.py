from django.contrib import admin

from .models import DataMonitoring


@admin.register(DataMonitoring)
class DataMonitoringAdmin(admin.ModelAdmin):
    list_display = [
        "checked_at", "drift_detected", "drifted_column_share",
        "performance_degraded", "retrain_recommended",
    ]
    list_filter = ["drift_detected", "performance_degraded", "retrain_recommended"]
    ordering = ["-checked_at"]
