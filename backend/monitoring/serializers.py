from rest_framework import serializers

from .models import DataMonitoring


class DataMonitoringSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataMonitoring
        fields = [
            "id", "checked_at", "drift_detected", "drifted_column_share",
            "drifted_column_count", "n_columns", "performance_degraded",
            "recent_mae", "reference_mae", "relative_increase",
            "retrain_recommended", "reason", "created_at",
        ]
        read_only_fields = fields
