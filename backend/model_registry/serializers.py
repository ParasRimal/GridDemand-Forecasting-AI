from rest_framework import serializers

from .models import ModelVersion


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = [
            "id", "model_name", "model_type", "status", "is_current_production",
            "mae", "rmse", "mape", "r2", "reason", "evaluated_at", "created_at",
        ]
        read_only_fields = fields
