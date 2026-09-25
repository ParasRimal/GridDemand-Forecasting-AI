from rest_framework import serializers

from .models import Forecast


class ForecastSerializer(serializers.ModelSerializer):
    class Meta:
        model = Forecast
        fields = [
            "id", "timestamp", "predicted_load_mw", "actual_load_mw",
            "model_type", "model_name", "created_at",
        ]
        read_only_fields = fields
