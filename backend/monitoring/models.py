from django.db import models


class DataMonitoring(models.Model):
    """One record of a drift/degradation check, mirroring the dict returned by
    ml_service/app/monitoring/retraining.py's decide_retraining().
    """

    checked_at = models.DateTimeField(help_text="When this check was run.")

    drift_detected = models.BooleanField()
    drifted_column_share = models.FloatField(help_text="e.g. 0.737 for 73.7% of features drifted.")
    drifted_column_count = models.IntegerField()
    n_columns = models.IntegerField()

    performance_degraded = models.BooleanField()
    recent_mae = models.FloatField(null=True, blank=True)
    reference_mae = models.FloatField(null=True, blank=True)
    relative_increase = models.FloatField(
        null=True, blank=True,
        help_text="(recent_mae - reference_mae) / reference_mae."
    )

    retrain_recommended = models.BooleanField()
    reason = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-checked_at"]

    def __str__(self) -> str:
        return f"DataMonitoring({self.checked_at}, retrain={self.retrain_recommended})"
