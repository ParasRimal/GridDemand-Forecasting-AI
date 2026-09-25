from django.db import models


class Forecast(models.Model):
    """A single day-ahead load prediction for one target hour.

    `actual_load_mw` starts null and is filled in later, once the real hour
    has passed and the true load is known -- this is what lets us later
    compute real-world accuracy, distinct from the offline test-set scores
    recorded in docs/MODELING_FINDINGS.md.
    """

    timestamp = models.DateTimeField(
        help_text="The target hour this forecast is for."
    )
    predicted_load_mw = models.FloatField(
        help_text="The forecasted load, in MW."
    )
    actual_load_mw = models.FloatField(
        null=True, blank=True,
        help_text="The real load once known; null until the target hour has passed."
    )
    model_type = models.CharField(
        max_length=20,
        help_text="'baseline' or 'candidate', matching ProductionModel.info()."
    )
    model_name = models.CharField(
        max_length=100,
        help_text="e.g. 'naive_lag_24' or 'random_forest'."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [models.Index(fields=["timestamp"])]

    def __str__(self) -> str:
        return f"Forecast({self.timestamp}, {self.model_name}={self.predicted_load_mw} MW)"
