from django.db import models


class ModelVersion(models.Model):
    """A record of one promotion decision, mirroring an entry from
    ml_service/app/training/registry.py's registry.json (either the current
    `production` block, or one `history` entry).

    Metrics are nullable because registry.json itself sometimes only records
    `mae` (e.g. the baseline's `metrics` field is null; see /metrics in the
    FastAPI service).
    """

    STATUS_CHOICES = [
        ("production", "Production"),
        ("rejected", "Rejected"),
    ]

    model_name = models.CharField(max_length=100, help_text="e.g. 'naive_lag_24', 'random_forest'.")
    model_type = models.CharField(max_length=20, help_text="'baseline' or 'candidate'.")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    is_current_production = models.BooleanField(
        default=False,
        help_text="True for exactly one row: whatever registry.json currently lists as production."
    )

    mae = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)
    mape = models.FloatField(null=True, blank=True)
    r2 = models.FloatField(null=True, blank=True)

    reason = models.TextField(
        blank=True,
        help_text="The promotion-gate reason string, e.g. from registry.py's evaluate_promotion()."
    )
    evaluated_at = models.DateTimeField(help_text="When this decision was recorded in registry.json.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-evaluated_at"]

    def __str__(self) -> str:
        return f"ModelVersion({self.model_name}, {self.status}, mae={self.mae})"
