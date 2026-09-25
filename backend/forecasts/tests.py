from django.test import TestCase
from django.utils import timezone

from .models import Forecast


class ForecastModelTests(TestCase):
    def test_forecast_can_be_created_without_an_actual_value_yet(self):
        f = Forecast.objects.create(
            timestamp=timezone.now(),
            predicted_load_mw=85.3,
            model_type="baseline",
            model_name="naive_lag_24",
        )
        self.assertIsNone(f.actual_load_mw)
        self.assertIsNotNone(f.created_at)

    def test_actual_load_can_be_filled_in_later(self):
        f = Forecast.objects.create(
            timestamp=timezone.now(),
            predicted_load_mw=85.3,
            model_type="baseline",
            model_name="naive_lag_24",
        )
        f.actual_load_mw = 88.1
        f.save()
        f.refresh_from_db()
        self.assertEqual(f.actual_load_mw, 88.1)

    def test_default_ordering_is_newest_timestamp_first(self):
        from datetime import timedelta
        now = timezone.now()
        older = Forecast.objects.create(timestamp=now - timedelta(hours=2), predicted_load_mw=1, model_type="baseline", model_name="x")
        newer = Forecast.objects.create(timestamp=now, predicted_load_mw=2, model_type="baseline", model_name="x")
        self.assertEqual(list(Forecast.objects.all()), [newer, older])
