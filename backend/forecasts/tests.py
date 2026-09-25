from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

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


class ForecastAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.f1 = Forecast.objects.create(
            timestamp=timezone.now(), predicted_load_mw=85.3,
            model_type="baseline", model_name="naive_lag_24",
        )

    def test_list_endpoint_returns_created_forecasts(self):
        response = self.client.get("/api/forecasts/")
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"] if "results" in response.json() else response.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["predicted_load_mw"], 85.3)

    def test_detail_endpoint_returns_one_forecast(self):
        response = self.client.get(f"/api/forecasts/{self.f1.id}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_name"], "naive_lag_24")

    def test_write_methods_are_not_allowed(self):
        response = self.client.post("/api/forecasts/", {
            "timestamp": timezone.now().isoformat(), "predicted_load_mw": 1.0,
            "model_type": "baseline", "model_name": "x",
        })
        self.assertEqual(response.status_code, 405)
