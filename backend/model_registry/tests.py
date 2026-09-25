from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from .models import ModelVersion


class ModelVersionTests(TestCase):
    def test_rejected_candidate_can_be_recorded_with_only_mae(self):
        mv = ModelVersion.objects.create(
            model_name="xgboost_full",
            model_type="candidate",
            status="rejected",
            mae=10.67,
            reason="xgboost_full did not beat production by more than 0.3 MW (improvement was -1.66 MW).",
            evaluated_at=timezone.now(),
        )
        self.assertIsNone(mv.rmse)
        self.assertFalse(mv.is_current_production)

    def test_baseline_can_be_current_production(self):
        mv = ModelVersion.objects.create(
            model_name="naive_lag_24",
            model_type="baseline",
            status="production",
            is_current_production=True,
            mae=9.01,
            evaluated_at=timezone.now(),
        )
        self.assertTrue(mv.is_current_production)

    def test_our_real_registry_history_can_be_recorded_as_two_rows(self):
        ModelVersion.objects.create(
            model_name="naive_lag_24", model_type="baseline", status="production",
            is_current_production=True, mae=9.01, evaluated_at=timezone.now(),
        )
        ModelVersion.objects.create(
            model_name="xgboost_full", model_type="candidate", status="rejected",
            mae=10.67, evaluated_at=timezone.now(),
        )
        ModelVersion.objects.create(
            model_name="random_forest_no_season", model_type="candidate", status="rejected",
            mae=10.40, evaluated_at=timezone.now(),
        )
        self.assertEqual(ModelVersion.objects.count(), 3)
        current = ModelVersion.objects.filter(is_current_production=True)
        self.assertEqual(current.count(), 1)
        self.assertEqual(current.first().model_name, "naive_lag_24")


class ModelVersionAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.mv = ModelVersion.objects.create(
            model_name="naive_lag_24", model_type="baseline", status="production",
            is_current_production=True, mae=9.01, evaluated_at=timezone.now(),
        )

    def test_list_endpoint_returns_created_versions(self):
        response = self.client.get("/api/model-versions/")
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"] if "results" in response.json() else response.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model_name"], "naive_lag_24")

    def test_write_methods_are_not_allowed(self):
        response = self.client.post("/api/model-versions/", {
            "model_name": "x", "model_type": "candidate", "status": "rejected",
            "evaluated_at": timezone.now().isoformat(),
        })
        self.assertEqual(response.status_code, 405)
