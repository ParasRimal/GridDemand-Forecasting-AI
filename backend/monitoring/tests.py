from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from .models import DataMonitoring


class DataMonitoringTests(TestCase):
    def test_can_record_a_drift_only_retrain_decision(self):
        dm = DataMonitoring.objects.create(
            checked_at=timezone.now(),
            drift_detected=True,
            drifted_column_share=0.737,
            drifted_column_count=14,
            n_columns=19,
            performance_degraded=False,
            recent_mae=9.01,
            reference_mae=9.01,
            relative_increase=0.0002,
            retrain_recommended=True,
            reason="Retraining recommended: data drift detected (74% of 19 features drifted, threshold 50%).",
        )
        self.assertTrue(dm.retrain_recommended)
        self.assertFalse(dm.performance_degraded)

    def test_can_record_a_no_action_decision(self):
        dm = DataMonitoring.objects.create(
            checked_at=timezone.now(),
            drift_detected=False,
            drifted_column_share=0.1,
            drifted_column_count=2,
            n_columns=19,
            performance_degraded=False,
            retrain_recommended=False,
        )
        self.assertFalse(dm.retrain_recommended)

    def test_default_ordering_is_newest_check_first(self):
        from datetime import timedelta
        now = timezone.now()
        older = DataMonitoring.objects.create(
            checked_at=now - timedelta(days=1), drift_detected=False, drifted_column_share=0.0,
            drifted_column_count=0, n_columns=19, performance_degraded=False, retrain_recommended=False,
        )
        newer = DataMonitoring.objects.create(
            checked_at=now, drift_detected=True, drifted_column_share=0.737,
            drifted_column_count=14, n_columns=19, performance_degraded=False, retrain_recommended=True,
        )
        self.assertEqual(list(DataMonitoring.objects.all()), [newer, older])


class DataMonitoringAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.dm = DataMonitoring.objects.create(
            checked_at=timezone.now(), drift_detected=True, drifted_column_share=0.737,
            drifted_column_count=14, n_columns=19, performance_degraded=False, retrain_recommended=True,
        )

    def test_list_endpoint_returns_created_checks(self):
        response = self.client.get("/api/monitoring/")
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"] if "results" in response.json() else response.json()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["retrain_recommended"])

    def test_write_methods_are_not_allowed(self):
        response = self.client.post("/api/monitoring/", {
            "checked_at": timezone.now().isoformat(), "drift_detected": False,
            "drifted_column_share": 0.0, "drifted_column_count": 0, "n_columns": 19,
            "performance_degraded": False, "retrain_recommended": False,
        })
        self.assertEqual(response.status_code, 405)
