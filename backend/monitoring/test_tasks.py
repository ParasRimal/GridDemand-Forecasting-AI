from django.test import TestCase, override_settings

from .models import DataMonitoring
from .tasks import run_drift_and_performance_check


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class RetrainingTaskTests(TestCase):
    def test_task_creates_a_data_monitoring_record(self):
        self.assertEqual(DataMonitoring.objects.count(), 0)
        result = run_drift_and_performance_check.delay()
        payload = result.get()

        self.assertEqual(DataMonitoring.objects.count(), 1)
        record = DataMonitoring.objects.first()
        self.assertEqual(record.id, payload["record_id"])

    def test_result_matches_our_known_real_world_finding(self):
        """Locks in the actual Task 61 finding: drift detected, no degradation, retrain recommended."""
        run_drift_and_performance_check.delay().get()
        record = DataMonitoring.objects.first()

        self.assertTrue(record.drift_detected)
        self.assertGreater(record.drifted_column_share, 0.5)
        self.assertFalse(record.performance_degraded)
        self.assertTrue(record.retrain_recommended)
