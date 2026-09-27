"""Tests for the sync_registry management command, using a fake registry.json
so this doesn't depend on the real project's models/registry.json existing
or having any particular content."""
import json

from django.core.management import call_command
from django.test import TestCase

from . import models
from .management.commands import sync_registry


FAKE_REGISTRY = {
    "production": {
        "type": "baseline",
        "name": "naive_lag_24",
        "mae": 9.01,
        "promoted_at": None,
    },
    "history": [
        {
            "timestamp": "2026-09-25T01:49:41.328169+00:00",
            "challenger": "xgboost_full",
            "challenger_mae": 10.67,
            "challenger_metrics": {"mae": 10.67, "rmse": 13.42, "mape": 13.8, "r2": 0.671},
            "promoted": False,
            "reason": "xgboost_full did not beat production by more than 0.3 MW.",
        },
    ],
}


class SyncRegistryCommandTests(TestCase):
    def _write_fake_registry(self, tmp_path, monkeypatch, data):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(data))
        monkeypatch.setattr(sync_registry, "REGISTRY_PATH", path)
        return path

    def test_command_creates_expected_rows(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps(FAKE_REGISTRY))
            with patch.object(sync_registry, "REGISTRY_PATH", path):
                call_command("sync_registry")

        self.assertEqual(models.ModelVersion.objects.count(), 2)

        production = models.ModelVersion.objects.get(is_current_production=True)
        self.assertEqual(production.model_name, "naive_lag_24")
        self.assertEqual(production.status, "production")
        self.assertEqual(production.mae, 9.01)

        rejected = models.ModelVersion.objects.get(model_name="xgboost_full")
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(rejected.mae, 10.67)
        self.assertEqual(rejected.rmse, 13.42)
        self.assertFalse(rejected.is_current_production)

    def test_command_is_idempotent_running_twice_does_not_duplicate(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps(FAKE_REGISTRY))
            with patch.object(sync_registry, "REGISTRY_PATH", path):
                call_command("sync_registry")
                call_command("sync_registry")

        self.assertEqual(models.ModelVersion.objects.count(), 2)

    def test_command_handles_a_missing_registry_file_gracefully(self):
        from pathlib import Path
        from unittest.mock import patch

        with patch.object(sync_registry, "REGISTRY_PATH", Path("/nonexistent/registry.json")):
            call_command("sync_registry")  # should not raise

        self.assertEqual(models.ModelVersion.objects.count(), 0)

    def test_exactly_one_row_is_marked_current_production(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps(FAKE_REGISTRY))
            with patch.object(sync_registry, "REGISTRY_PATH", path):
                call_command("sync_registry")

        current = models.ModelVersion.objects.filter(is_current_production=True)
        self.assertEqual(current.count(), 1)
