"""Sync backend/model_registry's ModelVersion table from the ML pipeline's
real models/registry.json (the actual promotion history from Phase 3).
"""
import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.utils.dateparse import parse_datetime
from django.utils.timezone import make_aware, is_naive

from model_registry.models import ModelVersion

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"


class Command(BaseCommand):
    help = "Sync ModelVersion rows from the real models/registry.json"

    def handle(self, *args, **options):
        if not REGISTRY_PATH.exists():
            self.stderr.write(f"registry.json not found at {REGISTRY_PATH}")
            return

        with open(REGISTRY_PATH) as f:
            registry = json.load(f)

        ModelVersion.objects.all().delete()

        production = registry["production"]
        prod_time = production.get("promoted_at")
        prod_dt = parse_datetime(prod_time) if prod_time else None
        if prod_dt is None:
            # Baseline was never "promoted" via the gate; use the earliest history entry's time.
            first_check = registry["history"][0]["timestamp"] if registry["history"] else None
            prod_dt = parse_datetime(first_check) if first_check else None
        if prod_dt and is_naive(prod_dt):
            prod_dt = make_aware(prod_dt)

        ModelVersion.objects.create(
            model_name=production["name"],
            model_type=production["type"],
            status="production",
            is_current_production=True,
            mae=production.get("mae"),
            rmse=(production.get("metrics") or {}).get("rmse"),
            mape=(production.get("metrics") or {}).get("mape"),
            r2=(production.get("metrics") or {}).get("r2"),
            reason="Default production model (naive baseline, never displaced).",
            evaluated_at=prod_dt,
        )

        for entry in registry["history"]:
            dt = parse_datetime(entry["timestamp"])
            if dt and is_naive(dt):
                dt = make_aware(dt)
            metrics = entry.get("challenger_metrics", {})
            ModelVersion.objects.create(
                model_name=entry["challenger"],
                model_type="candidate",
                status="production" if entry["promoted"] else "rejected",
                is_current_production=False,
                mae=metrics.get("mae"),
                rmse=metrics.get("rmse"),
                mape=metrics.get("mape"),
                r2=metrics.get("r2"),
                reason=entry["reason"],
                evaluated_at=dt,
            )

        count = ModelVersion.objects.count()
        self.stdout.write(self.style.SUCCESS(f"Synced {count} ModelVersion rows from registry.json"))
