"""Create a daily periodic schedule for the drift/performance check.

Using a migration (rather than manual django admin clicks) keeps this
schedule reproducible and tracked in git.
"""
from django.db import migrations


def create_schedule(apps, schema_editor):
    IntervalSchedule = apps.get_model("django_celery_beat", "IntervalSchedule")
    PeriodicTask = apps.get_model("django_celery_beat", "PeriodicTask")

    schedule, _ = IntervalSchedule.objects.get_or_create(every=1, period="days")
    PeriodicTask.objects.get_or_create(
        name="Daily drift and performance check",
        task="monitoring.tasks.run_drift_and_performance_check",
        interval=schedule,
    )


def remove_schedule(apps, schema_editor):
    PeriodicTask = apps.get_model("django_celery_beat", "PeriodicTask")
    PeriodicTask.objects.filter(name="Daily drift and performance check").delete()


class Migration(migrations.Migration):
    dependencies = [
        ("monitoring", "0001_initial"),
        ("django_celery_beat", "0019_alter_periodictasks_options"),
    ]
    operations = [migrations.RunPython(create_schedule, remove_schedule)]
