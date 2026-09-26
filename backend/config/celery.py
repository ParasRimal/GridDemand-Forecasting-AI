"""Celery application for the GridPredict Django backend.

Run a worker with:
    celery -A config worker --loglevel=info
Run the beat scheduler with:
    celery -A config beat --loglevel=info
"""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
