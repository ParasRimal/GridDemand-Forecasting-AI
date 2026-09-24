"""Central configuration for the GridPredict ML pipeline.

Every decision made during data inspection lives here, so no other file
needs to hardcode these values.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
# This file is ml_service/app/config.py -> project root is 2 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "weather and load dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ahmedabad_processed.csv"

# ------------------------------------------------------------- dataset
# Primary dataset: Ahmedabad hourly weather + electric load (Nov 2021 - Nov 2022).
# PowerLoad_Dataset.csv was rejected as primary: its load is statistically
# indistinguishable from random noise (see notebooks/inspect_signal.py).
TARGET_COLUMN = "Electric Load (MW)"
DATE_PARTS = ["YEAR", "Month", "Day", "Hour"]
WEATHER_COLUMNS = ["irradiance", "temperature", "dewpoint", "specific humidity", "wind speed"]

# ------------------------------------------------------------ cleaning
# Both rules turn bad values into NaN. Raw CSV is never edited, and target
# values are never invented by interpolation.
IRRADIANCE_MISSING_CODE = -999.0   # sensor "no measurement" code (24 hours on 2022-01-07/08)
MIN_VALID_LOAD_MW = 0.0            # load <= this is treated as a meter dropout (79 rows)

# ------------------------------------------------------------ forecast
# Day-ahead forecasting: predict the load 24 hours after the moment of forecast.
FORECAST_HORIZON_HOURS = 24
LAG_HOURS = [24, 48, 168]          # 1 day, 2 days, 1 week

# Leakage guard: a lag shorter than the horizon would use the future.
assert all(lag >= FORECAST_HORIZON_HOURS for lag in LAG_HOURS), (
    "Every lag must be >= FORECAST_HORIZON_HOURS, otherwise the future leaks into the features."
)

# --------------------------------------------------- chronological split
# Boundaries are inclusive. No shuffling, and the test set is the latest period.
TRAIN_END = "2022-06-30 23:00:00"
VALIDATION_END = "2022-08-31 23:00:00"
# Test = everything after VALIDATION_END.

# ------------------------------------------------------------- holidays
# Ahmedabad is in Gujarat, India. Holidays are known in advance, so no leakage.
HOLIDAY_COUNTRY = "IN"
HOLIDAY_SUBDIVISION = "GJ"

# -------------------------------------------------------------- rolling
# Windows END at (target hour - horizon), so they only use loads known at forecast time.
ROLLING_WINDOWS_HOURS = [24, 168]   # last day, last week
ROLLING_MIN_FRACTION = 0.5          # a window needs >= 50% of its hours to have data

# -------------------------------------------------------------- weather
# Decision: use the ACTUAL weather at the target hour. In production this stands
# in for a weather forecast, so real-world accuracy would be somewhat lower.
# Phase 2 compares models with, without, and with lagged weather.
WEATHER_ASSUMPTION = "actual_at_target_hour"
