# GridPredict

**Automated day-ahead electricity demand forecasting, model selection, drift monitoring and safe model promotion.**

GridPredict forecasts the electricity load of a city 24 hours ahead. It trains several models, compares them fairly against a naive baseline, tracks data drift, and only replaces the production model when a challenger is clearly better. It is a full pipeline (data engineering, forecasting, MLOps, backend, frontend, containers), not a notebook.

The data is hourly weather and electric load from Ahmedabad, India (1 Nov 2021 to 1 Nov 2022).

## Table of contents

1. Headline result
2. Problem statement
3. Architecture
4. Technology stack
5. Dataset
6. Data cleaning
7. Feature engineering
8. Models and evaluation methodology
9. Model selection
10. Promotion gate and production model
11. Monitoring and retraining
12. API reference
13. Dashboard
14. Installation and configuration
15. Running the project
16. Testing
17. Project structure
18. Git workflow
19. Known limitations
20. Future improvements

## 1. Headline result (read this first)

Random Forest, XGBoost and LightGBM were compared against a naive baseline: predict that tomorrow's load at a given hour equals today's load at the same hour (`lag_24`).

All three models beat the baseline on the validation period (Jul-Aug 2022). On the untouched test period (Sep-Nov 2022) the picture reversed:

| Test set (Sep-Nov 2022) | MAE (MW) | RMSE (MW) | R2 |
|---|---|---|---|
| Baseline, same hour yesterday | **9.01** | **12.54** | **0.713** |
| Random Forest, no season features | 10.40 | 13.24 | 0.680 |
| XGBoost | 10.67 | 13.42 | 0.671 |

No trained model beat the baseline on unseen data, so the promotion gate kept the baseline in production. This is the system working as intended: it refuses to deploy a model that would make forecasts worse. The first pick (XGBoost) won on a single two-month validation window and then lost on the test set, which is why model selection was rebuilt around a rolling backtest (section 9). The full narrative is in `docs/MODELING_FINDINGS.md` and on the dashboard's About page.

## 2. Problem statement

Grid operators must commit generation a day ahead, so the useful question is: given what is known now, what will demand be at this hour tomorrow? Two things make this hard to do honestly:

- **Leakage.** A model that sees recent hours (`lag_1`) looks excellent in testing and is useless when forecasting a day ahead. Every feature here is restricted to information available 24 hours before the target hour.
- **Weak baselines look strong.** Electricity load repeats daily, so "same hour yesterday" is hard to beat. A model is only worth deploying if it beats that.

GridPredict's goal is to make these checks automatic and enforced in code, not left to discipline.

## 3. Architecture

    Browser
      |
      v
    React dashboard (nginx, port 5173)
      |                          |
      v                          v
    Django REST API (8001)     FastAPI ML service (8000)
      |        |                 |            |
      |        |                 |            +--> Redis (prediction cache)
      |        |                 +--> production model (models/registry.json)
      |        |
      |        +--> SQLite (ModelVersion, DataMonitoring, Forecast)
      |
      +--> Celery worker <--- Celery beat (daily schedule)
               |
               +--> Redis (broker) and the ML monitoring code

| Component | Port | Responsibility |
|---|---|---|
| FastAPI (`ml_service/`) | 8000 | Loads the production model, serves predictions, caches them in Redis |
| Django + DRF (`backend/`) | 8001 | Stores model history and monitoring checks, read-only REST API, admin |
| Celery worker | none | Runs the drift and performance check |
| Celery beat | none | Schedules that check once a day (stored in the database) |
| Redis | 6379 | Celery broker and FastAPI prediction cache |
| React + Vite + nginx (`frontend/`) | 5173 | Eight-page dashboard |

Training and evaluation code lives in `ml_service/app/` and is shared: FastAPI, Django and Celery import it instead of duplicating logic.

## 4. Technology stack

| Area | Tools |
|---|---|
| Data and ML | Python 3.12, pandas, NumPy, scikit-learn, XGBoost, LightGBM, joblib, holidays |
| Monitoring | Evidently 0.7 |
| ML API | FastAPI, Uvicorn, Pydantic |
| Backend | Django 6, Django REST Framework, django-cors-headers |
| Automation | Celery 5, django-celery-beat, Redis |
| Frontend | React 19, Vite 8, Tailwind CSS 4, Recharts, React Router, react-markdown |
| Delivery | Docker, Docker Compose, nginx |
| Testing | pytest, Django test runner |

## 5. Dataset

Two datasets were available. They come from different sources, with different units and locations, so they were never merged.

**`weather and load dataset.csv` (primary).** Ahmedabad, 8,779 hourly rows, no missing hours. Columns: `YEAR, Month, Day, Hour, irradiance, temperature, dewpoint, specific humidity, wind speed, Electric Load (MW)`. It has strong structure:

- Daily cycle: about 58 MW at 5 am, about 89 MW at 3 pm.
- Weekly cycle: Sunday lowest, Thursday to Saturday highest.
- Seasonal cycle: Nov-Jan lowest, Apr-Jun and Sep highest.
- Correlation with load: temperature 0.66, irradiance 0.37.
- Autocorrelation: 0.94 at 1 hour, 0.82 at 24 hours, 0.67 at 168 hours.

**`PowerLoad_Dataset.csv` (rejected).** 10,000 rows, irregularly spaced (only about 1 hour in 5 present). Its weather columns are uniformly distributed over round ranges (for example temperature exactly -5 to 45), the hourly averages are flat at about 500 kW, correlations with every weather variable are below 0.02, and autocorrelation at every lag is near zero. That is consistent with synthetic noise, so no model can learn from it. It stays in `data/raw/` only as documented evidence.

The inspection scripts are in `notebooks/`.

## 6. Data cleaning

Cleaning rules live in `ml_service/app/config.py` and are applied in `data_loader.py`. The raw CSV is never modified.

- `irradiance == -999` is a sensor "no measurement" code (24 consecutive hours on 7-8 Jan 2022). It becomes NaN.
- `Electric Load (MW) <= 0` (79 rows in 22 short runs, almost all exactly 0) looks like meter dropouts. It becomes NaN.
- Missing values are never interpolated, because inventing target values would put fake numbers into training.
- The loader also rejects duplicate timestamps, sorts unsorted data, re-inserts missing hours as NaN rows so lags stay exact, and checks required columns.
- The largest loads (up to 187 MW, around a hot spell on 11 Apr 2022) come in smooth runs and are kept as real demand.

## 7. Feature engineering

The model predicts the load at a target hour `t`, and the forecast is made at `t - 24h`. Every feature respects that.

| Group | Features | Why |
|---|---|---|
| Calendar (7) | hour, day_of_week, is_weekend, month, quarter, week_of_year, is_holiday | Daily, weekly and seasonal cycles; offices close on holidays. Known in advance, so no leakage. Holidays use India plus Gujarat (`holidays` package). |
| Weather (5) | irradiance, temperature, dewpoint, specific humidity, wind speed | Temperature drives cooling demand. Uses the actual value at the target hour (see limitations). |
| Lags (3) | lag_24, lag_48, lag_168 | Load 1 day, 2 days and 1 week earlier. Looked up by timestamp, not row position. |
| Rolling (4) | rolling_mean_24, rolling_std_24, rolling_mean_168, rolling_std_168 | Recent level and variability. Windows end at t-24, never at t. |

`day` and `year` are deliberately excluded because with one year of data they let tree models memorize dates.

Leakage protection is enforced, not assumed:

- A lag shorter than the forecast horizon raises an error, and config has an assertion.
- Lags use timestamp lookup, and a missing earlier hour gives NaN instead of a wrong value.
- An end-to-end test overwrites the load from t-23h to t with 9999 and asserts the features at t do not change.

The first 168 hours (no complete lag history) and rows with a missing target are dropped last, after lags and rolling windows are computed on the full timeline. Result: 8,532 rows and 19 features.

## 8. Models and evaluation methodology

**Models:** `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, built by name in `training/model_factory.py` with fixed, untuned settings (seed 42) stored in `config.py`. All three handle NaN in features natively.

**Chronological split (no random `train_test_split`):**

| Part | Period | Rows | Use |
|---|---|---|---|
| Train | 8 Nov 2021 - 30 Jun 2022 | 5,583 | Fit models |
| Validation | 1 Jul - 31 Aug 2022 | 1,468 | Compare models |
| Test | 1 Sep - 1 Nov 2022 | 1,481 | Final report only |

**Metrics** (`evaluation/metrics.py`, verified against scikit-learn):

- **MAE**: average absolute error in MW. Easiest to interpret, and the main selection metric.
- **RMSE**: like MAE but punishes large misses more.
- **MAPE**: average error as a percentage of actual load.
- **R2**: share of the load's variation explained. 1 is perfect, 0 is no better than the mean.

The baseline and every model are scored on exactly the same rows (those where `lag_24` exists).

Validation results (Jul-Aug 2022, 1,448 rows):

| | MAE | RMSE | MAPE | R2 |
|---|---|---|---|---|
| Baseline lag_24 | 9.25 | 13.29 | 12.8% | 0.220 |
| Random Forest | 8.52 | 11.58 | 12.6% | 0.407 |
| XGBoost | 8.47 | 11.46 | 12.4% | 0.420 |
| LightGBM | 8.61 | 11.67 | 12.7% | 0.399 |

## 9. Model selection

**Version 1 (failed).** Pick the model with the lowest validation MAE among those beating the baseline. XGBoost won by 0.05 MW and then lost to the baseline on the test set by 1.66 MW. Two months is too short a window to trust.

**Version 2 (current).** A rolling-origin backtest (`training/backtest.py`) trains on everything before each month, with a 24-hour gap, and tests on that month, for 7 monthly folds (Feb-Aug 2022). It never touches the test set. The rule, written before the results were seen:

1. Eligible only if it beats the baseline in at least 70% of folds and has a lower mean MAE.
2. Lowest mean MAE wins.
3. Candidates within 0.1 MW of the best are tied, and the one that beats the baseline in the most folds wins.

| Backtest, 7 folds | Folds beating baseline | Mean MAE |
|---|---|---|
| Baseline | - | 9.02 |
| Random Forest | 6 / 7 | 8.54 |
| XGBoost | 4 / 7 | 8.72 |
| LightGBM | 2 / 7 | 9.15 |
| Random Forest without season columns | **6 / 7** | **8.44** |

Dropping `month`, `quarter` and `week_of_year` helped Random Forest slightly. Predicting the change from yesterday's load instead of the raw load did not help consistently. That candidate was trained on train plus validation and scored once on the test set, as a labelled report. It still lost (MAE 10.40 vs 9.01).

A likely reason: the test period has a higher average load than any training period, and tree models cannot extrapolate beyond levels they have seen, while the baseline copies the current level automatically. This is an explanation, not a proven cause.

Top features of the saved Random Forest: lag_24 (0.437), lag_48 (0.155), temperature (0.123), rolling_mean_24 (0.053).

## 10. Promotion gate and production model

`models/registry.json` records what is in production and every promotion decision. Before anything is promoted, production is the naive baseline. A challenger replaces production only if it beats the current production MAE by more than 0.3 MW (`PROMOTION_MIN_MAE_IMPROVEMENT_MW`) on a shared evaluation. Every attempt, accepted or not, is appended to the history.

Current state: the baseline is in production, and both trained candidates are recorded as rejected (improvements of -1.66 MW and -1.39 MW). The FastAPI service reads this file at startup through a `ProductionModel` wrapper, so `/predict` works the same whether the baseline or a trained model is active. Trained models are saved as `models/<name>_<UTC timestamp>/model.joblib` plus `metadata.json`, and saving never overwrites an older model.

## 11. Monitoring and retraining

Retraining is based on evidence, never on "the data changed".

- **Drift** (`monitoring/drift.py`): Evidently compares feature distributions between a reference and a current dataset. Evidently picks the test per column based on sample size. On the real train vs test data (thousands of rows) it used Wasserstein distance for numeric columns and Jensen-Shannon distance for binary ones, where a larger value means more drift. On small samples (under about 1,000 rows) it switches to p-value tests such as K-S, where a smaller value means more drift. The code handles each method's comparison direction explicitly, and raises an error for an unknown method. The dataset counts as drifted when more than 50% of features drifted (`DRIFT_SHARE_THRESHOLD`).
- **Performance** (`monitoring/performance.py`): degraded when recent MAE exceeds the reference MAE by more than 15% (`PERFORMANCE_DEGRADATION_THRESHOLD`).
- **Decision** (`monitoring/retraining.py`): retrain when drift and/or degradation is present, with a written reason.
- **Automation:** a Celery task runs the decision daily (a schedule created by a migration and editable in Django admin) and stores each result as a `DataMonitoring` row.

Real result on the train vs test comparison: 14 of 19 features drifted (73.7%), no performance degradation, retraining recommended. This matches the model behaviour above.

## 12. API reference

### FastAPI ML service (port 8000, interactive docs at `/docs`)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness and current production model |
| GET | `/model` | Production model details and its required features |
| GET | `/metrics` | Recorded metrics of the production model |
| POST | `/predict` | Forecast for one target hour |
| POST | `/predict/batch` | Many forecasts, each validated independently |

The required fields depend on the production model. The baseline needs only `lag_24`, and a trained model needs its full feature list. A missing field returns HTTP 422 naming it. Example:

    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"timestamp": "2022-09-15T14:00:00", "lag_24": 85.3}'

Responses are cached in Redis for 300 seconds, keyed by a hash of the inputs. If Redis is down, caching is skipped and predictions still work.

### Django REST API (port 8001, all read-only)

| Path | Description |
|---|---|
| `/api/forecasts/` | Stored forecasts (currently empty) |
| `/api/model-versions/` | Promotion history |
| `/api/monitoring/` | Drift and performance checks |
| `/api/historical-demand/` | Daily min, average and max load from the processed dataset |
| `/api/feature-importance/` | Importances from the saved Random Forest |
| `/api/modeling-findings/` | Text of `docs/MODELING_FINDINGS.md` |
| `/health/redis/` | Redis connectivity |
| `/admin/` | Django admin |

## 13. Dashboard

| Page | Shows |
|---|---|
| Dashboard | Production model, latest drift check, model history |
| Try a Forecast | Form generated from the production model's required fields |
| Model History | MAE chart and the promotion decision log |
| System Status | Live status of Django, FastAPI and Redis |
| Monitoring History | Every drift and performance check |
| Historical Demand | Daily load chart showing the seasonal cycle |
| Feature Importance | What the Random Forest relies on |
| About | The modelling findings document |

## 14. Installation and configuration

Requirements: Python 3.12, Node 22, Docker with Compose, and the raw CSV files in `data/raw/` (they are not in Git).

    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env

Environment variables (`.env`, never committed):

| Variable | Purpose |
|---|---|
| `DEBUG`, `DJANGO_SECRET_KEY` | Django settings. Generate a real secret key. |
| `REDIS_HOST`, `REDIS_PORT` | Redis connection. Use `127.0.0.1`, not `localhost`. |
| `ML_API_URL`, `DJANGO_API_URL` | Service URLs (documentation for now; the frontend hardcodes them) |
| `SQLITE_PATH`, `ALLOWED_HOSTS` | Set by Docker Compose |
| `MLFLOW_TRACKING_URI`, `DATABASE_URL` | Present in `.env.example` but not used (see limitations) |

On the development machine `localhost` failed to resolve reliably for Redis while `127.0.0.1` worked, so `127.0.0.1` is used everywhere.

## 15. Running the project

### Step A: regenerate the files that Git ignores

    python -c "from ml_service.app.preprocessing.pipeline import build_and_save; build_and_save()"
    python -m ml_service.app.training.train_pipeline
    python -m ml_service.app.training.train_candidate

These create the processed CSV and the trained model folders. The Docker build copies them from disk, so do this first.

### Step B, option 1: Docker (recommended)

    sudo systemctl stop redis-server      # only if a native Redis is running
    docker compose up -d --build
    docker compose exec django python manage.py sync_registry
    docker compose exec django python -c "import django,os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings'); django.setup(); from monitoring.tasks import run_drift_and_performance_check as t; t.delay()"

Open http://127.0.0.1:5173. SQLite lives in the `django_data` volume and survives `docker compose down`. Never run `docker compose down -v`, because it deletes the database.

### Step B, option 2: run natively

    sudo systemctl start redis-server
    uvicorn ml_service.app.main:app --port 8000
    cd backend && python manage.py migrate && python manage.py sync_registry && python manage.py runserver 127.0.0.1:8001
    cd backend && celery -A config worker --loglevel=info
    cd backend && celery -A config beat --loglevel=info
    cd frontend && npm install && npm run dev

Run each command in its own terminal.

## 16. Testing

    python -m pytest tests/ -q
    cd backend && python manage.py test

175 pytest tests and 22 Django tests, all passing.

- **Data:** duplicate, unsorted and missing timestamps, missing columns, cleaning rules.
- **Features:** calendar, lag and rolling correctness, plus end-to-end leakage tests.
- **Models:** training, seeded reproducibility, NaN handling, metrics against scikit-learn.
- **Selection:** the selection rule, backtest fold boundaries, promotion gate.
- **Monitoring:** drift detection and the retraining decision.
- **API and cache:** every endpoint, invalid input, batch errors, Redis fail-open behaviour.
- **Django:** models, read-only endpoints, the Celery task, the `sync_registry` command.

## 17. Project structure

    backend/         Django project: forecasts, model_registry, monitoring, users apps
    ml_service/app/
        config.py            every threshold and setting in one place
        preprocessing/       loader, calendar, lag and rolling features, pipeline
        training/            split, baseline, models, trainer, backtest, registry
        evaluation/          metrics and selection rules
        monitoring/          drift, performance, retraining decision
        caching/             Redis prediction cache
        inference/           production model wrapper
        routes/, schemas/    FastAPI endpoints and Pydantic models
        main.py              FastAPI entrypoint
    frontend/        React + Vite + Tailwind dashboard
    models/          model artifacts (ignored) and registry.json (tracked)
    data/            raw, processed, external (ignored)
    docs/            MODELING_FINDINGS.md
    notebooks/       data inspection scripts
    tests/           pytest suite
    Dockerfile.fastapi, Dockerfile.django, Dockerfile.frontend, docker-compose.yml

## 18. Git workflow

Small commits with conventional prefixes (`feat:`, `fix:`, `test:`, `docs:`, `style:`, `chore:`), one logical change each, tests run before every commit. Secrets (`.env`), data, model binaries, the SQLite file and Celery Beat's schedule file are gitignored.

## 19. Known limitations

- **One year of data.** Seasonal effects cannot be separated from trends, and the test season (Sep-Oct) was never seen in training.
- **Weather is the actual value at the target hour**, standing in for a weather forecast. Real deployments would see somewhat lower accuracy.
- **The test set was viewed more than once.** After XGBoost lost, a second candidate was chosen by backtest and scored on it as a labelled report. It was not used to pick or tune anything, but the test set is no longer a clean unseen evaluation.
- **MLflow is not integrated.** Run details are stored in each model folder's `metadata.json` and in `models/registry.json`.
- **The daily drift check compares the static train and test split.** There is no live data stream yet, so the same comparison repeats.
- **The `Forecast` table is empty.** Predictions are cached in Redis but not saved to Django.
- **Django uses `runserver`**, which is for development. Use gunicorn for real deployment.
- **Docker images are about 3 GB** because they install the whole `requirements.txt`.
- **The Docker build copies gitignored files** from disk, so Step A must be run first.
- **The frontend hardcodes service URLs** (127.0.0.1:8000 and 8001), which works because the browser reaches published ports.
- **The `users` app is an empty placeholder.** Django's built-in user model is used for admin only, with no API authentication.

## 20. Future improvements

MLflow experiment tracking, several years of data, a level-aware model (for example predicting the change from a seasonal baseline, or blending with the baseline), saving forecasts and filling in actuals so real-world error can be measured, a live data feed for the drift check, per-service requirements files, API authentication, and gunicorn behind a reverse proxy.
