# GridPredict — Modeling Findings

This document summarizes the data investigation and model-selection process
for GridPredict's day-ahead electricity load forecast, and states the
current production decision honestly.

## 1. Dataset selection

Two candidate datasets were available. `PowerLoad_Dataset.csv` (10,000 rows,
2018-2023) was rejected as the primary dataset after inspection showed:

- Weather columns (temperature, humidity, wind speed) uniformly distributed
  over suspiciously round ranges, consistent with synthetic generation.
- No daily, weekly, or seasonal pattern in the load (hourly averages flat at
  ~500 across all 24 hours).
- Near-zero correlation (|r| < 0.02) between load and every weather variable.
- Near-zero autocorrelation at every tested lag (1h to 168h).

`weather and load dataset.csv` (Ahmedabad, India; 8,779 rows, hourly,
Nov 2021-Nov 2022) was selected instead, showing:

- A clear daily cycle (load ~58 MW at 5am, ~89 MW at 3pm).
- A clear weekly cycle (Sunday lowest, Thu-Sat highest).
- A clear seasonal cycle (Nov-Jan lowest, Apr-Jun/Sep highest).
- Strong correlation with temperature (r = 0.66) and irradiance (r = 0.37).
- Strong autocorrelation (lag 1h: 0.94, lag 24h: 0.82, lag 168h: 0.67).

**Limitation:** the dataset covers only one calendar year, so seasonal
effects and year-over-year trends cannot be separated.

## 2. Data cleaning

- `irradiance == -999` (a sensor "no data" code, 24 consecutive hours on
  2022-01-07/08) was set to NaN.
- `Electric Load (MW) <= 0` (79 rows in 22 short runs, likely meter
  dropouts) was set to NaN. No values were interpolated or invented.
- The underlying hourly timeline was otherwise complete (0 missing hours).

## 3. Forecast horizon and leakage prevention

The system forecasts **24 hours ahead** (day-ahead), the standard horizon
in grid operations. This was chosen deliberately over a 1-hour-ahead
forecast, which would trivially score well using `lag_1` but has little
operational value.

Every lag feature (`lag_24`, `lag_48`, `lag_168`) and rolling window
(`rolling_mean/std_24`, `rolling_mean/std_168`) is built to use only
information available 24 hours before the target hour. This is enforced by
an assertion in configuration and verified by automated end-to-end tests
that corrupt the "future" 23 hours of data and confirm the computed
features do not change.

Weather features use the **actual weather at the target hour**, which
stands in for a weather forecast. Real deployments would use a forecast,
which has its own error, so live accuracy would likely be somewhat lower
than reported here.

## 4. Baseline

Before any machine learning, two naive baselines were established:

| Baseline | Validation MAE | Validation R² |
|---|---|---|
| Same hour yesterday (`lag_24`) | 9.25 MW | 0.220 |
| Same hour last week (`lag_168`) | 12.36 MW | -0.187 |

"Same hour yesterday" is a strong baseline and became the bar every model
had to beat.

## 5. Model comparison

Random Forest, XGBoost, and LightGBM were trained with fixed, untuned
settings and scored on a chronological validation split (Jul-Aug 2022).

| Model | Validation MAE | Validation R² |
|---|---|---|
| Baseline | 9.25 | 0.220 |
| Random Forest | 8.52 | 0.407 |
| **XGBoost** | **8.47** | **0.420** |
| LightGBM | 8.61 | 0.399 |

XGBoost was initially selected by a rule requiring only the single best
validation MAE.

## 6. The single-window selection failed on the test set

XGBoost was trained on the full training set and scored **once** on the
held-out test period (Sep-Nov 2022):

| Test set | MAE | R² |
|---|---|---|
| Baseline | **9.01** | **0.713** |
| XGBoost | 10.67 | 0.671 |

**XGBoost lost to the baseline on every metric on unseen data**, despite
winning on the 2-month validation window. This showed that a single
validation window is not a reliable basis for model selection.

## 7. Rolling backtest and re-selection

A 7-fold rolling-origin backtest (expanding training window, one calendar
month held out per fold, Feb-Aug 2022, never touching the test set) was
built to get a more robust comparison:

| Model | Folds beating baseline | Mean MAE |
|---|---|---|
| **Random Forest** | **6 / 7** | **8.54** |
| XGBoost | 4 / 7 | 8.72 |
| LightGBM | 2 / 7 | 9.15 |

Two variants were also tested: dropping calendar features that only
identify the season (`month`, `quarter`, `week_of_year`), and predicting
the change from yesterday's load rather than the raw value. Random Forest
without the season columns was the best-performing eligible combination
(mean MAE 8.44, 6/7 folds), under a selection rule fixed *before* running
the comparison (eligible = beats baseline in >=70% of folds AND lower mean
MAE; ties within 0.1 MW broken by most folds beaten).

## 8. The re-selected candidate also failed on the test set

The backtest-selected candidate (Random Forest, no season features) was
trained on train+validation data and scored once, as a labeled report, on
the test set:

| Test set | MAE | R² |
|---|---|---|
| Baseline | **9.01** | **0.713** |
| Random Forest (no season) | 10.40 | 0.680 |

This candidate also lost to the baseline, though by a smaller margin than
XGBoost (1.39 MW vs 1.66 MW behind).

**Likely explanation:** the test period (Sep-Oct) has a higher average
load than any period the models were trained on, and tree-based models
cannot extrapolate beyond the value ranges they were trained on. The naive
baseline sidesteps this entirely because it simply copies a recent actual
value, which automatically tracks any level shift.

## 9. Current production decision

A promotion gate was built requiring a challenger to beat the *current*
production model's MAE by more than 0.3 MW on a shared evaluation. Applied
to both candidates against the baseline's test MAE (9.01):

- XGBoost: **not promoted** (would have made MAE 1.66 MW worse)
- Random Forest (no season): **not promoted** (would have made MAE 1.39 MW worse)

**As of this writing, the naive "same hour yesterday" baseline remains the
production model.** This is not a failure of the pipeline — it is the
pipeline correctly refusing to deploy a model that would degrade forecast
quality, exactly as intended by the "avoid replacing the production model
merely because a new model was trained" requirement.

## 10. What would likely help

With more data (multiple years, to give tree models exposure to a full
range of seasonal load levels and let the promotion decision be tested
across more than one held-out season) or an explicitly level-aware
architecture (e.g. modeling load relative to a slower-moving seasonal
baseline, or blending trained-model output with the naive baseline), a
genuine improvement over the baseline is plausible. Retraining
periodically as more real data accumulates, combined with the drift
monitoring built in the next phase, is the mechanism by which this project
is designed to keep re-testing that possibility over time.
