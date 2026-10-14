# Paris air-quality forecasting

Reproducible forecasts for the [X/HEC Paris air-quality challenge](https://www.kaggle.com/competitions/x-hec-ts-2025-predicting-air-quality-in-paris). The task is to predict 504 hourly values for NO₂, CO, O₃, PM10 and PM2.5. The challenge scores the mean of five pollutant-specific MAEs. It closed on **October 15, 2025**, so this repository reports local backtests rather than a current leaderboard rank.

## Approach

The initial experiments compare a seasonal median, a calendar-and-history LightGBM model, and a weather-aware LightGBM model. Each model forecasts the entire 21-day window directly. Pollution lag features begin at 504 hours, so every value they read is known at the forecast origin. ERA5 weather for Paris comes from [Open-Meteo's historical weather API](https://open-meteo.com/en/docs/historical-weather-api); the challenge explicitly permits weather for the test period as if perfectly forecast. No pollutant measurements from the test period are used.

Four rolling-origin backtests use complete 504-hour windows. Missing historical pollutant labels are omitted from each pollutant's MAE. Results are local validation estimates, not hidden-test results.

| Method | Mean five-pollutant MAE |
| --- | ---: |
| Seasonal median | 7.255 |
| LightGBM, calendar and safe pollutant lags | 6.875 |
| LightGBM, calendar, lags and ERA5 weather | **4.885** |

The weather model improves the local mean by about 33% relative to the seasonal median. [Per-fold and per-pollutant results](results/backtest.json) show where it helps and where errors remain high, especially O₃ and NO₂. Short weather lags improved all four backtests relative to the initial weather model (4.925), but this feature choice used those same windows. The validation windows are not independent and the result does not establish leaderboard performance.

## Reproduce

Python 3.10+ is required. Download the competition ZIP from Kaggle and put it at `data/raw/competition.zip`; raw competition data and weather caches are excluded from Git.

```bash
python -m pip install -e ".[dev]"
paris-air fetch-weather
paris-air backtest
paris-air submit --method lightgbm_weather
python -m pytest -q
```

`fetch-weather` downloads only weather variables from Open-Meteo's ERA5 archive for Paris and caches them in `data/cache/`. The backtest writes a JSON report to `outputs/backtest.json`; the submission command writes a 504-row CSV to `outputs/submission.csv`. Both directories are ignored by Git. The script checks the competition's exact column order and test IDs before writing a submission.

The four validation cutoffs are specified in [`experiment.py`](src/paris_air_quality/experiment.py). Model settings and feature definitions are in [`models.py`](src/paris_air_quality/models.py) and [`features.py`](src/paris_air_quality/features.py). The archived [backtest report](results/backtest.json) includes input file hashes.

## Limits

- The competition is closed; a new first-place placement cannot be claimed.
- Historical ERA5 is treated as perfectly known future weather, which follows the challenge's stated allowance but is optimistic for operational forecasting.
- The provided pollutant series has long missing intervals, particularly CO in 2021. Scores use available validation labels only.
- The timestamps are continuous hourly strings without explicit timezone information. Weather is aligned to GMT; this assumption should be checked before any operational use.
- Feature choices were informed by the reported validation windows. An untouched holdout would be needed for an unbiased estimate of the final model-selection step.
- No model is judged by the hidden test labels in this repository.
