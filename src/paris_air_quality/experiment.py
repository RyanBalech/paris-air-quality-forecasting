"""Rolling-origin evaluation and submission generation."""
from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .data import TARGETS, competition_mae, load_competition
from .features import HORIZON_HOURS
from .models import direct_lightgbm, seasonal_median
from .weather import load_weather

FOLDS = (
    "2023-09-03 22", "2024-03-01 22", "2024-06-01 22", "2024-08-12 22",
)
METHODS = ("seasonal_median", "lightgbm_calendar", "lightgbm_weather")


def forecast(method: str, history: pd.DataFrame, future: pd.DatetimeIndex,
             weather: pd.DataFrame | None) -> pd.DataFrame:
    if method == "seasonal_median":
        return seasonal_median(history, future)
    if method == "lightgbm_calendar":
        return direct_lightgbm(history, future)
    if method == "lightgbm_weather":
        if weather is None:
            raise ValueError("Weather cache is required for lightgbm_weather")
        return direct_lightgbm(history, future, weather)
    raise ValueError(f"Unknown method: {method}")


def backtest(zip_path: Path, weather_path: Path, output: Path,
             folds: tuple[str, ...] = FOLDS) -> dict:
    train, _, _ = load_competition(zip_path)
    weather = load_weather(weather_path)
    records = []
    for cutoff_text in folds:
        cutoff = pd.Timestamp(cutoff_text)
        history = train.loc[:cutoff]
        actual = train.loc[cutoff + pd.Timedelta(hours=1):].iloc[:HORIZON_HOURS]
        if len(actual) != HORIZON_HOURS:
            raise ValueError(f"Incomplete 504-hour validation window after {cutoff}")
        for method in METHODS:
            prediction = forecast(method, history, actual.index, weather)
            metric = competition_mae(actual, prediction)
            records.append({"cutoff": cutoff_text, "method": method, **metric})
            print(f"{cutoff_text} {method}: {metric['score']:.4f}")
    summary = {}
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        summary[method] = {
            "mean_score": sum(r["score"] for r in rows) / len(rows),
            "per_target": {target: sum(r["per_target"][target] for r in rows) / len(rows)
                           for target in TARGETS},
        }
    report = {"metric": "mean of observed-label MAE for five pollutants",
              "horizon_hours": HORIZON_HOURS, "folds": records, "summary": summary,
              "competition_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
              "weather_cache_sha256": hashlib.sha256(weather_path.read_bytes()).hexdigest(),
              "model_settings": {"train_years": 3, "n_estimators": 500,
                                 "objective": "regression_l1", "random_state": 42},
              "weather_features": "current conditions, 3/6/24-hour lags, 6/24-hour means",
              "software": {"python": platform.python_version(),
                           **{name: version(name) for name in
                              ("numpy", "pandas", "scikit-learn", "lightgbm")}},
              "weather_source": "Open-Meteo ERA5 archive, Paris, GMT",
              "limitations": ["Historical reanalysis weather is treated as perfectly known, as permitted by the challenge.",
                              "Validation ignores missing pollutant labels; the hidden test labels are unavailable."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def submission(zip_path: Path, weather_path: Path, method: str, output: Path) -> Path:
    train, test, sample = load_competition(zip_path)
    weather = load_weather(weather_path) if method == "lightgbm_weather" else None
    prediction = forecast(method, train, test.index, weather)
    result = sample.copy()
    for target in TARGETS:
        result[target] = prediction[target].to_numpy()
    if result.isna().any().any() or len(result) != HORIZON_HOURS:
        raise ValueError("Submission has missing predictions or an invalid row count")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
